"""Execute pinned native scheduler/async/verification Python with CPU doubles.

This avoids importing the CUDA extension and preserves actual method bodies.
It does not qualify CUDA kernels, cache writes or model continuation quality.
"""
import ast,asyncio,os,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import torch
ROOT=Path(__file__).resolve().parents[2]
ENGINE=Path(os.environ.get('EXLLAMAV3_ENGINE_ROOT',ROOT.parent/'engine'))
sys.path.insert(0,str(ROOT/'glm'))
from glm_concurrent_async import concurrent_generator_class
from glm_async import responsive_generator_class


def source_class(relative,name,methods=None,extra=None):
    source=ast.parse((ENGINE/relative).read_text())
    node=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name==name)
    if methods is not None:
        node.body=[m for m in node.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and m.name in methods]
    for child in ast.walk(node):
        if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):child.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    env=dict(torch=torch,asyncio=asyncio,PAGE_SIZE=256,ThreadPoolExecutor=ThreadPoolExecutor,
             cuda_sync_active=lambda:None,logger=NS(error=lambda *a,**kw:None))
    env.update(extra or {})
    exec(compile(ast.fix_missing_locations(module),str(ENGINE/relative),'exec'),env)
    return env[name],env


def engine_paired_state():
    """P16: upstream dev (7de419b) wraps each job's recurrent state in PairedState(target, draft). The engine's own
    class, compiled from source unchanged (its properties keep their decorators). Stage 1 has no draft side."""
    path=ENGINE/'exllamav3/cache/recurrent.py'
    node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='PairedState')
    env={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env)
    return env['PairedState']


def paired(target, draft=None):
    return engine_paired_state()(target, draft)

class FairScheduling(unittest.TestCase):
    def fixture(self,jobs,rounds=1,chunk=None):
        cls,_=source_class('exllamav3/generator/generator.py','Generator',['iterate'])
        g=cls();g.cache=NS(initialized=True,owner_serial=1);g.cache_owner_serial=1;g.draft_cache=None
        g.active_jobs=jobs;g.draft_model=None;g.recurrent_cache=object();g.ngram_match_min=0;g.visualizer=None
        g.fair_gen_rounds=rounds;g.fair_chunk_size=chunk;g.iterate_start_jobs=lambda r:None
        g.hybrid_draft_tokens=0;g.disk_page_cache=None   # P16: upstream iterate reads both
        g.calls=[];g.recurrent_checkpoint=lambda:g.calls.append('checkpoint')
        g.iterate_gen=lambda r,d=None:g.calls.append('gen');g.reap_failed_job=lambda j,e,r:g.active_jobs.remove(j)
        return g
    def job(self,done,progress=True):
        class J:
            def __init__(self):self.done=done;self.chunks=[]
            def is_prefill_done(self):return self.done
            def prefill(self,results,chunk=None):
                self.chunks.append(chunk)
                if not self.done and progress:results.append(dict(stage='prefill'))
        return J()
    def test_default_one_round(self):
        g=self.fixture([self.job(True),self.job(False)]);g.iterate();self.assertEqual(g.calls,['checkpoint','gen'])
    def test_overlap_runs_four_checkpoints_and_rounds(self):
        a,b=self.job(True),self.job(False);g=self.fixture([a,b],4,512);g.iterate()
        self.assertEqual(g.calls,['checkpoint','gen']*4);self.assertEqual(b.chunks,[512])
    def test_cache_walk_only_does_not_gain_rounds(self):
        g=self.fixture([self.job(True),self.job(False,False)],4,512);g.iterate();self.assertEqual(g.calls,['checkpoint','gen'])
    def test_extra_rounds_stop_when_streaming_job_leaves(self):
        a,b=self.job(True),self.job(False);g=self.fixture([a,b],4,512)
        def decode(results,draft=None):g.calls.append('gen');g.active_jobs.remove(a)
        g.iterate_gen=decode;g.iterate();self.assertEqual(g.calls,['checkpoint','gen'])
    def test_failed_prefill_does_not_remove_other_sequence(self):
        a,b=self.job(True),self.job(False);g=self.fixture([a,b],4,512)
        b.prefill=lambda *_:(_ for _ in ()).throw(ValueError('one prefill'))
        g.iterate();self.assertEqual(g.active_jobs,[a]);self.assertEqual(g.calls,['checkpoint','gen'])

class BatchedVerification(unittest.TestCase):
    def test_compact_rows_and_independent_acceptance_rewind(self):
        cls,_=source_class('exllamav3/generator/generator.py','Generator',['iterate_gen'])
        class State:
            def __init__(self,name):self.name=name;self.rewinds=[]
            def rewind(self,n):self.rewinds.append(n)
        class J:
            def __init__(self,name,done,ids):
                self.name=name;self.done=done;self.ids=ids;self.time_first_token=1;self.embeddings=[]
                self.recurrent_state=paired(State(name));self.new_tokens=0;self.filters=[];self.checkpoint_rewound=False
                self.rejected_draft_tokens=0;self.accepted_draft_tokens=0;self.samples=[]
                self.filter_futures=[];self.logit_masks=[]
                self.sequences=[NS(block_index_tensor=torch.tensor([[ids,ids+1]],dtype=torch.int32),
                                  kv_position=0,allocated_pages=[NS(kv_position=3)],sequence_ids=[1])]
            def is_prefill_done(self):return self.done
            def get_max_seq_len(self):return 2
            def get_input_ids_list(self,draft,idx,add_to_cache):
                return [torch.cat([torch.tensor([[self.ids]]),draft[idx:idx+1]],dim=1)]
            def prepare_logit_mask(self):pass
            def prepare_sampling_past_ids(self):pass
            def receive_logits(self,logits):return torch.argmax(logits,dim=-1),None,None,None
            def receive_sample(self,logits,token,*args):
                self.samples.append(token.item());self.sequences[0].kv_position+=1
                return False,token,False
            def is_checkpoint_boundary(self):return False
        a,p,b=J('A',True,7),J('prefill',False,30),J('B',True,15)
        g=cls();g.active_jobs=[a,p,b];g.num_draft_tokens=2;g.recurrent_cache=object();g.dflash_draft=True
        g.mtp_draft=False;g.draft_calibrator=None;g.record_draft_stats=False;g.cache=object();g.draft_cache=object()
        g.target_recurrent=True;g.draft_recurrent=False   # P16: upstream iterate_gen keys batch states on these
        g._staging=lambda name,rows,width=None,dtype=torch.int32:torch.zeros((rows,width) if width is not None else(rows,),dtype=dtype)
        observations=[]
        logits=torch.full((2,3,120),-100.0)
        for row,ids in enumerate(((1,2,3),(99,5,6))):
            for col,token in enumerate(ids):logits[row,col,token]=100
        def forward(input_ids,params):
            observations.append((input_ids.clone(),list(params['recurrent_states']),params['block_table'].clone()))
            params['export_states']=[torch.zeros(2,3,2)];return logits
        g.model=NS(caps={},forward=forward,prefetch_tokens=lambda _:None)
        g.draft_model=NS(draft_verifier_params={},update_kv_from_target=lambda **kw:observations.append(kw))
        g.num_remaining_jobs=lambda:len(g.active_jobs)
        g.iterate_gen([],torch.tensor([[1,2],[4,5]]))
        self.assertEqual(observations[0][0].tolist(),[[7,1,2],[15,4,5]])
        self.assertEqual([s.name for s in observations[0][1]],['A','B'])
        self.assertEqual(observations[0][2][:,:2].tolist(),[[7,8],[15,16]])
        self.assertEqual(a.samples,[1,2,3]);self.assertEqual(b.samples,[99])
        self.assertEqual(a.recurrent_state.target.rewinds,[0]);self.assertEqual(b.recurrent_state.target.rewinds,[2])
        self.assertEqual(observations[1]['lengths'],[3,1]);self.assertEqual(g.active_jobs,[a,p,b])



class ActualRecurrentOwnership(unittest.TestCase):
    def fixture(self):
        from collections import deque
        cache_cls,_=source_class('exllamav3/cache/cache.py','Cache',['get_new_state'])
        cache=cache_cls();cache.free_list=deque([0,1]);cache.values=torch.zeros(2,4)
        class State:
            def __init__(self,cache,handle,position,clear):
                self.cache=cache;self.handle=handle;self.position=position;self.freed=False
                if clear:cache.values[handle].zero_()
            def free(self):
                if not self.freed:self.cache.free_list.append(self.handle);self.freed=True
        cache.recurrent_state_cls=State
        cls,_=source_class('exllamav3/generator/job.py','Job',
                           ['allocate_pages','deallocate_pages','free_recurrent_state'])
        released=[];pt=NS(deallocate_pages=lambda pages:released.append(tuple(pages)))
        # P16: upstream Job.allocate_pages takes the pair from Generator.new_recurrent_state (engine source)
        gen_cls,_=source_class('exllamav3/generator/generator.py','Generator',
                               ['new_recurrent_state','recurrent_state_from_stashed'],
                               dict(PairedState=engine_paired_state()))
        generator=gen_cls();generator.cache=cache;generator.recurrent_cache=object()
        generator.target_recurrent=True;generator.draft_recurrent=False;generator.draft_cache=None
        def make(serial):
            class Seq:
                max_cached_pages=0
                def __init__(self):self.allocated_pages=[serial]
                def allocate_pages(self,pt,rc,protected):
                    self.asserted=(pt,rc,protected)
                    return 1,0,0,None
            j=cls();j.generator=generator;j.pagetable=pt
            j.sequences=[Seq()];j.all_unique_hashes=[];j.recurrent_state=None
            j.cached_pages=j.total_pages=j.non_sequential_pages=0;j.serial_number=serial
            return j
        return cache,make,released
    def test_native_fresh_slots_and_cancelled_slot_reuse_clear_only_own_state(self):
        cache,make,released=self.fixture();a,b=make(1),make(2)
        a.allocate_pages();b.allocate_pages()
        self.assertNotEqual(a.recurrent_state.target.handle,b.recurrent_state.target.handle)
        ah=a.recurrent_state.target.handle;bh=b.recurrent_state.target.handle
        cache.values[ah].fill_(11);cache.values[bh].fill_(22)
        a.deallocate_pages();c=make(3);c.allocate_pages()
        self.assertEqual(c.recurrent_state.target.handle,ah)
        self.assertTrue(torch.equal(cache.values[ah],torch.zeros(4)))
        self.assertTrue(torch.equal(cache.values[bh],torch.full((4,),22.)))
        self.assertEqual(c.cached_pages,0);self.assertEqual(released,[(1,)])
    def test_actual_reap_releases_failed_job_without_freeing_survivor(self):
        cache,make,released=self.fixture();a,b=make(1),make(2)
        a.allocate_pages();b.allocate_pages();cache.values[b.recurrent_state.target.handle].fill_(22)
        cls,_=source_class('exllamav3/generator/generator.py','Generator',['reap_failed_job'])
        g=cls();g.active_jobs=[a,b];error=ValueError('one-job prefill');results=[]
        g.reap_failed_job(a,error,results)
        self.assertEqual(g.active_jobs,[b]);self.assertIsNone(a.recurrent_state)
        self.assertEqual(list(cache.free_list),[0]);self.assertIs(results[0]['error'],error)
        self.assertTrue(torch.equal(cache.values[b.recurrent_state.target.handle],torch.full((4,),22.)))


class ActualConstructor(unittest.TestCase):
    def test_native_defaults_true_then_owned_native_bind_false(self):
        """The engine's own Generator.__init__ with a recurrent (sliding-window) DFlash2 drafter, then the guard."""
        from glm_native_concurrency import bind_native_concurrency
        from test_native_concurrency import generator as native_generator
        fixture = native_generator()
        fixture.cache.reset_states = lambda: None
        fixture.draft_cache.reset_states = lambda: None
        fixture.draft_cache.max_num_tokens = fixture.cache.max_num_tokens
        fixture.draft_cache.max_history = fixture.cache.max_history = 16
        fixture.model.config = NS(vocab_size=32)
        fixture.model.caps = {'recurrent_states': True}
        draft = NS(caps={'dflash_draft': True, 'default_draft_size': 7, 'attach_target': True, 'recurrent_states': True},
                   attach_to=lambda _: None)
        fixture.draft_cache.model = draft
        class TorchCPU:
            def __getattr__(self, name): return getattr(torch, name)
            def empty(self, *args, **kw):
                kw.pop('pin_memory', None); return torch.empty(*args, **kw)
        class PT:
            def __init__(self, g, cache):
                self.generator = g; self.cache = cache; self.max_pages = 128; self.cpu_tier = None
                self.all_pages = [NS(page_index=i) for i in range(128)]
        made = []
        def recurrent_cache(model, size, draft_model):
            made.append((model, size, draft_model)); return NS()
        cls, _ = source_class('exllamav3/generator/generator.py', 'Generator', ['__init__', 'num_remaining_jobs'],
                              dict(torch=TorchCPU(), PageTable=PT, RecurrentCache=recurrent_cache))
        g = cls(fixture.model, fixture.cache, None, max_batch_size=2, draft_model=draft,
                draft_cache=fixture.draft_cache, num_draft_tokens=7)
        try:
            self.assertTrue(g.enable_defrag); self.assertTrue(g.draft_recurrent)
            self.assertEqual(made[0][2], draft)                   # one RecurrentCache holds both sides
            m = bind_native_concurrency(g, 2)
            self.assertFalse(g.enable_defrag); self.assertIs(g._glm_native_concurrency, m)
        finally: g.filter_pool.shutdown(wait=True)

class ActualAsync(unittest.IsolatedAsyncioTestCase):
    async def test_real_responsive_executor_defers_enqueue_until_step_done(self):
        start,finish=threading.Event(),threading.Event()
        class Sync:
            def __init__(self,*args,**kw):self.jobs=[];self.closed=False;self.calls=0
            def enqueue(self,j):self.jobs.append(j)
            def cancel(self,j):
                if j in self.jobs:self.jobs.remove(j)
            def iterate(self):
                self.calls+=1
                if self.calls==1:start.set();finish.wait(3)
                time.sleep(.001)
                return []
            def close(self):self.closed=True
        sentinel=object()
        cls,env=source_class('exllamav3/generator/async_generator.py','AsyncGenerator',extra=dict(Generator=Sync,_CANCELLED_SENTINEL=sentinel))
        g=concurrent_generator_class(responsive_generator_class(cls),sentinel)()
        class J:
            def __init__(self):self.job=object();self.results=[]
            def put_result(self,r):self.results.append(r)
        a,b=J(),J()
        g.enqueue(a)
        for _ in range(100):
            if start.is_set():break
            await asyncio.sleep(.005)
        self.assertTrue(start.is_set());g.enqueue(b);await asyncio.sleep(.01)
        self.assertNotIn(b.job,g.generator.jobs)
        finish.set()
        for _ in range(100):
            if b.job in g.generator.jobs:break
            await asyncio.sleep(.005)
        self.assertIn(b.job,g.generator.jobs)
        await g.cancel(a);self.assertIn(b.job,g.generator.jobs)
        await g.close();self.assertTrue(g.generator.closed)
        self.assertTrue(all(t.name!='glm-gpu_0' for t in threading.enumerate()))

if __name__=='__main__':unittest.main()
