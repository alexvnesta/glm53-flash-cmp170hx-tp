from pathlib import Path
import os,sys,threading
from types import SimpleNamespace as NS
import unittest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'glm'))
from glm_concurrent_ring import ConcurrentRing,bind_concurrent_rings,verify_engine_sources

class Job:
    def __init__(self,table,prefilled=True):
        self.sequences=[NS(block_index_tensor=torch.tensor([table],dtype=torch.int32),max_cached_pages=None)]
        self.is_requeued=False;self.embeddings=[];self.filters=[];self.banned_strings=[]
        self.prefix_token=None;self.orig_max_rq_tokens=None;self.all_unique_hashes=[b'old']
        self.prefilled=prefilled
    def is_prefill_done(self):return self.prefilled


def generator():
    storage=[torch.zeros(64,256,1,dtype=torch.int32),torch.zeros(64,256,1,dtype=torch.float16)]
    target=NS(num_slots=2,owner_serial=1,max_num_tokens=32768,max_history=7)
    draft_model=NS();draft=NS(model=draft_model,dflash_ring_tokens=8192,dflash_ring_slots=2,max_num_tokens=32768,get_all_tensors=lambda:storage)
    g=NS(dflash_draft=True,model=NS(loaded_tp=True),cache=target,draft_model=draft_model,draft_cache=draft,
         max_batch_size=2,recurrent_cache=NS(),enable_defrag=False,cache_owner_serial=1,
         max_chunk_size=2048,active_jobs=[],pending_jobs=[])
    g.num_remaining_jobs=lambda:len(g.active_jobs)+len(g.pending_jobs)
    g.pagetable=NS(generator=g,cache=target,cpu_tier=None,all_pages=[NS(page_index=i) for i in range(128)])
    return g,storage

class RingContracts(unittest.TestCase):
    def setUp(self):
        self.g,self.storage=generator();self.stream={'id':41}
        self.manager=ConcurrentRing(self.g,2,lambda _:self.stream['id'])
        self.a=Job(list(range(48)));self.b=Job(list(range(80,128)))
    def table(self,jobs):return torch.cat([j.sequences[0].block_index_tensor for j in jobs])
    def test_disjoint_ring_wrap_with_shared_target_pages(self):
        self.b.sequences[0].block_index_tensor=self.a.sequences[0].block_index_tensor.clone()
        self.g.active_jobs=[self.a,self.b]
        with self.manager.step():
            self.manager.claim(self.a);self.manager.claim(self.b)
            ring=self.manager.params({'block_table':self.table([self.a,self.b])},draft_forward=True)['block_table']
            self.assertFalse(set(ring[0].tolist()) & set(ring[1].tolist()))
            self.assertEqual(ring[0,0].item(),ring[0,32].item())
            self.assertEqual(ring[1,0].item(),32)
            self.assertEqual(self.manager.peak_draft_batch,2)
    def test_pending_prefill_uses_own_single_row_partition(self):
        self.b.prefilled=False;self.g.active_jobs=[self.a,self.b]
        with self.manager.step():
            self.manager.claim(self.a);self.manager.claim(self.b)
            with self.manager.prefill(self.b):
                p=self.manager.params({'block_table':self.table([self.b])})
                self.assertEqual(p['block_table'][0,0].item(),32)
            p=self.manager.params({'block_table':self.table([self.a])})
            self.assertEqual(p['block_table'][0,0].item(),0)
    def test_cancel_reuse_does_not_write_other_sequence(self):
        self.g.active_jobs=[self.a,self.b]
        with self.manager.step():
            self.manager.claim(self.a);self.manager.claim(self.b)
            p=self.manager.params({'block_table':self.table([self.a,self.b])})['block_table']
            for row,label in enumerate((11,22)):
                for logical in range(48):self.storage[0][p[row,logical]].fill_(label)
        before=self.storage[0][32:].clone()
        # Cancellation happens under the async step lock on the HTTP thread;
        # release itself only changes CPU metadata after prior work settled.
        self.manager.release(self.a);self.g.active_jobs.remove(self.a)
        c=Job(list(range(40,88)));self.g.active_jobs.append(c)
        with self.manager.step():
            self.manager.claim(c)
            with self.manager.prefill(c):
                p=self.manager.params({'block_table':self.table([c])})['block_table']
                for logical in range(48):self.storage[0][p[0,logical]].fill_(33)
        self.assertTrue(torch.equal(before,self.storage[0][32:]));self.assertEqual(self.manager.reused_slots,1)
    def test_prepare_cold_is_local(self):
        self.manager.prepare(self.a);self.assertEqual(self.a.sequences[0].max_cached_pages,0)
        self.assertEqual(self.a.all_unique_hashes,[]);self.assertIsNone(self.b.sequences[0].max_cached_pages)
    def test_owner_and_page_replacement_refused(self):
        changes=[lambda:setattr(self.g,'model',NS(loaded_tp=True)),
                 lambda:setattr(self.g.cache,'owner_serial',2),
                 lambda:setattr(self.g.pagetable,'all_pages',list(self.g.pagetable.all_pages)),
                 lambda:self.g.pagetable.all_pages.__setitem__(3,NS(page_index=3)),
                 lambda:setattr(self.g,'enable_defrag',True),
                 lambda:setattr(self.g,'draft_model',NS()),
                 lambda:setattr(self.g.model,'_glm_tp_active_host_coordinator',object()),
                 lambda:setattr(self.g,'_glm_tp_active_host_coordinator',object()),
                 lambda:self.storage[0].resize_(64,256,2),
                 lambda:setattr(self.g.pagetable,'cpu_tier',object())]
        for change in changes:
            self.setUp();change()
            with self.subTest(change=change),self.assertRaises(RuntimeError):self.manager.check_owner()
    def test_foreign_row_or_stream_refuses_before_mapping(self):
        self.g.active_jobs=[self.a]
        with self.manager.step():
            self.manager.claim(self.a)
            bad=self.table([self.a]).clone();bad[0,0]+=1
            with self.assertRaises(RuntimeError):self.manager.params({'block_table':bad})
            self.stream['id']=42
            with self.assertRaises(RuntimeError):self.manager.params({'block_table':self.table([self.a])})
    def test_transfer_outside_worker_refused(self):
        with self.assertRaises(RuntimeError):self.manager.params({'block_table':self.table([self.a])})
        with self.manager.step():pass
        errors=[]
        def foreign():
            try:
                with self.manager.step():pass
            except RuntimeError as e:errors.append(e)
        t=threading.Thread(target=foreign);t.start();t.join();self.assertEqual(len(errors),1)
    def test_failure_restores_context_and_depth(self):
        with self.assertRaises(ValueError):
            with self.manager.step():
                with self.manager.prefill(self.a):raise ValueError('prefill failure')
        self.assertEqual(self.manager.depth,0);self.assertIsNone(self.manager.prefill_job)
        with self.manager.step():pass
    def test_native_default_defrag_disable_before_binding(self):
        self.g.enable_defrag=True
        m=bind_concurrent_rings(self.g,2)
        self.assertFalse(self.g.enable_defrag);self.assertIs(self.g.draft_cache._glm_concurrent_ring,m)
        m.close();self.assertFalse(self.g.enable_defrag)
    def test_binding_failure_restores_original_flag(self):
        self.g.enable_defrag=True;self.g.cache.num_slots=1
        with self.assertRaises(ValueError):bind_concurrent_rings(self.g,2)
        self.assertTrue(self.g.enable_defrag);self.assertFalse(hasattr(self.g,'_glm_concurrent_ring'))
    def test_source_pins_exact_engine(self):
        engine=Path(os.environ.get('EXLLAMAV3_ENGINE_ROOT',Path(__file__).resolve().parents[3]/'engine'))
        verify_engine_sources(engine)
        import tempfile
        with tempfile.TemporaryDirectory() as td,self.assertRaises(FileNotFoundError):verify_engine_sources(td)

if __name__=='__main__':unittest.main()
