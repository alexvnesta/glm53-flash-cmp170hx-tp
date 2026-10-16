import ast,copy,importlib.util,json
from pathlib import Path
import time,unittest
ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('qualify_concurrent_generation',ROOT/'scripts/qualify_concurrent_generation.py')
client=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(client)
FIXTURE=json.loads((ROOT/'tests/fixtures/concurrent_generation.json').read_text())

class APISource(unittest.TestCase):
    def test_validation_before_allocation_and_binding_before_yield(self):
        source=ast.parse((ROOT/'glm/glm_api.py').read_text())
        lifespan=next(n for n in source.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='lifespan')
        calls=[n for n in ast.walk(lifespan) if isinstance(n,ast.Call)]
        def line(name):return min(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id==name)
        modelinit=min(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='init')
        self.assertLess(line('concurrent_options'),modelinit)
        self.assertLess(line('install_native_concurrency'),modelinit)
        self.assertGreater(line('bind_native_concurrency'),modelinit)
        self.assertLess(line('bind_native_concurrency'),min(n.lineno for n in ast.walk(lifespan) if isinstance(n,ast.Yield)))
        self.assertLess(line('bind_native_concurrency'),line('bind_b2_retention'))
        text=ast.get_source_segment((ROOT/'glm/glm_api.py').read_text(),lifespan)
        # P16: the single-sequence draft ring is gone; that path refuses -swa_full (upstream bounded SWA state)
        self.assertIn('elif getattr(args, "swa_full", False):',text)
        self.assertNotIn('bounded_draft_cache()',text)
        self.assertIn('max_batch_size=concurrency.max_sequences',text)
    def test_per_response_cleanup_releases_owned_capacity_after_cancel(self):
        s=(ROOT/'glm/glm_api.py').read_text()
        tree=ast.parse(s)
        callback=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='cancel_and_report')
        self.assertTrue(any(isinstance(n,ast.Await) and isinstance(n.value,ast.Call)
                            and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='cancel'
                            for n in ast.walk(callback)))
        self.assertIn('OwnedJobCleanup(cancel_and_report, lock.release)',s)
        self.assertIn('await runtime["admission"].acquire(lock, request)',s)
        self.assertIn('OwnedStreamingResponse',s)
        self.assertIn('getattr(generator, "fair_gen_rounds", 1)',s)
        self.assertIn('getattr(generator, "fair_chunk_size", None)',s)
    def test_bc_batch_statics_before_model_load(self):
        s=(ROOT/'glm/glm_q8_staging.py').read_text()
        self.assertIn('bc_attn.MAX_BSZ = bc_mla.MAX_BSZ = mla_attn._bc_max_bsz = max_batch_size',s)
        self.assertIn('bc_attn.MAX_R = max_batch_size * bc_attn.MAX_QLEN',s)

class ClientContracts(unittest.TestCase):
    def response(self,case):
        oracle=case['oracle'];text=str(oracle['value']) if oracle['kind']=='integer' else '|'.join(oracle['values'])
        return dict(choices=[dict(message=dict(role='assistant',content=text,reasoning_content=''),finish_reason='stop')],usage=dict(prompt_tokens=66,completion_tokens=40,total_tokens=106))
    def test_oracles_refuse_cross_sequence_and_role_tokens(self):
        a,b=FIXTURE['parallel'];self.assertFalse(client.validate(a,self.response(b)))
        r=self.response(a);r['choices'][0]['message']['reasoning_content']='<|assistant|>'
        self.assertFalse(client.validate(a,r))
    def test_count_prefix_partial_last_only(self):
        self.assertTrue(client.valid_count_prefix('1\n2\n3\n4','length'))
        self.assertTrue(client.valid_count_prefix('\n'.join(map(str,range(1,404)))+'\n40','length'))
        self.assertFalse(client.valid_count_prefix('1\n3\n4','length'))
        self.assertFalse(client.valid_count_prefix('1\n2\n<|assistant|>','length'))
    def test_fake_transport_requires_actual_batch_counter_and_cancel_survival(self):
        parent=self
        class Fake:
            timeout=180
            def __init__(self):self.idles=0
            def idle(self,tag):
                self.idles+=1
                return dict(experimental_concurrent_generation=True,target_tp=True,max_active_sequences=2,prompt_cache=None,target_cpu_cache_budget_bytes=0,cache_tokens=393216,context_length=392960,draft_num_tokens=7,dynamic_draft_tokens=False,
                            concurrent_ring=dict(peak_draft_batch=2 if tag=='final' else 0,batched_draft_rounds=3 if tag=='final' else 0,active_ring_leases=0,reused_slots=3 if tag=='final' else 0,peak_target_batch=2 if tag=='final' else 0,batched_verifier_rounds=3 if tag=='final' else 0,mixed_prefill_steps=4 if tag=='final' else 0))
            def post(self,body,tag):
                for case in [*FIXTURE['parallel'],FIXTURE['mixed_prefill'],FIXTURE['slot_reuse']]:
                    if case['request']==body:return parent.response(case)
                raise AssertionError('Unknown exact request')
            def stream(self,body,tag,first,cancel=None):
                first.set()
                if cancel is not None:
                    if not cancel.wait(2):raise AssertionError('Cancellation was not attempted')
                return dict(content='1\n2\n3',reasoning_content='',first_text_ms=1,wall_ms=10,usage={'completion_tokens':3},
                            finish_reason=None if cancel else 'length',client_disconnect=cancel is not None)
        report=client.campaign(Fake(),FIXTURE)
        self.assertTrue(report['qualified']);self.assertTrue(report['whole_choice_equivalence'])
    def test_ledger_body_preserved_from_qualified_fixture(self):
        self.assertEqual(FIXTURE['mixed_prefill']['expected_prompt_tokens'],16528)
        self.assertGreater(sum(len(m['content']) for m in FIXTURE['mixed_prefill']['request']['messages']),40000)
        self.assertNotEqual(FIXTURE['parallel'][0]['oracle']['values'],FIXTURE['parallel'][1]['oracle']['values'])

if __name__=='__main__':unittest.main()
