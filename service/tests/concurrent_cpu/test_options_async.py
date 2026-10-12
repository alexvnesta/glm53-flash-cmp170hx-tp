import asyncio
import os
from unittest.mock import patch
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'glm'))
from glm_concurrent_options import concurrent_options,CapacityGate
from glm_concurrent_async import concurrent_generator_class
from glm_async import responsive_generator_class
from glm_admission import RequestAdmission,AdmissionPolicy


def args(**changes):
    values=dict(experimental_concurrent_generation=True, tensor_parallel=True,
                experimental_dflash2_tp=True,draft_model_dir='synthetic',mtp=False,
                autosplit_max_batch_size=2,cache_size=393216)
    values.update(changes)
    return NS(**values)

class Options(unittest.TestCase):
    def test_default_is_original_single(self):
        a=NS();o=concurrent_options(a,2048)
        self.assertEqual((o.enabled,o.max_sequences,o.generator_kwargs()),(False,1,{}))
    def test_explicit_capacity_and_fairness(self):
        o=concurrent_options(args(),2048)
        self.assertEqual(o.generator_kwargs(),dict(fair_gen_rounds=4,fair_chunk_size=512))
    def test_unsupported_configs_before_allocation(self):
        failures=[dict(autosplit_max_batch_size=1),dict(dflash_session_cache=True),
                  dict(dflash_prefix_cache=True),dict(target_cpu_cache_gib=1),
                  dict(cpu_cache_size=1),dict(tensor_parallel=False),dict(experimental_dflash2_tp=False),
                  dict(mtp=True),dict(num_draft_tokens=4),dict(concurrent_max_sequences=True),dict(concurrent_max_sequences=9),
                  dict(concurrent_fair_gen_rounds=0),dict(concurrent_fair_gen_rounds=True),
                  dict(concurrent_fair_chunk_size=257),dict(concurrent_fair_chunk_size=4096),
                  dict(cache_size=8192)]
        for patch in failures:
            with self.subTest(patch=patch),self.assertRaises(ValueError):concurrent_options(args(**patch),2048)
    def test_active_host_enabled_refused_without_changing_default_off(self):
        with patch.dict(os.environ,{'GLM53_TP_ACTIVE_HOST':'1'}):
            with self.assertRaises(ValueError):concurrent_options(args(),2048)
            self.assertFalse(concurrent_options(NS(),2048).enabled)
    def test_disabled_tuning_refused(self):
        with self.assertRaises(ValueError):concurrent_options(args(experimental_concurrent_generation=False,
                                                                 concurrent_max_sequences=2),2048)

class AsyncContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sentinel=object()
        class Base:
            def __init__(self):
                self._step_lock=asyncio.Lock();self.jobs={};self.events=[];self.error=None
            def enqueue(self,j):
                self.events.append(('enqueue',j.name));self.jobs[j.job]=j
            async def cancel(self,j):
                async with self._step_lock:
                    self.events.append(('cancel',j.name));self.jobs.pop(j.job,None)
            async def close(self):self.events.append(('close',))
        self.g=concurrent_generator_class(Base,self.sentinel)()
    def job(self,name):
        return NS(name=name,job=object(),put_result=lambda r:self.results.append((name,r)))
    async def test_no_enqueue_mutation_during_step(self):
        self.results=[]
        class AsyncJobDouble:
            def __init__(self,name):self.name=name;self.job=object();self.results=[]
            def put_result(self,result):self.results.append(result)
        a,b=AsyncJobDouble('A'),AsyncJobDouble('B')
        await self.g._step_lock.acquire()
        self.g.enqueue(a);self.g.enqueue(b);await asyncio.sleep(0)
        self.assertEqual(self.g.jobs,{})
        self.g._step_lock.release();await asyncio.sleep(0);await asyncio.sleep(0)
        self.assertEqual(len(self.g.jobs),2)
    async def test_cancel_before_task_starts_wakes_only_that_job(self):
        class J:
            def __init__(self,name):self.name=name;self.job=object();self.results=[]
            def put_result(self,r):self.results.append(r)
        a,b=J('A'),J('B');self.g.enqueue(a);self.g.enqueue(b)
        await self.g.cancel(a);await asyncio.sleep(0)
        self.assertEqual(a.results,[self.sentinel]);self.assertIn(b.job,self.g.jobs)
        self.assertNotIn(a,self.g._enqueue_tasks)
    async def test_close_wakes_unregistered_consumers(self):
        class J:
            name='A';job=object()
            def __init__(self):self.results=[]
            def put_result(self,r):self.results.append(r)
        j=J();self.g.enqueue(j);await self.g.close()
        self.assertEqual(j.results,[self.sentinel]);self.assertFalse(self.g._enqueue_tasks)
        with self.assertRaises(RuntimeError):self.g.enqueue(j)
    async def test_registration_failure_is_one_job_error(self):
        class J:
            name='bad';job=object()
            def __init__(self):self.results=[]
            def put_result(self,r):self.results.append(r)
        g=self.g
        # Inject into the actual parent operation, after the registration lock.
        g.__class__.__bases__[0].enqueue=lambda *_:(_ for _ in ()).throw(ValueError('bad request'))
        j=J();g.enqueue(j);await asyncio.sleep(0);await asyncio.sleep(0)
        self.assertIsInstance(j.results[0],ValueError);self.assertIsNone(g.error)
    async def test_two_admission_holders_and_single_release(self):
        gate=CapacityGate(2);a=RequestAdmission(AdmissionPolicy(timeout_seconds=1,poll_seconds=.01))
        class Request:
            async def is_disconnected(self):return False
        await a.acquire(gate,Request());await a.acquire(gate,Request())
        self.assertEqual(gate.active,2)
        waiting=asyncio.create_task(a.acquire(gate,Request()));await asyncio.sleep(.02)
        self.assertFalse(waiting.done());gate.release();await waiting
        self.assertEqual(gate.active,2);gate.release();gate.release()
        with self.assertRaises(ValueError):gate.release()

if __name__=='__main__':unittest.main()
