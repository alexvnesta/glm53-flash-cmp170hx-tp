"""A dead owned rank must fail closed while its native executor step is blocked."""
import asyncio
import ast
import multiprocessing
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'glm')]
import native_engine_fakes as nf
import glm_tp_runtime
from glm_async import responsive_generator_class
from glm_concurrent_async import concurrent_generator_class
from glm_b2_native import paired_async_generator_class


class WorkerChecks(unittest.TestCase):
    def test_real_owned_process_exit_and_closed_handle_without_cuda(self):
        child = multiprocessing.get_context('spawn').Process(target=time.sleep, args=(60,))
        child.start()
        model = NS(loaded_tp=True, mp_children=[NS(is_alive=Mock(side_effect=AssertionError('pseudo queried'))), child])
        try:
            self.assertIsNone(glm_tp_runtime.worker_exit_error(model))
            child.terminate()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                error = glm_tp_runtime.worker_exit_error(model)
                if error is not None:
                    break
                time.sleep(.01)
            self.assertIsInstance(error, BrokenPipeError)
            child.join(timeout=5)
            child.close()
            self.assertIsInstance(glm_tp_runtime.worker_exit_error(model), BrokenPipeError)
        finally:
            try:
                if child.is_alive():
                    child.kill(); child.join()
            except ValueError:
                pass
        self.assertFalse(torch.cuda.is_initialized())

    def test_clean_exit_is_also_dead_and_non_tp_is_unchanged(self):
        child = NS(pid=123, exitcode=0, is_alive=Mock(side_effect=AssertionError('exitcode already available')))
        self.assertIsInstance(glm_tp_runtime.worker_exit_error(NS(loaded_tp=True, mp_children=[child])), BrokenPipeError)
        self.assertIsNone(glm_tp_runtime.worker_exit_error(NS(loaded_tp=False, mp_children=[child])))
        self.assertIsNone(glm_tp_runtime.worker_exit_error(None))


class WorkerFailureContracts(unittest.IsolatedAsyncioTestCase):
    async def test_health_latches_registered_pending_and_new_jobs_without_step_lock(self):
        from fastapi import FastAPI, HTTPException
        import httpx
        scenario = nf.Scenario(nf.Engine())
        with scenario.installed():
            native, _, paired = scenario.generator()
            disk = NS(abort=Mock())
            native.disk_page_cache = paired.disk = disk
            native.model.loaded_tp = True
            native.model.mp_children = [NS(pid=100, exitcode=-9)]
            started, release = threading.Event(), threading.Event()
            calls = []

            def blocked_step():
                calls.append('iterate'); started.set()
                self.assertTrue(release.wait(5), 'test failed to release its stand-in step')
                return []

            class Job:
                def __init__(self):
                    self.job, self.results = object(), []
                def put_result(self, value):
                    self.results.append(value)

            a, b, pending, following = Job(), Job(), Job(), Job()

            class Base:
                def __init__(self):
                    self.generator, self.error = native, None
                    self.condition = asyncio.Condition()
                    self.jobs = {a.job: a, b.job: b}
                def enqueue(self, job):
                    raise AssertionError('failed rank reused for enqueue')
                def deliver_results(self, results):
                    raise AssertionError('late native results delivered after quarantine')

            wrapper = paired_async_generator_class(concurrent_generator_class(
                responsive_generator_class(Base, object()), object()))()
            task = cancel_waiter = None
            try:
                with patch.object(native, 'iterate', side_effect=blocked_step), \
                        patch.object(native, 'idle_work', side_effect=AssertionError('idle I/O after failure')):
                    task = asyncio.create_task(wrapper._run_iteration())
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    self.assertTrue(wrapper._step_lock.locked())
                    wrapper.enqueue(pending)
                    await asyncio.sleep(0)
                    self.assertIn(pending, wrapper._enqueue_tasks)
                    cancel_waiter = asyncio.create_task(wrapper.cancel(b))
                    await asyncio.sleep(0)
                    self.assertFalse(cancel_waiter.done())
                    tree = ast.parse((HERE.parents[1] / 'glm/glm_api.py').read_text())
                    names = {'ensure_engine_ready', 'health', 'models', 'complete'}
                    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
                    for node in functions:
                        node.decorator_list = []
                    scope = {'runtime': {'generator': wrapper}, 'HTTPException': HTTPException,
                             'glm_tp_runtime': glm_tp_runtime, 'time': time}
                    exec(compile(ast.Module(body=functions, type_ignores=[]), 'actual_glm_api', 'exec'), scope)
                    app = FastAPI()
                    app.get('/health')(scope['health'])
                    app.get('/v1/models')(scope['models'])
                    async def complete():
                        return await scope['complete'](None, None, False)
                    app.post('/v1/completions')(complete)
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                        response = await asyncio.wait_for(client.get('/health'), .5)
                        self.assertEqual(response.status_code, 503)
                        first = wrapper.error
                        self.assertIsNotNone(paired.failure)
                        self.assertTrue(disk.abort.called)
                        for job in (a, b, pending):
                            self.assertEqual(job.results, [first])
                        self.assertEqual(wrapper.jobs, {})
                        await asyncio.wait_for(cancel_waiter, .75)
                        await asyncio.wait_for(asyncio.gather(*(wrapper.cancel(job) for job in (a, b, pending))), .5)
                        self.assertTrue(wrapper._step_lock.locked(), 'cleanup must not unlock an in-flight step')
                        wrapper.enqueue(following)
                        self.assertEqual(following.results, [first])
                        for method, path in [('get', '/v1/models'), ('post', '/v1/completions')]:
                            with patch.dict(sys.modules, {'exllamav3': None}):
                                r = await asyncio.wait_for(getattr(client, method)(path), .5)
                            self.assertEqual(r.status_code, 503)
                        self.assertIs(wrapper.error, first)
                    release.set()
                    await asyncio.wait_for(task, 2)
                    self.assertEqual(calls, ['iterate'])
            finally:
                release.set()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                if cancel_waiter is not None:
                    await asyncio.gather(cancel_waiter, return_exceptions=True)
                wrapper._gpu_worker.shutdown(wait=True)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
