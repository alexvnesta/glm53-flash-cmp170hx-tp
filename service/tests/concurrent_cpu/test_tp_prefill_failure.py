"""Exercise TP faults through the real prefill loop, hooks and failure sink."""
import asyncio
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'glm')]
import native_engine_fakes as nf
from glm_async import responsive_generator_class
from glm_native_concurrency import tp_execution_failed


class PrefillFaults(unittest.TestCase):
    def test_typed_faults_latch_before_native_reap_or_checkpoint_cleanup(self):
        for error in (torch.OutOfMemoryError('graph allocation'),
                      torch.AcceleratorError('graph instantiation'),
                      TimeoutError('Timed out waiting for worker'),
                      BrokenPipeError('rank exited'), EOFError('rank closed')):
            with self.subTest(error=type(error).__name__):
                s = nf.Scenario(nf.Engine())
                with s.installed():
                    g, manager, coordinator = s.generator()
                    a, b = s.Job(nf.prompt(300), 32), s.Job(nf.prompt(400), 32)
                    g.enqueue(a); g.enqueue(b)
                    # Reach the actual prefill boundary before attaching the
                    # abort-only disk observer (allocation probes disk state).
                    with manager.step():
                        g.iterate_start_jobs([])
                    disk = NS(abort=Mock())
                    g.disk_page_cache = coordinator.disk = disk
                    g.recurrent_cache.disk_tier = disk
                    with patch.object(s.model, 'prefill', side_effect=error), \
                            patch.object(g, 'reap_failed_job', side_effect=AssertionError('unsafe per-job reap')):
                        with self.assertRaisesRegex(RuntimeError, 'shared native custody'):
                            g.iterate()
                    self.assertIs(coordinator.failure, error)
                    disk.abort.assert_called_once()
                    self.assertTrue(coordinator.stats()['terminal'])
                    # A failed rank cannot be touched again by a following iteration.
                    with patch.object(s.model, 'prefill', side_effect=AssertionError('rank reused')):
                        with self.assertRaisesRegex(RuntimeError, 'shared native custody'):
                            g.iterate()
                self.assertFalse(torch.cuda.is_initialized())

    def test_request_validation_failure_keeps_peer_and_generator_usable(self):
        s = nf.Scenario(nf.Engine())
        with s.installed():
            g, _, coordinator = s.generator()
            a, b = s.Job(nf.prompt(300), 32), s.Job(nf.prompt(400), 32)
            g.enqueue(a); g.enqueue(b)
            original = s.model.prefill
            calls = 0

            def prefill(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ValueError('bad request payload')
                return original(*args, **kwargs)

            with patch.object(s.model, 'prefill', side_effect=prefill):
                results = g.iterate()
            self.assertTrue(any(r.get('stage') == 'error' and r['job'] is a for r in results))
            self.assertIsNone(coordinator.failure)
            while g.num_remaining_jobs():
                g.iterate()
            self.assertEqual(len(b.text), 32)
            self.assertFalse(coordinator.stats()['terminal'])

    def test_classifier_does_not_promote_error_text_or_python_memory_errors(self):
        for error in (ValueError('CUDA error: out of memory'), RuntimeError('invalid request'),
                      MemoryError('host memory'), asyncio.CancelledError()):
            self.assertFalse(tp_execution_failed(error))


class AsyncFaults(unittest.IsolatedAsyncioTestCase):
    async def test_latched_native_fault_reaches_all_waiters_and_health_error(self):
        s = nf.Scenario(nf.Engine())
        with s.installed():
            g, _, coordinator = s.generator()
            a, b = s.Job(nf.prompt(300), 32), s.Job(nf.prompt(400), 32)
            g.enqueue(a); g.enqueue(b)
            notifications = [[], []]

            class Base:
                def __init__(self):
                    self.generator, self.error = g, None
                    self.condition = asyncio.Condition()
                    self.jobs = {a: NS(put_result=notifications[0].append),
                                 b: NS(put_result=notifications[1].append)}

            wrapper = responsive_generator_class(Base)()
            try:
                with patch.object(s.model, 'prefill', side_effect=torch.OutOfMemoryError('graph')):
                    await asyncio.wait_for(wrapper._run_iteration(), timeout=2)
                self.assertIsInstance(wrapper.error, RuntimeError)
                self.assertEqual(wrapper.jobs, {})
                self.assertEqual([len(n) for n in notifications], [1, 1])
                self.assertTrue(all(n[0] is wrapper.error for n in notifications))
                self.assertIsNotNone(coordinator.failure)
                from fastapi import FastAPI, HTTPException
                import httpx
                source = ast.parse((HERE.parents[1] / 'glm/glm_api.py').read_text())
                helper = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'ensure_engine_ready')
                handler = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'health')
                handler.decorator_list = []
                scope = {'runtime': {'generator': wrapper}, 'HTTPException': HTTPException}
                exec(compile(ast.Module(body=[helper,handler], type_ignores=[]), 'glm_api.py', 'exec'), scope)
                app = FastAPI()
                app.get('/health')(scope['health'])
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    response = await client.get('/health')
                self.assertEqual(response.status_code, 503)
                self.assertIn('restart', response.json()['detail'])
            finally:
                wrapper._gpu_worker.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
