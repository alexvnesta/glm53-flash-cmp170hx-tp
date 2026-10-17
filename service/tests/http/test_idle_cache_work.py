import asyncio
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from glm_async import responsive_generator_class
from glm_b2_native import B2Retention, StopSharedGeneration


class Engine:
    def __init__(self):
        self.backlog = True
        self.started, self.release = threading.Event(), threading.Event()
        self.threads = []

    def has_idle_work(self):
        return self.backlog

    def idle_work(self):
        self.threads.append(threading.get_ident())
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError('test did not release idle worker')
        self.backlog = False

    def iterate(self):
        self.threads.append(threading.get_ident())
        return [42]


class Base:
    def __init__(self, engine):
        self.generator = engine
        self.jobs = {}
        self.condition = asyncio.Condition()
        self.delivered = asyncio.Event()
        self.error = None
        self.iteration_task = asyncio.create_task(self._run_iteration())

    def deliver_results(self, results):
        assert results == [42]
        self.jobs.clear()
        self.delivered.set()

    async def close(self):
        self.iteration_task.cancel()
        await self.iteration_task


class IdlePersistenceContracts(unittest.IsolatedAsyncioTestCase):
    async def test_last_job_cancel_flushes_on_step_worker_and_drains_repeated_cancellation(self):
        loop_thread = threading.get_ident()
        started, release = threading.Event(), threading.Event()
        threads, notifications = [], []
        sentinel = object()

        class Owner:
            @contextmanager
            def step(self):
                threads.append(threading.get_ident())
                self.active = True
                try:yield
                finally:self.active = False

        owner = Owner()

        class CancellingEngine:
            _glm_native_concurrency = owner

            def cancel(self, job):
                self.assert_owner = owner.active
                started.set()
                if not release.wait(3):raise TimeoutError('cancel flush not released')
                assert owner.active

        engine = CancellingEngine()
        class CancelBase:
            def __init__(self, engine):
                self.generator, self.jobs, self.error = engine, {}, None
                self.iteration_task = asyncio.create_task(asyncio.Event().wait())

            async def close(self):
                self.iteration_task.cancel()
                try:await self.iteration_task
                except asyncio.CancelledError:pass

        server = responsive_generator_class(CancelBase, sentinel)(engine)
        step_thread = await asyncio.get_running_loop().run_in_executor(server._gpu_worker, threading.get_ident)
        job = SimpleNamespace(job=object(), put_result=lambda value: notifications.append((threading.get_ident(), value)))
        server.jobs[job.job] = job
        try:
            task = asyncio.create_task(server.cancel(job))
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertTrue(server._step_lock.locked())
            self.assertIn(job.job, server.jobs)
            release.set()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assertNotEqual(threads[0], loop_thread)
            self.assertEqual(threads[0], step_thread)
            self.assertTrue(engine.assert_owner)
            self.assertFalse(owner.active)
            self.assertFalse(server._step_lock.locked())
            self.assertNotIn(job.job, server.jobs)
            self.assertEqual(notifications, [(loop_thread, sentinel)])
        finally:
            release.set()
            await server.close()

    def test_uncertain_b2_custody_aborts_disk_before_raising(self):
        aborted = threading.Event()
        owner = B2Retention.__new__(B2Retention)
        owner.failure = None
        owner.generator = SimpleNamespace(disk_page_cache=SimpleNamespace(abort=aborted.set))
        fault = RuntimeError('injected custody failure')
        with self.assertRaises(StopSharedGeneration):
            owner._terminal(fault)
        self.assertTrue(aborted.is_set())
        self.assertIs(owner.failure, fault)

    async def test_idle_flush_keeps_loop_responsive_and_uses_same_worker_as_generation(self):
        engine = Engine()
        server = responsive_generator_class(Base)(engine)
        try:
            self.assertTrue(await asyncio.to_thread(engine.started.wait, 2))
            self.assertNotEqual(engine.threads[0], threading.get_ident())
            # A request can be registered while disk work is in flight; the HTTP loop stays available.
            server.jobs['job'] = object()
            async with server.condition:
                server.condition.notify_all()
            await asyncio.sleep(.01)
            self.assertFalse(server.delivered.is_set())
            engine.release.set()
            await asyncio.wait_for(server.delivered.wait(), 2)
            self.assertEqual(len(set(engine.threads)), 1)
            self.assertIsNone(server.error)
        finally:
            engine.release.set()
            await server.close()

    async def test_close_drains_running_idle_work(self):
        engine = Engine()
        server = responsive_generator_class(Base)(engine)
        self.assertTrue(await asyncio.to_thread(engine.started.wait, 2))
        closing = asyncio.create_task(server.close())
        await asyncio.sleep(.01)
        self.assertFalse(closing.done())
        engine.release.set()
        await asyncio.wait_for(closing, 2)

    async def test_idle_failure_disables_disk_writes_and_latches_engine(self):
        aborted = threading.Event()
        class Broken(Engine):
            def idle_work(self):
                raise RuntimeError('injected disk scheduling failure')
        engine = Broken()
        engine.disk_page_cache = SimpleNamespace(abort=aborted.set)
        server = responsive_generator_class(Base)(engine)
        try:
            await asyncio.wait_for(server.iteration_task, 2)
            self.assertTrue(aborted.is_set())
            self.assertIsInstance(server.error, RuntimeError)
            self.assertFalse(server.jobs)
        finally:
            await server.close()


if __name__=='__main__':unittest.main()
