"""Teardown order and failure semantics for the opt-in service path."""
import asyncio
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm'))
from glm_tp_lifecycle import tp_lifecycle, close_tp_service
from glm_async import responsive_generator_class

class TPLifecycle(unittest.IsolatedAsyncioTestCase):
    def objects(self,fail=None):
        events=[]
        def step(name):
            events.append(name)
            if fail==name: raise RuntimeError(name)
        class Model:
            loaded_tp=True
            def tp_drain_acks(self):step('drain')
            def unload(self):step('target_unload')
        class Draft:
            loaded_tp=False
            def unload(self):step('draft_unload')
        class Async:
            generator=NS(_glm_dflash_prefix_cache=NS(close=lambda:step('manager_close')))
            async def close(self):step('async_close')
        return events,Model(),Draft(),Async()
    async def test_explicit_order(self):
        events,model,draft,g=self.objects()
        await close_tp_service(model,draft,g)
        self.assertEqual(events,['async_close','manager_close','drain','target_unload','draft_unload'])
    async def test_startup_failure_before_generator_still_unloads(self):
        events,model,draft,g=self.objects()
        with self.assertRaisesRegex(ValueError,'startup'):
            async with tp_lifecycle(True,model,draft):raise ValueError('startup')
        self.assertEqual(events,['drain','target_unload','draft_unload'])
    async def test_original_error_kept_if_cleanup_fails(self):
        events,model,draft,g=self.objects('target_unload')
        with self.assertRaisesRegex(ValueError,'primary') as caught:
            async with tp_lifecycle(True,model,draft) as owned:
                owned['generator']=g;raise ValueError('primary')
        self.assertIn('target_unload',str(caught.exception.__notes__))
        self.assertEqual(events[-1],'draft_unload')
    async def test_clean_shutdown_surfaces_cleanup_failure(self):
        events,model,draft,g=self.objects('manager_close')
        with self.assertRaisesRegex(RuntimeError,'manager_close'):
            async with tp_lifecycle(True,model,draft) as owned:owned['generator']=g
        self.assertEqual(events[-1],'draft_unload')
    async def test_async_failure_still_attempts_native_cleanup(self):
        events,model,draft,g=self.objects('async_close')
        with self.assertRaisesRegex(RuntimeError,'async_close'):await close_tp_service(model,draft,g)
        self.assertEqual(events[-3:],['drain','target_unload','draft_unload'])
    async def test_ack_failure_does_not_skip_native_destroy(self):
        events,model,draft,g=self.objects('drain')
        with self.assertRaisesRegex(RuntimeError,'drain'):await close_tp_service(model,draft,g)
        self.assertEqual(events[-3:],['drain','target_unload','draft_unload'])
    async def test_cancel_during_join_waits_before_manager_and_models(self):
        events,model,draft,g=self.objects()
        joining=asyncio.Event();joined=asyncio.Event()
        async def close():
            events.append('join_started');joining.set()
            await joined.wait();events.append('join_complete')
        g.close=close
        task=asyncio.create_task(close_tp_service(model,draft,g))
        await joining.wait();task.cancel();await asyncio.sleep(0)
        self.assertEqual(events,['join_started'])
        self.assertFalse(task.done())
        joined.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(events,['join_started','join_complete','manager_close','drain','target_unload','draft_unload'])
    async def test_repeated_cancellation_cannot_detach_join(self):
        events,model,draft,g=self.objects()
        joining=asyncio.Event();joined=asyncio.Event()
        async def close():joining.set();await joined.wait();events.append('joined')
        g.close=close
        task=asyncio.create_task(close_tp_service(model,draft,g))
        await joining.wait()
        for _ in range(3):task.cancel();await asyncio.sleep(0)
        self.assertEqual(events,[])
        joined.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(events[0],'joined')
    async def test_actual_responsive_close_joins_real_executor_before_unload(self):
        events,model,draft,_=self.objects()
        permit=threading.Event();joining=asyncio.Event();loop=asyncio.get_running_loop()
        class Base:
            def __init__(self):self.generator=NS(_glm_dflash_prefix_cache=None)
            async def close(self):pass
        g=responsive_generator_class(Base)()
        future=g._gpu_worker.submit(permit.wait)
        original=g._gpu_worker.shutdown
        def shutdown(*args,**kwargs):
            loop.call_soon_threadsafe(joining.set)
            original(*args,**kwargs);events.append('executor_joined')
        g._gpu_worker.shutdown=shutdown
        task=asyncio.create_task(close_tp_service(model,draft,g))
        try:
            await joining.wait();task.cancel();await asyncio.sleep(0)
            self.assertEqual(events,[]);self.assertFalse(future.done())
        finally:permit.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(events,['executor_joined','drain','target_unload','draft_unload'])
    async def test_disabled_path_has_no_new_cleanup_actions(self):
        events,model,draft,g=self.objects()
        async with tp_lifecycle(False,model,draft) as owned:owned['generator']=g
        self.assertEqual(events,[])
