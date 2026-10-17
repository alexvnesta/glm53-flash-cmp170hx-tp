import asyncio
import unittest
from controller import Controller


class Backend:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.finish.set()
        self.fail = None

    async def switch(self, target):
        self.started.set()
        await self.finish.wait()
        self.calls.append(target)
        if self.fail == target:
            raise RuntimeError("deliberate failure")


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_glm_default_does_not_switch(self):
        b = Backend(); c = Controller(b)
        async with c.lease("glm"):
            self.assertEqual(c.inflight, 1)
        self.assertEqual(c.inflight, 0)
        self.assertEqual(b.calls, [])

    async def test_waits_for_entire_existing_stream_and_blocks_new_admissions(self):
        b = Backend(); c = Controller(b)
        first = c.lease("glm"); await first.__aenter__()
        q = c.lease("qwen"); qt = asyncio.create_task(q.__aenter__())
        await asyncio.sleep(0.01)
        g = c.lease("glm"); gt = asyncio.create_task(g.__aenter__())
        await asyncio.sleep(0.01)
        self.assertFalse(qt.done()); self.assertFalse(gt.done()); self.assertEqual(b.calls, [])
        await first.__aexit__(None, None, None)
        await qt
        self.assertEqual(b.calls, ["qwen"]); self.assertFalse(gt.done())
        await q.__aexit__(None, None, None)
        await gt
        self.assertEqual(b.calls, ["qwen", "glm"])
        await g.__aexit__(None, None, None)

    async def test_cancelled_waiter_does_not_leak_or_switch(self):
        b = Backend(); c = Controller(b)
        async with c.lease("glm"):
            task = asyncio.create_task(c.lease("qwen").__aenter__())
            await asyncio.sleep(0.01); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(c.inflight, 1); self.assertEqual(c.waiters, 0)
        self.assertEqual(b.calls, [])

    async def test_cancelled_switch_completes_before_admissions_resume(self):
        b = Backend(); b.finish.clear(); c = Controller(b)
        task = asyncio.create_task(c.lease("qwen").__aenter__())
        await b.started.wait(); task.cancel()
        await asyncio.sleep(0.01)
        self.assertTrue(c.lock.locked())
        b.finish.set()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(c.active, "qwen"); self.assertEqual(c.inflight, 0)
        self.assertFalse(c.switching); self.assertEqual(c.waiters, 0)

    async def test_failed_qwen_load_restores_glm(self):
        b = Backend(); b.fail = "qwen"; c = Controller(b)
        with self.assertRaises(RuntimeError):
            async with c.lease("qwen"): pass
        self.assertEqual(b.calls, ["qwen", "glm"])
        self.assertEqual(c.active, "glm"); self.assertEqual(c.inflight, 0)

    async def test_ttl_never_interrupts_a_request_and_returns_to_glm(self):
        b = Backend(); now = [0]; c = Controller(b, active="qwen", ttl=10, clock=lambda: now[0])
        self.assertFalse(await c.expire()); now[0] = 11
        async with c.lease("qwen"):
            self.assertFalse(await c.expire())
        self.assertFalse(await c.expire()); now[0] = 22
        self.assertTrue(await c.expire()); self.assertEqual(c.active, "glm")

    async def test_observation_does_not_extend_qwen_ttl(self):
        b = Backend(); now = [0]; c = Controller(b, active="qwen", ttl=10, clock=lambda: now[0])
        now[0] = 11
        async with c.lease("qwen", activity=False): pass
        self.assertTrue(await c.expire())

    async def test_request_exception_releases_lease(self):
        b = Backend(); c = Controller(b)
        with self.assertRaises(ValueError):
            async with c.lease("glm"): raise ValueError()
        self.assertEqual(c.inflight, 0)


if __name__ == "__main__": unittest.main()
