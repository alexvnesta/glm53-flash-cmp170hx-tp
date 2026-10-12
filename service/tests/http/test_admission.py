import asyncio
import unittest
from glm_admission import RequestAdmission, AdmissionPolicy, AdmissionError


class Request:
    disconnected = False
    async def is_disconnected(self):
        return self.disconnected


class Tests(unittest.IsolatedAsyncioTestCase):
    def gate(self, size=8):
        return RequestAdmission(AdmissionPolicy(size, 1, .005))

    async def test_fifo_with_one_owner(self):
        lock = asyncio.Lock()
        await lock.acquire()
        gate = self.gate()
        order = []
        async def worker(index):
            await gate.acquire(lock, Request())
            order.append(index)
            await asyncio.sleep(.001)
            lock.release()
        tasks = []
        for i in range(5):
            tasks.append(asyncio.create_task(worker(i)))
            await asyncio.sleep(.01)
        self.assertEqual(gate.waiting, 5)
        lock.release()
        await asyncio.gather(*tasks)
        self.assertEqual(order, list(range(5)))
        self.assertFalse(lock.locked())
        self.assertEqual(gate.statistics()['admitted'], 5)

    async def test_full_queue_rejects_without_taking_lock(self):
        gate, lock = self.gate(1), asyncio.Lock()
        await lock.acquire()
        first = asyncio.create_task(gate.acquire(lock, Request()))
        await asyncio.sleep(.02)
        with self.assertRaises(AdmissionError) as error:
            await gate.acquire(lock, Request())
        self.assertEqual(error.exception.status, 429)
        self.assertTrue(lock.locked())
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertTrue(lock.locked())
        lock.release()
        self.assertEqual(gate.waiting, 0)

    async def test_pending_disconnect_preserves_owner_and_next_waiter(self):
        gate, lock, request = self.gate(), asyncio.Lock(), Request()
        await lock.acquire()
        disconnected = asyncio.create_task(gate.acquire(lock, request))
        await asyncio.sleep(.01)
        request.disconnected = True
        with self.assertRaises(AdmissionError) as error:
            await disconnected
        self.assertEqual(error.exception.status, 499)
        self.assertTrue(lock.locked())
        lock.release()
        await gate.acquire(lock, Request())
        lock.release()
        self.assertEqual(gate.waiting, 0)

    async def test_disconnect_after_successful_acquire_releases_it(self):
        gate, lock = self.gate(), asyncio.Lock()
        class Race(Request):
            def __init__(self):
                self.calls = 0
            async def is_disconnected(self):
                self.calls += 1
                return self.calls == 2
        with self.assertRaises(AdmissionError):
            await gate.acquire(lock, Race())
        self.assertFalse(lock.locked())
        self.assertEqual(gate.waiting, 0)

    async def test_cancel_during_disconnect_check_releases_acquired_lock(self):
        gate, lock, entered = self.gate(), asyncio.Lock(), asyncio.Event()
        class Wait(Request):
            calls = 0
            async def is_disconnected(self):
                self.calls += 1
                if self.calls == 2:
                    entered.set()
                    await asyncio.Event().wait()
                return False
        task = asyncio.create_task(gate.acquire(lock, Wait()))
        await entered.wait()
        self.assertTrue(lock.locked())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(lock.locked())
        self.assertEqual(gate.waiting, 0)

    async def test_timeout_preserves_current_owner(self):
        gate, lock = self.gate(), asyncio.Lock()
        await lock.acquire()
        with self.assertRaises(AdmissionError):
            await gate.acquire(lock, Request())
        self.assertTrue(lock.locked())
        lock.release()
        self.assertEqual(gate.statistics()['timed_out'], 1)
        self.assertEqual(gate.waiting, 0)

    async def test_cancelled_waiter_leaves_fifo_queue(self):
        gate, lock = self.gate(), asyncio.Lock()
        await lock.acquire()
        tasks = [asyncio.create_task(gate.acquire(lock, Request())) for _ in range(3)]
        await asyncio.sleep(.01)
        tasks[1].cancel()
        with self.assertRaises(asyncio.CancelledError):
            await tasks[1]
        lock.release()
        await tasks[0]
        lock.release()
        await tasks[2]
        lock.release()
        self.assertEqual(gate.waiting, 0)

    def test_policy_bounds(self):
        for size in (0, 65, True, 2.5):
            with self.assertRaises(ValueError):
                AdmissionPolicy(max_waiters=size).validate()
        for timeout in (0, 1801, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                AdmissionPolicy(timeout_seconds=timeout).validate()


if __name__ == '__main__':
    unittest.main()
