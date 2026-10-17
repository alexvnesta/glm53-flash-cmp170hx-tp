"""One GPU owner, with request leases held through the last streamed byte."""
import asyncio
import time
from contextlib import asynccontextmanager


class Controller:
    def __init__(self, backend, active="glm", ttl=900, clock=time.monotonic):
        self.backend, self.active, self.ttl, self.clock = backend, active, ttl, clock
        self.inflight = 0
        self.waiters = 0
        self.switching = False
        self.last_activity = clock()
        self.lock = asyncio.Lock()
        self.idle = asyncio.Condition()

    async def _change(self, target):
        self.switching = True
        try:
            async with self.idle:
                await self.idle.wait_for(lambda: self.inflight == 0)
            # Hold the admission lock until the service transaction completes,
            # even if the request that asked for the switch disconnects.
            task = asyncio.create_task(self.backend.switch(target))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                self.active = target
                raise
            self.active = target
        except Exception:
            self.active = None
            try:
                await self.backend.switch("glm")
                self.active = "glm"
            except Exception:
                pass
            raise
        finally:
            self.switching = False

    @asynccontextmanager
    async def lease(self, target, activity=True):
        if target not in (None, "glm", "qwen"):
            raise ValueError("unknown backend")
        if self.waiters >= 16:
            raise RuntimeError("switcher queue full")
        self.waiters += 1
        try:
            async with self.lock:
                if target is None:
                    target = self.active or "glm"
                if self.active != target:
                    await self._change(target)
                async with self.idle:
                    self.inflight += 1
        finally:
            self.waiters -= 1
        try:
            yield
        finally:
            async with self.idle:
                self.inflight -= 1
                if activity:
                    self.last_activity = self.clock()
                self.idle.notify_all()

    async def expire(self):
        if self.active != "qwen" or self.inflight or self.lock.locked():
            return False
        if self.clock() - self.last_activity < self.ttl:
            return False
        async with self.lock:
            if self.active == "qwen" and not self.inflight and self.clock() - self.last_activity >= self.ttl:
                await self._change("glm")
                return True
        return False
