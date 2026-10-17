"""Keep native EXL3 GPU steps off the HTTP event loop; engine sources stay intact."""
import asyncio
from concurrent.futures import ThreadPoolExecutor


def responsive_generator_class(base, cancellation_sentinel=None):
    class ResponsiveAsyncGenerator(base):
        def __init__(self, *args, **kwargs):
            self._gpu_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glm-gpu")
            self._step_lock = asyncio.Lock()
            try:
                super().__init__(*args, **kwargs)
            except BaseException:
                self._gpu_worker.shutdown(wait=True)
                raise

        async def _run_iteration(self):
            try:
                while True:
                    async with self.condition:
                        while not self.jobs:
                            if getattr(self.generator, 'has_idle_work', lambda: False)():
                                try:
                                    await asyncio.wait_for(self.condition.wait(), timeout=0.25)
                                except asyncio.TimeoutError:
                                    break
                            else:
                                await self.condition.wait()
                    async with self._step_lock:
                        operation = self.generator.iterate if self.jobs else self.generator.idle_work
                        future = asyncio.get_running_loop().run_in_executor(
                            self._gpu_worker, operation)
                        try:
                            results = await asyncio.shield(future)
                        except asyncio.CancelledError:
                            # Do not release the lock while an uncancellable GPU
                            # step still reads the job/page table being cancelled.
                            await future
                            raise
                        if results is not None:
                            self.deliver_results(results)
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                return
            except Exception as error:
                self.error = error
                # Match upstream's failure policy: uncertain cache contents must
                # never be persisted for another process, including idle-I/O faults.
                disk = getattr(self.generator, "disk_page_cache", None)
                if disk is not None:
                    disk.abort()
                for job in self.jobs.values():
                    job.put_result(error)
                self.jobs.clear()

        async def cancel(self, job):
            async with self._step_lock:
                owner = getattr(self.generator, "_glm_native_concurrency", None)
                if owner is None:
                    return await super().cancel(job)
                interrupted = False
                if self.error is None:
                    def cancel_native():
                        # Removing the last job can flush completed KV pages to
                        # disk. That is a native transfer, just like iterate and
                        # idle_work, and must use their persistent thread/guard.
                        with owner.step():
                            self.generator.cancel(job.job)
                    future = asyncio.get_running_loop().run_in_executor(self._gpu_worker, cancel_native)
                    while True:
                        try:
                            await asyncio.shield(future)
                            break
                        except asyncio.CancelledError:
                            # Keep the lock until the uncancellable transfer has
                            # settled, even if this caller is cancelled again.
                            interrupted = True
                # Preserve upstream AsyncGenerator.cancel's queue semantics on
                # the HTTP loop; asyncio.Queue notifications are not thread safe.
                if job.job in self.jobs:
                    del self.jobs[job.job]
                    sentinel = cancellation_sentinel
                    if sentinel is None:
                        from exllamav3.generator.async_generator import _CANCELLED_SENTINEL
                        sentinel = _CANCELLED_SENTINEL
                    job.put_result(sentinel)
                if interrupted:
                    raise asyncio.CancelledError

        async def close(self):
            try:
                await super().close()
            finally:
                await asyncio.to_thread(self._gpu_worker.shutdown, wait=True,
                                        cancel_futures=True)

    return ResponsiveAsyncGenerator
