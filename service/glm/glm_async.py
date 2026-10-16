"""Keep native EXL3 GPU steps off the HTTP event loop; engine sources stay intact."""
import asyncio
from concurrent.futures import ThreadPoolExecutor


def responsive_generator_class(base):
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
                await super().cancel(job)

        async def close(self):
            try:
                await super().close()
            finally:
                await asyncio.to_thread(self._gpu_worker.shutdown, wait=True,
                                        cancel_futures=True)

    return ResponsiveAsyncGenerator
