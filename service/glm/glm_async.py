"""Keep native EXL3 GPU steps off the HTTP event loop; engine sources stay intact."""
import asyncio
from concurrent.futures import ThreadPoolExecutor


def fail_generation(owner, error):
    """Latch on the HTTP loop, without waiting for the GPU step or touching ranks."""
    if owner.error is None:
        owner.error = error
    native = owner.generator
    paired = getattr(native, '_glm_b2_retention', None)
    if paired is not None:
        from glm_b2_native import StopSharedGeneration
        try:
            paired._terminal(owner.error)
        except StopSharedGeneration:
            pass
        if paired.shared is not None:
            paired.shared.tickets.clear()
    else:
        disk = getattr(native, 'disk_page_cache', None)
        if disk is not None:
            disk.abort()
    registered = list(owner.jobs.values())
    for job in registered:
        job.put_result(owner.error)
    owner.jobs.clear()
    try:
        current = asyncio.current_task()
    except RuntimeError:
        current = None
    registered_ids = {id(job) for job in registered}
    for job, task in list(getattr(owner, '_enqueue_tasks', {}).items()):
        # A register() caller already handles its own error and ticket cleanup.
        if task is current:
            continue
        if id(job) not in registered_ids:
            job.put_result(owner.error)
        task.cancel()
    return owner.error


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
                while self.error is None:
                    async with self.condition:
                        while not self.jobs and self.error is None:
                            if getattr(self.generator, 'has_idle_work', lambda: False)():
                                try:
                                    await asyncio.wait_for(self.condition.wait(), timeout=0.25)
                                except asyncio.TimeoutError:
                                    break
                            else:
                                await self.condition.wait()
                    if self.error is not None:
                        return
                    async with self._step_lock:
                        if self.error is not None:
                            return
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
                        if results is not None and self.error is None:
                            self.deliver_results(results)
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                return
            except Exception as error:
                fail_generation(self, error)

        async def cancel(self, job):
            if self.error is not None:
                # A dead rank can leave the executor inside a collective. Its
                # consumers must release HTTP admission without acquiring that
                # step's lock or issuing native cleanup against failed workers.
                self.jobs.pop(job.job, None)
                return
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
