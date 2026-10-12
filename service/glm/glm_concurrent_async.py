"""Serialize native enqueue and cancellation with the persistent GPU step owner."""
import asyncio


def concurrent_generator_class(responsive_base, cancellation_sentinel=None):
    if cancellation_sentinel is None:
        from exllamav3.generator.async_generator import _CANCELLED_SENTINEL
        cancellation_sentinel = _CANCELLED_SENTINEL
    class ConcurrentAsyncGenerator(responsive_base):
        def __init__(self, *args, **kwargs):
            self._enqueue_tasks = {}
            self._concurrent_closing = False
            super().__init__(*args, **kwargs)

        def enqueue(self, job):
            if self._concurrent_closing:
                raise RuntimeError("Concurrent generator is closing")
            if job in self._enqueue_tasks or job.job in self.jobs:
                raise ValueError("Async job already belongs to this generator")

            async def register():
                try:
                    async with self._step_lock:
                        if self._concurrent_closing:
                            raise RuntimeError("Concurrent generator is closing")
                        # This invokes the native queue mutation only after the
                        # prior uncancellable executor step has settled.
                        super(ConcurrentAsyncGenerator, self).enqueue(job)
                except asyncio.CancelledError:
                    return
                except Exception as error:
                    job.put_result(error)
                finally:
                    self._enqueue_tasks.pop(job, None)
            self._enqueue_tasks[job] = asyncio.create_task(register())

        async def cancel(self, job):
            task = self._enqueue_tasks.get(job)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._enqueue_tasks.pop(job, None)
                job.put_result(cancellation_sentinel)
            # Responsive.cancel owns the same lock before touching native state.
            await super().cancel(job)

        async def close(self):
            self._concurrent_closing = True
            pending = list(self._enqueue_tasks.items())
            tasks = [task for _, task in pending]
            for job, task in pending:
                task.cancel()
                job.put_result(cancellation_sentinel)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._enqueue_tasks.clear()
            await super().close()
    return ConcurrentAsyncGenerator
