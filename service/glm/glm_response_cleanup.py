"""One job/lock cleanup shared by an iterator and its entire ASGI response.

The existing API uses asyncio. AnyIO shields level cancellation; an owned
asyncio task also lets repeated Task.cancel calls drain before lock release.
No model, Torch, engine, network or service imports are performed here.
"""
import asyncio
import sys

import anyio
from starlette.responses import StreamingResponse


async def _drain(future):
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
    result = future.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class OwnedJobCleanup:
    """Callbacks belong to exactly one successfully admitted job and held lock.

    Construct immediately after creating that job. The release callback is
    synchronous, e.g. lock.release. The cancellation callback is asynchronous
    and must drain its native operation before returning or raising. A cached
    failure stays observable to every close caller; cancellation is never retried.
    """
    def __init__(self, cancel, release):
        if not callable(cancel) or not callable(release):
            raise TypeError('Owned cancellation and release callbacks required')
        self._cancel, self._release = cancel, release
        self._task = None
        self.cancel_error = None
        self.release_attempted = False
        self.released = False

    async def _finish(self):
        with anyio.CancelScope(shield=True):
            try:
                operation = asyncio.ensure_future(self._cancel())
                await _drain(operation)
            except BaseException as error:
                self.cancel_error = error
                raise
            finally:
                self.release_attempted = True
                self._release()
                self.released = True

    async def close(self):
        # No await occurs between checking and publishing this owned task.
        # Both response/iterator callers run on the same HTTP event loop.
        if self._task is None:
            self._task = asyncio.create_task(self._finish())
        with anyio.CancelScope(shield=True):
            await _drain(self._task)


class OwnedStreamingResponse(StreamingResponse):
    """Cleanup includes response-start/send failures before iterator entry."""
    def __init__(self, content, *, cleanup, **kwargs):
        if not isinstance(cleanup, OwnedJobCleanup):
            raise TypeError('One explicit OwnedJobCleanup required')
        self.owned_cleanup = cleanup
        super().__init__(content, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            primary = sys.exception()
            try:
                await self.owned_cleanup.close()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                if cleanup_error is not primary:
                    primary.add_note(f'Owned job cleanup also failed: {cleanup_error!r}')
                # Preserve the original header/send/cancellation failure after
                # the owned cancellation task is finished and release attempted.
