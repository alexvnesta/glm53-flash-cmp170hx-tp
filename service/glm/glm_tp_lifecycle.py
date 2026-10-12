"""Explicit teardown of an experimental TP service's own workers and models."""
from contextlib import asynccontextmanager
import asyncio
import sys


async def _join_generator_close(generator):
    # ResponsiveAsyncGenerator.close joins its executor in a finally block.
    # Shield that entire coroutine: cancellation of the lifespan must not detach
    # its asyncio.to_thread(shutdown) while a native GPU step is still running.
    task = asyncio.create_task(generator.close())
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    task.result()  # Surface an inner close failure after the task settles.
    if cancellation is not None:
        raise cancellation


async def close_tp_service(model, draft_model, async_generator=None):
    """Join HTTP/GPU executor before releasing rank cache mappings and models.

    A cleanup failure is surfaced after all owned resources have been attempted;
    the original lifespan failure remains primary when there already is one.
    """
    failures = []
    if async_generator is not None:
        try:
            await _join_generator_close(async_generator)
        except BaseException as error:
            failures.append(error)
        manager = getattr(async_generator.generator, "_glm_dflash_prefix_cache", None)
        if manager is not None and callable(getattr(manager, "close", None)):
            try:
                manager.close()
            except BaseException as error:
                failures.append(error)
    for owned in (model, draft_model):
        if owned is None:
            continue
        if getattr(owned, "loaded_tp", False):
            try:
                owned.tp_drain_acks()
            except BaseException as error:
                failures.append(error)
        try:
            owned.unload()
        except BaseException as error:
            failures.append(error)
    if failures:
        for extra in failures[1:]:
            failures[0].add_note(f"Additional TP teardown failure: {extra!r}")
        raise failures[0]


@asynccontextmanager
async def tp_lifecycle(enabled, model, draft_model):
    owned = {"generator": None}
    try:
        yield owned
    finally:
        if enabled:
            primary = sys.exception()
            try:
                await close_tp_service(model, draft_model, owned["generator"])
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(f"TP service teardown also failed: {cleanup_error!r}")
