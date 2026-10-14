"""Experimental target-only TP host page slots; allocation occurs on overflow.

The parent names a whole-page slot. Each rank owns its fixed segment ranges,
registered mappings and current-stream transfer order. No draft cache is attached.

P16: moved unchanged from the engine (exllamav3/util/tp_lazy_cache.py, public-repo engine patch 0005) into the
service; only the two engine imports became absolute. TP ranks import this module by name through the sys.path the
spawned workers inherit (PYTHONPATH and glm_api.py put service/glm on it). Its one engine dependency,
model_tp_fn.mp_cpu_cache_tensors, is unchanged upstream; util/pinned_arena.py is byte-identical.
"""
import threading
import mmap
import torch
from exllamav3.util.pinned_arena import PinnedArena


def _tensors(context, cache_id):
    from exllamav3.model.model_tp_fn import mp_cpu_cache_tensors
    return tuple(mp_cpu_cache_tensors(context, [cache_id]))


def _signature(tensors):
    return tuple((id(t), t.data_ptr(), tuple(t.shape), t.dtype, t.device) for t in tensors)


class LazyRankSlotPool:
    def __init__(self, owner, cache_id, tensors, max_slots, max_pages):
        if type(max_slots) is not int or max_slots < 2:
            raise ValueError("Lazy TP tier requires at least two slots")
        self.owner, self.cache_id = owner, cache_id
        self.tensors = tuple(tensors)
        self.signature = _signature(self.tensors)
        self.segments = []
        offset = 0
        for tensor in self.tensors:
            if (tensor.device.type != "cuda" or tensor.ndim < 2
                    or tensor.shape[0] != max_pages or not tensor.is_contiguous()):
                raise ValueError("Lazy TP tier requires contiguous CUDA page tensors")
            size = tensor[0].numel() * tensor.element_size()
            self.segments.append((offset, tuple(tensor.shape[1:]), tensor.dtype, size))
            offset = (offset + size + 255) & ~255
        self.slab_size = (offset + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        if not self.slab_size:
            raise ValueError("Rank has no target paged tensors")
        self.max_slots = max_slots
        self.slots, self.arenas = {}, {}
        self.streams = None
        self.transfer_thread = None

    def check(self, tensors, transfer=True):
        if _signature(tensors) != self.signature:
            raise ValueError("Lazy TP target tensor ownership/layout changed")
        if not transfer:
            return
        thread = threading.get_ident()
        streams = {t.device.index: torch.cuda.current_stream(t.device) for t in self.tensors}
        keys = {device: stream.cuda_stream for device, stream in streams.items()}
        if self.streams is None:
            self.streams = streams
            self.transfer_thread = thread
        elif (thread != self.transfer_thread
              or keys != {device: stream.cuda_stream for device, stream in self.streams.items()}):
            raise ValueError("Lazy TP tier requires its original serialized thread/current CUDA streams")

    def prepare(self, slot):
        if type(slot) is not int or not 0 <= slot < self.max_slots:
            raise ValueError("Invalid lazy TP slot")
        if slot in self.slots:
            return True
        arena = None
        try:
            arena = PinnedArena(self.slab_size, "Lazy TP target overflow page", required=True)
            views = [arena.tensor[offset:offset + size].view(dtype).view(shape)
                     for offset, shape, dtype, size in self.segments]
        except (RuntimeError, OSError, MemoryError, ValueError):
            if arena is not None:
                arena.close()
            return False
        self.arenas[slot], self.slots[slot] = arena, views
        return True

    def drop(self, slot):
        # Used for a partly prepared fresh slot, and final teardown. Explicit
        # stream completion prevents unregistering a mapping with DMA in flight.
        for stream in (self.streams or {}).values():
            stream.synchronize()
        self.slots.pop(slot, None)
        arena = self.arenas.pop(slot, None)
        if arena is not None:
            arena.close()

    def close(self):
        for slot in tuple(self.arenas):
            self.drop(slot)
        self.tensors = ()


def mp_tp_lazy_init(context, owner, cache_id, max_slots=0, max_pages=None):
    tensors = _tensors(context, cache_id)
    offset = 0
    for tensor in tensors:
        offset = (offset + tensor[0].numel() * tensor.element_size() + 255) & ~255
    size = (offset + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
    if max_slots:
        if context.get("glm_tp_lazy_cache") is not None or context.get("cpu_page_cache") is not None:
            raise ValueError("Rank already has an owned CPU page cache")
        context["glm_tp_lazy_cache"] = LazyRankSlotPool(owner, cache_id, tensors, max_slots, max_pages)
    return size


def _pool(context, owner, cache_id, transfer=True):
    pool = context.get("glm_tp_lazy_cache")
    if pool is None or pool.owner != owner or pool.cache_id != cache_id:
        raise ValueError("Foreign/missing lazy TP page tier")
    pool.check(_tensors(context, cache_id), transfer=transfer)
    return pool


def mp_tp_lazy_prepare(context, owner, cache_id, slot):
    return _pool(context, owner, cache_id).prepare(slot)


def mp_tp_lazy_drop(context, owner, cache_id, slot):
    _pool(context, owner, cache_id, transfer=False).drop(slot)


def mp_tp_lazy_store(context, owner, cache_id, slot, page_index):
    pool = _pool(context, owner, cache_id)
    for buffer, tensor in zip(pool.slots[slot], pool.tensors):
        if not 0 <= page_index < tensor.shape[0]:
            raise ValueError("TP page index outside target pool")
        buffer.copy_(tensor[page_index], non_blocking=True)


def mp_tp_lazy_fetch(context, owner, cache_id, slot, page_index):
    pool = _pool(context, owner, cache_id)
    for buffer, tensor in zip(pool.slots[slot], pool.tensors):
        if not 0 <= page_index < tensor.shape[0]:
            raise ValueError("TP page index outside target pool")
        tensor[page_index].copy_(buffer, non_blocking=True)


def mp_tp_lazy_transfer(context, owner, cache_id, ops):
    """Run (store, slot, page_index) copies queued by the parent, as single-page store/fetch calls would.

    The whole batch is validated before the first copy. Copies are then enqueued on the same fixed current
    stream in exactly the parent's order, so a slot read (fetch) still precedes a later store that reuses it.
    """
    pool = _pool(context, owner, cache_id)
    plan = []
    for store, slot, page_index in ops:
        buffers = pool.slots.get(slot)
        if type(store) is not bool or buffers is None or type(page_index) is not int:
            raise ValueError("Invalid lazy TP batched transfer")
        for tensor in pool.tensors:
            if not 0 <= page_index < tensor.shape[0]:
                raise ValueError("TP page index outside target pool")
        plan.append((store, buffers, page_index))
    for store, buffers, page_index in plan:
        for buffer, tensor in zip(buffers, pool.tensors):
            if store:
                buffer.copy_(tensor[page_index], non_blocking=True)
            else:
                tensor[page_index].copy_(buffer, non_blocking=True)


def mp_tp_lazy_close(context, owner, cache_id):
    pool = context.get("glm_tp_lazy_cache")
    if pool is None:
        return
    if pool.owner != owner or pool.cache_id != cache_id:
        raise ValueError("Cannot close a foreign lazy TP tier")
    pool.close()
    context["glm_tp_lazy_cache"] = None
