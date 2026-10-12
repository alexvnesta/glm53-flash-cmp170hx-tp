"""Demand-allocated LS target tier using pinned installed CPUPageCache methods.

Constructor policy and optional host-allocation failure handling differ. The
default eviction policy remains upstream. GLM53_CPU_DUPLICATE_RECYCLE=1 opts into
recycling host duplicates of complete, currently referenced GPU pages first.
Raw copies remain upstream, with a fixed per-device stream/thread contract.
"""
from collections import deque
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
import torch

import exllamav3.generator.cpu_cache as native

SOURCE_SHA256 = "4ba765dff08ac77e40ee1849b581385b6a98727c560c536910019b4657bf5906"
PAGETABLE_SHA256 = "0cde876b9969df7120ed7137b35bff3040dc3a7613c9f5a9547e58da4a9849ea"


def align(n, a):
    return (n + a - 1) // a * a


class HostAllocationUnavailable(MemoryError):
    pass


class ProtectedHostEntries(HostAllocationUnavailable):
    """No safe host slot: preserve an incoming chain and skip this spill."""


def current_transfer_streams(segments):
    """One stream per fixed device, without changing the caller's stream."""
    devices = sorted({str(t.device) for t, _, _, _ in segments})
    return tuple((device, torch.cuda.current_stream(torch.device(device)).cuda_stream)
                 for device in devices)


class LazyTargetCPUPageCache(native.CPUPageCache):
    reservation_policy = "pressure_demand_pinned_slabs"

    def __init__(self, caches, max_size):
        recycle = os.environ.get("GLM53_CPU_DUPLICATE_RECYCLE", "0")
        if recycle not in ("0", "1"):
            raise ValueError("GLM53_CPU_DUPLICATE_RECYCLE must be exactly 0 or 1")
        source = Path(native.__file__).read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("Lazy target tier requires the pinned installed CPUPageCache source")
        if len(caches) != 1 or caches[0].model.loaded_tp:
            raise ValueError("Lazy target tier supports exactly one LS target cache")
        if type(max_size) is not int or max_size <= 0:
            raise ValueError("Lazy target byte budget must be a positive integer")
        self._target_cache = caches[0]
        self._generator_owner = None
        self._prefer_referenced_duplicates = recycle == "1"
        self._page_geometry = self._segment_geometry = self._transfer_owner = None
        self.tp_groups = []
        self.tp = False
        self.segments = []
        offset = 0
        for layer in caches[0].layers.values():
            for t in layer.get_tensors():
                if t is None:
                    continue
                if t.device.type != "cuda":
                    raise ValueError("Load target cache tensors before attaching the host tier")
                page_shape = tuple(t.shape[1:])
                nbytes = t[0].numel() * t.element_size()
                self.segments.append((t, offset, page_shape, t.dtype))
                offset = align(offset + nbytes, 256)
        if not offset:
            raise ValueError("No paged target cache tensors")
        self.slab_size = self.slot_size = align(offset, 4096)
        self.max_slots = max_size // self.slot_size
        if self.max_slots < 2:
            raise ValueError("Target CPU tier must hold at least two complete page images")
        self.pagetable = None
        self.entries = {}
        self.slot_slabs = []
        self.slot_views = []
        self.free_slots = deque()
        self.num_slots = 0
        self._order = deque()
        self._order_pops = 0
        self._order_rebuild = max(64, self.max_slots // 8)
        self.metrics = {"pushes": 0, "dedup_hits": 0, "restores": 0,
                        "evictions": 0, "cold_allocs": 0, "pin_failures": 0, "skipped_spills": 0,
                        "duplicate_evictions": 0, "protected_spill_skips": 0}
        self._growth_disabled = False
        self._local_cold_allocs = self._tp_cold_allocs = 0
        # Upstream _new_slot already has a synchronous no-spare fallback. With
        # no producer thread, exactly that path pins one page only on its first
        # actual eviction. Recycled slots reuse existing pinned slabs.
        self._spare = deque()
        self._spare_cond = threading.Condition()
        self._stop_event = threading.Event()
        self._alloc_thread = None

    def attach(self, pagetable):
        if self._prefer_referenced_duplicates:
            module = importlib.import_module("exllamav3.generator.pagetable")
            source = Path(module.__file__).read_bytes()
            if hashlib.sha256(source).hexdigest() != PAGETABLE_SHA256:
                raise ValueError("Duplicate recycling requires the pinned installed PageTable source")
            if not isinstance(pagetable, module.PageTable):
                raise ValueError("Duplicate recycling requires the native PageTable")
            generator = pagetable.generator
            max_pages = self._target_cache.max_num_tokens // 256
            if (pagetable.cache is not self._target_cache or generator.cache is not self._target_cache
                    or generator.model is not self._target_cache.model or generator.model.loaded_tp
                    or generator.max_batch_size != 1 or generator.enable_defrag
                    or pagetable.max_pages != max_pages or len(pagetable.all_pages) != max_pages
                    or any(t.shape[0] != max_pages for t, _, _, _ in self.segments)):
                mismatched = [dict(segment=i, device=str(t.device), pages=t.shape[0])
                              for i, (t, _, _, _) in enumerate(self.segments)
                              if t.shape[0] != max_pages]
                snapshot = dict(enable_defrag=generator.enable_defrag,
                                max_batch_size=generator.max_batch_size,
                                model_loaded_tp=generator.model.loaded_tp,
                                generator_cache_matches=generator.cache is self._target_cache,
                                model_owner_matches=generator.model is self._target_cache.model,
                                pagetable_cache_matches=pagetable.cache is self._target_cache,
                                expected_pages=max_pages, pagetable_pages=pagetable.max_pages,
                                physical_page_count=len(pagetable.all_pages),
                                segment_count=len(self.segments),
                                mismatched_segments=mismatched[:8],
                                mismatched_segment_count=len(mismatched))
                raise ValueError("Duplicate recycling requires fixed serialized LS page geometry: "
                                 + json.dumps(snapshot, sort_keys=True))
            self._page_geometry = (pagetable, pagetable.all_pages, max_pages,
                                   tuple(id(page) for page in pagetable.all_pages))
            self._generator_owner = generator
            self._segment_geometry = self._segments_signature()
        super().attach(pagetable)

    def _segments_signature(self):
        return tuple((id(t), t.data_ptr(), str(t.device), tuple(t.shape), t.dtype,
                      offset, page_shape, dtype)
                     for t, offset, page_shape, dtype in self.segments)

    def _check_transfer_owner(self):
        """Refuse drift before any raw transfer or host slot mutation.

        Native copy_ uses the current stream of each CUDA segment. A slot byte
        range belongs to exactly one fixed device, so a prior H2D read and the
        recycled-slot D2H write are ordered on that same stream. Distinct devices
        operate on disjoint ranges. The serialized single worker also owns page
        metadata through allocation; health reads do not claim or evict pages.
        """
        if not self._prefer_referenced_duplicates:
            return
        if self._page_geometry is None:
            raise ValueError("Attach duplicate recycling before transfers")
        pt, all_pages, max_pages, page_ids = self._page_geometry
        g = pt.generator
        if (self.pagetable is not pt or pt.cache is not self._target_cache
                or g is not self._generator_owner
                or g.cache is not self._target_cache or g.model is not self._target_cache.model
                or self._target_cache.max_num_tokens != max_pages * 256
                or pt.all_pages is not all_pages or pt.max_pages != max_pages
                or tuple(id(page) for page in all_pages) != page_ids
                or g.model.loaded_tp or g.max_batch_size != 1 or g.enable_defrag
                or tuple(id(t) for t in self._target_cache.get_all_tensors() if t is not None)
                   != tuple(id(t) for t, _, _, _ in self.segments)
                or self._segments_signature() != self._segment_geometry):
            raise ValueError("Duplicate recycling page/segment geometry changed")
        owner = (threading.get_ident(), current_transfer_streams(self.segments))
        if self._transfer_owner is None:
            self._transfer_owner = owner
        elif self._transfer_owner != owner:
            raise ValueError("Duplicate recycling transfer thread/stream changed")

    def _referenced_duplicate(self, phash, entry):
        pt = self.pagetable
        page = pt.referenced_pages.get(phash)
        if (page is None or page.pagetable is not pt or page.ref_count <= 0
                or page.phash != phash or page.kv_position != 256 or page.can_revert
                or phash[:8] == bytes(8) or page.prev_hash != entry["prev_hash"]
                or type(page.page_index) is not int
                or not 0 <= page.page_index < pt.max_pages
                or pt.all_pages[page.page_index] is not page
                or page.sequence.device.type != "cpu" or entry["tokens"].device.type != "cpu"
                or not torch.equal(page.sequence, entry["tokens"])):
            return False
        return True

    def _evict_one(self, protect):
        if not self._prefer_referenced_duplicates:
            return super()._evict_one(protect)
        self._check_transfer_owner()
        assert self.entries, "CPU page cache has no entries to evict (logic error)"
        # A protected hash may already have been restored by this allocation.
        # Native GPU eviction cannot repurpose a referenced page, so only its
        # redundant host image may be recycled, even while the hash is protected.
        candidates = sorted(tuple(self.entries.items()),
                            key=lambda item: (item[1]["access_serial"], item[1]["slot"], item[0]))
        for phash, entry in candidates:
            if self._referenced_duplicate(phash, entry):
                if self.entries.get(phash) is not entry:
                    raise ValueError("Duplicate recycling host entry changed")
                self.entries.pop(phash)
                self._order.clear()
                self.metrics["evictions"] += 1
                self.metrics["duplicate_evictions"] += 1
                return entry["slot"]
        if protect and all(phash in protect for phash in self.entries):
            # Native _evict_one eventually takes a protected entry when all are
            # protected. This optional policy must preserve their sole host copy.
            self.metrics["protected_spill_skips"] += 1
            raise ProtectedHostEntries("All host entries are protected and not restored")
        # Rebuild the native tree/LRU snapshot so newly stored unprotected pages
        # are represented before its protected-entry deferral loop runs.
        self._order.clear()
        return super()._evict_one(protect)

    @property
    def pinned_bytes(self):
        return len(self.slot_slabs) * self.slab_size

    def _make_slab(self):
        if self._growth_disabled:
            raise HostAllocationUnavailable("Host pin growth disabled after an allocation failure")
        try:
            slab = torch.empty((self.slab_size,), dtype=torch.uint8, pin_memory=True)
        except (MemoryError, RuntimeError) as error:
            message = str(error).lower()
            if (not isinstance(error, MemoryError) and not any(s in message for s in
                    ("out of memory", "cannot allocate memory", "memory allocation", "pin memory", "malloc"))):
                raise  # CUDA/context/shape faults must still fail the engine.
            self._growth_disabled = True
            self.metrics["pin_failures"] += 1
            raise HostAllocationUnavailable("Pinned host allocation unavailable") from error
        views = []
        for t, offset, page_shape, dtype in self.segments:
            nbytes = t[0].numel() * t.element_size()
            views.append(slab[offset:offset+nbytes].view(dtype).view(page_shape))
        return slab, views

    def _new_slot(self, protect):
        if self._growth_disabled and self.num_slots:
            return self.free_slots.popleft() if self.free_slots else self._evict_one(protect)
        return super()._new_slot(protect)

    def store(self, page, serial, protect=None):
        self._check_transfer_owner()
        try:
            return super().store(page, serial, protect)
        except HostAllocationUnavailable:
            # Optional prefix preservation may degrade to cold replay. Throwing
            # mid-PageTable.allocate_pages would leave its partial claim list
            # inaccessible to the Job's cleanup, so host capacity failure must
            # not interrupt that native allocation transaction.
            self.metrics["skipped_spills"] += 1

    def fetch(self, phash, page_index, serial):
        self._check_transfer_owner()
        return super().fetch(phash, page_index, serial)
