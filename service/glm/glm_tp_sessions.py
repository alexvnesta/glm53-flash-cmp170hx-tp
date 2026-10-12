"""Experimental paired completed-session retention for the TP target and single-device DFlash ring.

Target pages stay in ExLlama's GPU/CPU page tiers. This adapter owns only raw
draft-window/token snapshots and scalar references to native KDA checkpoints.
It never retains native tensor objects after the native checkpoint is evicted.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import threading
import uuid

from glm_tp_prefix import PreviousRequestCache, PAGE_SIZE, WINDOW_TOKENS

COOKIE = "_glm_dflash_session_cookie"
from glm_tp_prefix import validate_engine_sources



@dataclass(frozen=True)
class PairedCheckpoint:
    position: int
    key: bytes
    prefix: object
    page_indices: tuple
    tensors: tuple
    cookie: str
    namespace: str
    native_bytes: int

    @property
    def identity(self):
        return self.key, self.cookie

    @property
    def host_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.prefix, *self.tensors))

    @property
    def paired_bytes(self):
        return self.host_bytes + self.native_bytes


class MultiSessionCache(PreviousRequestCache):
    """Byte-bounded LRU of completed paired checkpoints; one active request.

    Native checkpoint buffers remain owned by RecurrentCache. An epoch cookie
    fences reuse of a key after native LRU eviction/recreation. Dropping a pair
    returns its owned native stash through the native host-pool lifecycle.
    """
    def __init__(self, generator, max_bytes=1024**3, max_checkpoints=8, namespace=""):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Session byte budget must be a positive integer")
        if type(max_checkpoints) is not int or max_checkpoints < 2:
            raise ValueError("Session retention requires at least two checkpoint slots")
        super().__init__(generator)  # Includes the explicit TP/native checkpoint gates.
        self.max_bytes = max_bytes
        self.max_checkpoints = max_checkpoints
        self.namespace = f"{namespace}:{uuid.uuid4().hex}"
        self.registry = OrderedDict()
        self.current_job = None
        self._lock = threading.RLock()
        self.evictions = self.pruned = self.captures = self.too_large = 0
        self.native_releases = 0
        self._native_release = self._release_native
        # A prompt and latest-decode snapshot are both live during one request.
        native_min = sum(l.get_checkpoint_size()
                         for l in generator.cache.get_all_recurrent_layers().values())
        draft_max = sum(min(WINDOW_TOKENS // PAGE_SIZE, self.ring_pages)
                        * t[0].numel() * t.element_size() for t in self.storage)
        token_max = generator.cache.max_num_tokens * 8
        if max_bytes < 2 * (native_min + draft_max + token_max):
            raise ValueError("Session byte budget must hold prompt and decode paired checkpoints")

    def _all_pairs(self):
        pairs = dict(self.registry)
        for cp in (self.prompt_checkpoint, self.decode_checkpoint, self.plan):
            if cp is not None:
                pairs[cp.identity] = cp
        return pairs

    def _valid(self, cp):
        stash = self._paired_stash(cp.key, cp.position)
        return (cp.namespace == self.namespace and stash is not None
                and stash.get(COOKIE) == cp.cookie and stash["position"] == cp.position
                and self.generator.pagetable.is_resumable(cp.key))

    def _discard_native_key(self, key, cookie=None):
        self._tp_fence()
        rc = self.generator.recurrent_cache
        stash = rc.get(key)
        if stash is None or (cookie is not None and stash.get(COOKIE) != cookie):
            return
        handle = stash.get("tp_handle")
        if type(handle) is not int or handle < 0:
            raise ValueError("Cannot release a TP checkpoint without a native handle")
        # Native checkpoints may be aliased by multiple keys. Removing a key
        # never releases its rank-owned image while another native key uses it.
        rc.pop(key)
        if not any(v.get("tp_handle") == handle for v in rc.values()):
            from exllamav3.cache.recurrent import mp_cache_recurrent_del
            self.generator.model.tp_dispatch_all(mp_cache_recurrent_del, (id(rc), handle))
            self.native_releases += 1
        rc.update_total_size()

    def _release_native(self, cp):
        self._discard_native_key(cp.key, cp.cookie)

    def _release_if_unused(self, cp):
        if cp is not None and cp.identity not in self._all_pairs():
            self._native_release(cp)

    def _prune(self):
        for identity, cp in list(self.registry.items()):
            if not self._valid(cp):
                del self.registry[identity]
                self.pruned += 1
                self._release_if_unused(cp)
        for attr in ("prompt_checkpoint", "decode_checkpoint"):
            cp = getattr(self, attr)
            if cp is not None and cp is not self.plan and not self._valid(cp):
                setattr(self, attr, None)
                self._release_if_unused(cp)

    def _cold(self):
        # A miss is not a page eviction. Preserve all unrelated target chains.
        self._tp_fence()
        self.generator.recurrent_cache.prune_stranded()
        self._prune()

    def _bound(self):
        while (sum(cp.paired_bytes for cp in self._all_pairs().values()) > self.max_bytes
               or len(self._all_pairs()) > self.max_checkpoints):
            protected = {cp.identity for cp in (self.prompt_checkpoint, self.decode_checkpoint, self.plan)
                         if cp is not None}
            identity = next((k for k in self.registry if k not in protected), None)
            if identity is None:
                raise RuntimeError("Current paired snapshots exceed the configured retention bound")
            cp = self.registry.pop(identity)
            self.evictions += 1
            self._release_if_unused(cp)

    def _prepare_snapshot_room(self, job, cost):
        """Release replaceable/idle pairs before allocating another host snapshot."""
        pos = job.sequences[0].kv_position
        boundary = (len(job.sequences[0].input_ids) - 1) // PAGE_SIZE * PAGE_SIZE
        attr = "prompt_checkpoint" if pos <= boundary else "decode_checkpoint"
        old = getattr(self, attr)
        setattr(self, attr, None)
        self._release_if_unused(old)
        while (sum(cp.paired_bytes for cp in self._all_pairs().values()) + cost > self.max_bytes
               or len(self._all_pairs()) + 1 > self.max_checkpoints):
            protected = {cp.identity for cp in (self.prompt_checkpoint, self.decode_checkpoint, self.plan)
                         if cp is not None}
            identity = next((k for k in self.registry if k not in protected), None)
            if identity is None:
                raise RuntimeError("Cannot reserve a paired snapshot within the retention bound")
            cp = self.registry.pop(identity)
            self.evictions += 1
            self._release_if_unused(cp)

    def _snapshot_cost(self, job, native_bytes):
        seq = job.sequences[0]
        pages = min(seq.kv_position // PAGE_SIZE, WINDOW_TOKENS // PAGE_SIZE)
        return (sum(pages * t[0].numel() * t.element_size() for t in self.storage)
                + seq.kv_position * seq.sequence_ids.torch().element_size() + native_bytes)

    def invalidate(self, reason):
        """Discard only unpublished work; completed unrelated sessions survive."""
        with self._lock:
            old = (self.prompt_checkpoint, self.decode_checkpoint, self.plan)
            self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
            self.job = self.current_job = None
            self.last_reason = reason
            for cp in old:
                self._release_if_unused(cp)
            self._prune()

    def invalidate_all(self, reason, require_idle=True):
        """Call before changing model/layout/source epoch; no cross-owner reuse."""
        with self._lock:
            if require_idle and self.current_job is not None:
                raise ValueError("Source/layout invalidation requires an idle generator")
            old = tuple(self._all_pairs().values())
            self.registry.clear()
            self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
            self.job = self.current_job = None
            self.last_reason = reason
            for cp in old:
                self._release_if_unused(cp)

    def begin(self, job):
        import torch
        with self._lock:
            self._tp_fence()
            if self.current_job is not None:
                raise ValueError("Session retention requires serialized requests")
            if len(job.sequences) != 1:
                raise ValueError("Session retention requires exactly one sequence")
            self._prune()
            self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
            self.current_job = job
            supported = (len(job.sequences) == 1 and not job.is_requeued
                         and not job.embeddings and not job.filters and not job.banned_strings
                         and job.prefix_token is None and job.orig_max_rq_tokens is None)
            self.job = job if supported else None
            ids = job.sequences[0].sequence_ids.torch()
            if supported:
                for cp in sorted(self.registry.values(), key=lambda p: p.position, reverse=True):
                    if (cp.position < ids.shape[-1] and self._valid(cp)
                            and torch.equal(ids[:, :cp.position], cp.prefix)):
                        self.plan = self.prompt_checkpoint = cp
                        self.registry.move_to_end(cp.identity)
                        self.last_reason = "matched"
                        break
            if self.plan is None:
                self.misses += 1
                self.last_reason = "prefix_or_state_miss" if supported else "unsupported_request"
                self._cold()

    def restrict(self, job):
        with self._lock:
            if job is not self.current_job:
                raise ValueError("Session jobs must enter the serialized begin hook")
            # Even unsupported jobs must run cold: native reuse without a paired
            # draft window would make the modulo ring refer to unrelated history.
            pages = self.plan.position // PAGE_SIZE if self.plan is not None else 0
            seq = job.sequences[0]
            seq.max_cached_pages = pages
            job.all_unique_hashes = list(set(seq.page_hashes[:pages]))

    def restore(self, job, allocate):
        with self._lock:
            if job is not self.job or self.plan is None:
                return
            cp = self.plan
            # Native/CPU-tier eviction can occur between planning and allocation.
            # Base restore performs the exact position/count checks and cold retry.
            if not self._valid(cp):
                job.deallocate_pages()
                self.prompt_checkpoint = self.plan = None
                seq = job.sequences[0]
                seq.kv_position = 0
                seq.prefill_complete = False
                seq.max_cached_pages = 0
                job.all_unique_hashes = []
                job.cached_pages = job.cached_tokens = job.total_pages = job.non_sequential_pages = 0
                job.last_recurrent_checkpoint_pos = None
                self.misses += 1
                self.last_reason = "allocation_epoch_miss"
                self._cold()
                allocate(job)
                return
            super().restore(job, allocate)

    def before_stash(self, job, interval=None):
        """Ensure a new raw ring capture is paired with a newly stashed target.

        Native put() keeps an existing checkpoint under the same page hash.
        Reuse an existing paired snapshot in that case, or remove an unpaired
        old checkpoint before put() so the current target/draft states stay paired.
        """
        with self._lock:
            if job is not self.job:
                return
            seq = job.sequences[0]
            pos = seq.kv_position
            if (not pos or pos % PAGE_SIZE or job.last_recurrent_checkpoint_pos == pos
                    or not job.is_checkpoint_boundary(interval)):
                return
            key = seq.allocated_pages[pos // PAGE_SIZE - 1].phash
            rc = self.generator.recurrent_cache
            stash = rc.get(key)
            identity = (key, stash.get(COOKIE)) if stash is not None else None
            cp = self._all_pairs().get(identity)
            if cp is not None and self._valid(cp):
                return
            self._prepare_snapshot_room(job, self._snapshot_cost(job, job.recurrent_state.checkpoint_size))
            if stash is None:
                return
            self._discard_native_key(key)

    def capture(self, job):
        import torch
        with self._lock:
            if job is not self.job:
                return
            seq = job.sequences[0]
            pos = seq.kv_position
            if not pos or pos % PAGE_SIZE or job.recurrent_state.position != pos:
                return
            key = seq.allocated_pages[pos // PAGE_SIZE - 1].phash
            self._tp_fence()
            stash = self._paired_stash(key, pos)
            if stash is None or stash["position"] != pos:
                return
            cookie = stash.get(COOKIE)
            cp = self._all_pairs().get((key, cookie))
            if cp is None:
                end_page = pos // PAGE_SIZE
                indices = tuple(p % self.ring_pages for p in
                                range(max(0, end_page - WINDOW_TOKENS // PAGE_SIZE), end_page))
                cost = self._snapshot_cost(job, stash["checkpoint_size"])
                if cost > self.max_bytes:
                    self.too_large += 1
                    return
                self._prepare_snapshot_room(job, cost)
                cookie = uuid.uuid4().hex
                stash[COOKIE] = cookie
                saved = []
                for src in self.storage:
                    dst = torch.empty((len(indices), *src.shape[1:]), dtype=src.dtype, device="cpu")
                    for row, index in enumerate(indices):
                        dst[row].copy_(src[index])  # Blocking publication fence.
                    saved.append(dst)
                cp = PairedCheckpoint(pos, key, seq.sequence_ids.torch_slice(0, pos).clone(),
                                      indices, tuple(saved), cookie, self.namespace, stash["checkpoint_size"])
                self.captures += 1
            prompt_boundary = (len(seq.input_ids) - 1) // PAGE_SIZE * PAGE_SIZE
            attr = "prompt_checkpoint" if pos <= prompt_boundary else "decode_checkpoint"
            old = getattr(self, attr)
            setattr(self, attr, cp)
            self._release_if_unused(old)
            self._prune()
            self._bound()

    def finish(self, results):
        with self._lock:
            for result in results:
                if result.get("job") is not self.current_job:
                    continue
                if result.get("stage") == "error":
                    self.invalidate("generation_error")
                    return
                if result.get("eos"):
                    for cp in (self.prompt_checkpoint, self.decode_checkpoint):
                        if cp is not None and self._valid(cp):
                            self.registry[cp.identity] = cp
                            self.registry.move_to_end(cp.identity)
                    self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
                    self.job = self.current_job = None
                    self._prune()
                    self._bound()
                    return

    def stats(self):
        with self._lock:
            pairs = self._all_pairs()
            tier = self.generator.cpu_page_cache
            rc = self.generator.recurrent_cache
            return {"mode": "tp_multi_session", "hits": self.hits, "misses": self.misses,
                    "restored_tokens": self.restored_tokens, "last_reason": self.last_reason,
                    "snapshots": len(pairs), "completed_checkpoints": len(self.registry),
                    "draft_snapshot_bytes": sum(cp.host_bytes for cp in pairs.values()),
                    "host_bytes": sum(cp.host_bytes for cp in pairs.values()),
                    "paired_native_bytes": sum(cp.native_bytes for cp in pairs.values()),
                    "paired_host_bytes": sum(cp.paired_bytes for cp in pairs.values()),
                    "paired_budget_bytes": self.max_bytes, "max_checkpoints": self.max_checkpoints,
                    "evictions": self.evictions, "pruned": self.pruned, "captures": self.captures,
                    "native_releases": self.native_releases, "snapshot_too_large": self.too_large,
                    "native_cache_bytes": rc.current_size, "native_cache_budget_bytes": rc.max_size,
                    "target_cpu_tier": None if tier is None else tier.stats()}

    def close(self):
        # The API calls this only after its GPU worker has fully stopped.
        self.invalidate_all("closed", require_idle=False)


def attach_target_cpu_tier(generator, max_bytes, factory=None):
    """TP target pages only: draft modulo ring is never attached to this tier."""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("CPU target byte budget must be a nonnegative integer")
    if not max_bytes:
        return None
    if (not generator.model.loaded_tp or generator.num_remaining_jobs()
            or generator.cpu_page_cache is not None or generator.pagetable.cpu_tier is not None):
        raise ValueError("TP target tier requires an idle TP generator with no existing tier")
    if factory is None:
        from glm_tp_cpu_cache import LazyTPTargetCPUPageCache
        factory = LazyTPTargetCPUPageCache
    tier = factory([generator.cache], max_bytes)
    try:
        tier.attach(generator.pagetable)
    except BaseException as error:
        try:
            tier.close()
        except BaseException as cleanup_error:
            error.add_note(f"CPU tier cleanup also failed: {cleanup_error!r}")
        raise
    generator.cpu_page_cache = generator.pagetable.cpu_tier = tier
    return tier


def enable_session_cache(generator, max_bytes=1024**3, max_checkpoints=8,
                         target_cpu_bytes=0, namespace="", tier_factory=None):
    """Install locally gated hooks after caches load and before any enqueue."""
    pagetable = getattr(generator, "pagetable", None)
    if (generator.num_remaining_jobs() or getattr(generator, "active_jobs", ())
            or getattr(generator, "pending_jobs", ()) or getattr(generator, "job_serial", 0)
            or getattr(generator, "_glm_dflash_prefix_cache", None) is not None
            or getattr(generator, "cpu_page_cache", None) is not None
            or getattr(pagetable, "cpu_tier", None) is not None
            or getattr(pagetable, "referenced_pages", {})):
        raise ValueError("Session cache must be installed on a fresh idle generator")
    validate_engine_sources()
    from exllamav3.generator.generator import Generator
    if not isinstance(generator, Generator):
        raise ValueError("Session cache requires the pinned native Generator")
    manager = MultiSessionCache(generator, max_bytes, max_checkpoints, namespace)
    original_defrag = generator.enable_defrag
    had_manager = hasattr(generator, "_glm_dflash_prefix_cache")
    original_manager = getattr(generator, "_glm_dflash_prefix_cache", None)
    tier = None
    installed_hooks = []
    try:
        # Session identities require stable physical pages. Native Generator
        # defaults this flag to True; disable it in this fresh-owned transaction
        # before the optional target tier validates/attaches its fixed geometry.
        generator.enable_defrag = False
        tier = attach_target_cpu_tier(generator, target_cpu_bytes, tier_factory)
        # Reuse the already-qualified previous-request wrapper call order. Its
        # dynamic manager lookup also preserves the previous-request rollback mode.
        from glm_tp_prefix import enable_prefix_cache
        from exllamav3.generator.job import Job
        enable_prefix_cache(generator, manager=manager)
        if not getattr(Job.maybe_stash_recurrent, "_glm_tp_dflash_session_stash", False):
            original_stash = Job.maybe_stash_recurrent

            def stash_hook(self, cache, *args, **kwargs):
                m = getattr(self.generator, "_glm_dflash_prefix_cache", None)
                if isinstance(m, MultiSessionCache):
                    m.before_stash(self, kwargs.get("interval", args[0] if args else None))
                return original_stash(self, cache, *args, **kwargs)

            stash_hook._glm_tp_dflash_session_stash = True
            Job.maybe_stash_recurrent = stash_hook
            installed_hooks.append((Job, "maybe_stash_recurrent", original_stash, stash_hook))
        if not getattr(Generator.clear_queue, "_glm_tp_dflash_session_clear", False):
            original_clear = Generator.clear_queue

            def clear_hook(self):
                m = getattr(self, "_glm_dflash_prefix_cache", None)
                if isinstance(m, MultiSessionCache):
                    m.invalidate("queue_cleared")
                return original_clear(self)

            clear_hook._glm_tp_dflash_session_clear = True
            Generator.clear_queue = clear_hook
            installed_hooks.append((Generator, "clear_queue", original_clear, clear_hook))
        generator._glm_dflash_prefix_cache = manager
    except BaseException as error:
        generator.enable_defrag = original_defrag
        if had_manager:
            generator._glm_dflash_prefix_cache = original_manager
        elif hasattr(generator, "_glm_dflash_prefix_cache"):
            del generator._glm_dflash_prefix_cache
        for owner, name, original, installed in reversed(installed_hooks):
            if getattr(owner, name) is installed:
                setattr(owner, name, original)
        if tier is not None:
            if generator.cpu_page_cache is tier:
                generator.cpu_page_cache = None
            if generator.pagetable.cpu_tier is tier:
                generator.pagetable.cpu_tier = None
            try:
                tier.close()
            except BaseException as cleanup_error:
                error.add_note(f"CPU tier cleanup also failed: {cleanup_error!r}")
        raise
    return manager
