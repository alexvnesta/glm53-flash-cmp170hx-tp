"""Uninstalled B2 paired-retention foundation; existing composition gates stay shut.

Callers must already hold the source-bound persistent native step owner. This
module does not install hooks, enable flags, allocate a model or attach a tier.
"""
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from threading import RLock

PAGE_SIZE = 256
WINDOW_PAGES = 8
COOKIE = "_glm_dflash_session_cookie"


class UnsafeCleanupError(RuntimeError):
    """Future integration must latch the shared generator, not reap one job."""
    terminal = True


class SnapshotBudgetExceeded(ValueError):
    """Protected images exceed a declared budget before any snapshot allocation."""


class Phase(Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class Pair:
    position: int
    key: bytes
    prefix: object
    page_indices: tuple
    tensors: tuple
    cookie: str
    namespace: str
    native_bytes: int

    def __post_init__(self):
        import torch
        if (type(self.position) is not int or self.position <= 0 or self.position % PAGE_SIZE
                or type(self.key) is not bytes or not self.key
                or type(self.cookie) is not str or not self.cookie
                or type(self.namespace) is not str or not self.namespace
                or type(self.native_bytes) is not int or self.native_bytes < 0):
            raise ValueError("Paired identity/position/native bytes are invalid")
        expected = tuple(p % 32 for p in range(max(0, self.position // PAGE_SIZE - WINDOW_PAGES), self.position // PAGE_SIZE))
        if (self.page_indices != expected or not self.tensors
                or self.prefix.device.type != "cpu" or self.prefix.dtype != torch.long
                or tuple(self.prefix.shape) != (1, self.position)):
            raise ValueError("Paired prefix/window geometry is invalid")
        for tensor in (self.prefix, *self.tensors):
            if tensor.device.type != "cpu" or tensor._base is not None or not tensor.is_contiguous():
                raise ValueError("Paired snapshots must own contiguous CPU copies")
        if any(t.shape[0] != len(expected) or t.shape[1] != PAGE_SIZE for t in self.tensors):
            raise ValueError("Paired draft planes have invalid raw page geometry")

    @property
    def identity(self):
        return self.key, self.cookie

    @property
    def host_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.prefix, *self.tensors))

    @property
    def paired_bytes(self):
        return self.host_bytes + self.native_bytes


@dataclass
class JobPairs:
    job: object
    phase: Phase = Phase.PLANNED
    plan: Pair | None = None
    prompt: Pair | None = None
    decode: Pair | None = None
    shared: Pair | None = None
    shared_info: object = None

    def images(self):
        return tuple(cp for cp in (self.plan, self.prompt, self.decode, self.shared) if cp is not None)


@dataclass(frozen=True)
class Reservation:
    job: object
    role: str
    size: int
    serial: int


class B2Pairs:
    """One completed LRU and independent job records; no native tensor ownership.

    release_native must be the source-bound cookie/alias-aware native release,
    not a direct handle deletion. All current pairs and reservations count
    against the one shared paired budget. A reservation fails before allocation
    if protected live pairs cannot fit; it never expands the budget.
    """
    def __init__(self, max_bytes, max_checkpoints, namespace, release_native):
        if (type(max_bytes) is not int or max_bytes <= 0
                or type(max_checkpoints) is not int or max_checkpoints < 4
                or not namespace or not callable(release_native)):
            raise ValueError("B2 pairs require finite budget, four slots and native custody")
        self.max_bytes = max_bytes
        self.max_checkpoints = max_checkpoints
        self.namespace = namespace
        self.release_native = release_native
        self.completed = OrderedDict()
        self.jobs = {}
        self.reservations = {}
        self.quarantined = {}
        self.terminal = False
        self.serial = 0
        self.evictions = 0
        self.lock = RLock()

    def _all(self):
        images = dict(self.completed)
        images.update(self.quarantined)
        for state in tuple(self.jobs.values()):
            for cp in state.images():
                existing = images.get(cp.identity)
                if existing is not None and existing is not cp:
                    raise ValueError("A paired cookie has conflicting immutable images")
                images[cp.identity] = cp
        return images

    def _drop_if_unused(self, cp):
        if cp is not None and cp.identity not in self._all():
            try:
                self.release_native(cp)
            except BaseException as error:
                self.quarantined[cp.identity] = cp
                self.terminal = True
                raise UnsafeCleanupError("Native paired release uncertain; image retained") from error

    def _state(self, job, phases):
        if self.terminal:
            raise UnsafeCleanupError("B2 paired registry is terminal")
        state = self.jobs.get(job)
        if state is None or state.job is not job or state.phase not in phases:
            raise ValueError("Job is not in the required paired lifecycle phase")
        return state

    def begin(self, job, plan=None):
        with self.lock:
            if self.terminal:
                raise UnsafeCleanupError("B2 paired registry is terminal")
            if job in self.jobs or len(self.jobs) >= 2:
                raise ValueError("B2 paired foundation permits two distinct jobs")
            if plan is not None:
                if plan.namespace != self.namespace or self.completed.get(plan.identity) is not plan:
                    raise ValueError("Resume plan is not an owned completed paired image")
                self.completed.move_to_end(plan.identity)
            state = JobPairs(job, plan=plan)
            self.jobs[job] = state
            return state

    def activate(self, job):
        with self.lock:
            state = self._state(job, {Phase.PLANNED})
            state.phase = Phase.ACTIVE

    def abandon_plan(self, job):
        """Native epoch/cache miss changes this job only; caller retries cold."""
        with self.lock:
            state = self._state(job, {Phase.PLANNED, Phase.ACTIVE})
            cp, state.plan = state.plan, None
            self._drop_if_unused(cp)

    def restored(self, job):
        with self.lock:
            state = self._state(job, {Phase.ACTIVE})
            if state.plan is None:
                raise ValueError("Job has no resume plan")
            state.prompt, state.plan = state.plan, None

    def reserve(self, job, role, size):
        with self.lock:
            state = self._state(job, {Phase.ACTIVE})
            if role not in ("prompt", "decode") or type(size) is not int or size <= 0:
                raise ValueError("Paired snapshot reservation requires a role and positive size")
            if job in self.reservations:
                raise ValueError("Job already has an unfinished snapshot reservation")
            old = getattr(state, role)
            setattr(state, role, None)
            self._drop_if_unused(old)
            while True:
                images = self._all()
                used = sum(cp.paired_bytes for cp in images.values())
                reserved = sum(r.size for r in self.reservations.values())
                if (used + reserved + size <= self.max_bytes
                        and len(images) + len(self.reservations) + 1 <= self.max_checkpoints):
                    break
                protected = {cp.identity for current in self.jobs.values() for cp in current.images()}
                idle = next((identity for identity in self.completed if identity not in protected), None)
                if idle is None:
                    raise SnapshotBudgetExceeded("Protected B2 pairs cannot fit the declared snapshot budget")
                removed = self.completed.pop(idle)
                self.evictions += 1
                self._drop_if_unused(removed)
            self.serial += 1
            reservation = Reservation(job, role, size, self.serial)
            self.reservations[job] = reservation
            return reservation

    def commit(self, reservation, cp):
        with self.lock:
            state = self._state(reservation.job, {Phase.ACTIVE})
            if self.reservations.get(reservation.job) is not reservation:
                raise ValueError("Snapshot reservation is stale or belongs to another job")
            if (cp.namespace != self.namespace or cp.paired_bytes > reservation.size
                    or cp.position <= 0 or cp.position % PAGE_SIZE
                    or not cp.cookie or not cp.key):
                raise ValueError("Snapshot violates its exact reservation")
            existing = self._all().get(cp.identity)
            if existing is not None and existing is not cp:
                raise ValueError("Cannot replace an already paired native cookie")
            setattr(state, reservation.role, cp)
            del self.reservations[reservation.job]

    def reuse(self, job, role, identity):
        """Existing native put reused a cookie: keep its original paired image."""
        with self.lock:
            state = self._state(job, {Phase.ACTIVE})
            if role not in ("prompt", "decode") or job in self.reservations:
                raise ValueError("Reuse requires an unreserved snapshot role")
            cp = self._all().get(identity)
            if cp is None:
                raise ValueError("Existing native cookie has no owned paired image")
            old = getattr(state, role)
            setattr(state, role, cp)
            self._drop_if_unused(old)
            return cp

    def adopt_shared(self, job, cp):
        with self.lock:
            state = self._state(job, {Phase.ACTIVE})
            if self._all().get(cp.identity) is not cp:
                raise ValueError("Shared reference must alias an existing owned immutable pair")
            old, state.shared = state.shared, cp
            self._drop_if_unused(old)

    def finish(self, job, *, success):
        with self.lock:
            state = self._state(job, {Phase.PLANNED, Phase.ACTIVE})
            if success and (state.phase is not Phase.ACTIVE or job in self.reservations):
                raise ValueError("Only a completed active transaction can publish pairs")
            images = state.images()
            if success:
                for cp in (state.prompt, state.decode, state.shared):
                    if cp is not None:
                        self.completed[cp.identity] = cp
                        self.completed.move_to_end(cp.identity)
                state.phase = Phase.COMPLETE
            else:
                state.phase = Phase.CANCELLED
            self.reservations.pop(job, None)
            del self.jobs[job]
            # Aliased plan/prompt/decode images release only once, after last user.
            for identity, cp in {cp.identity: cp for cp in images}.items():
                self._drop_if_unused(cp)


def _lease(ring, job):
    ring.check_owner(transfer=True)
    if (ring.depth != 1 or job not in ring.leases or len(job.sequences) != 1
            or ring.slots != 2 or ring.ring_pages != 32):
        raise ValueError("Snapshot copy requires the actual leased native step")
    slot = ring.leases[job]
    if type(slot) is not int or not 0 <= slot < ring.slots:
        raise ValueError("Invalid job ring partition")
    if any(t.shape[0] != ring.slots * ring.ring_pages for t in ring.storage):
        raise ValueError("Draft physical rows differ from source-bound ring geometry")
    return slot


def require_settled_replay(job):
    """A retained checkpoint contains committed state, never a staged pass."""
    state = job.recurrent_state
    if (state is None or getattr(state, "replay_failed", True)
            or getattr(state, "last_history", None) != 0
            or getattr(state, "spec_row", "missing") is not None
            or getattr(state, "spec_shape", "missing") is not None
            or type(getattr(state, "parity", None)) is not int or state.parity not in (0, 1)):
        raise UnsafeCleanupError("Paired retention requires a settled replay state")


def _stash(ring, job, key, cookie, native_bytes):
    """Join page, target position and current native cookie before draft copy."""
    require_settled_replay(job)
    g = ring.generator
    try:
        g.model.tp_drain_acks()
    except BaseException as error:
        raise UnsafeCleanupError("Native target fence failed before draft copy") from error
    seq = job.sequences[0]
    anchor = seq.allocated_pages[seq.kv_position // PAGE_SIZE - 1]
    stash = g.recurrent_cache.get(key)
    if (anchor.phash != key or anchor.kv_position != PAGE_SIZE
            or stash is None or stash.get("position") != seq.kv_position
            or stash.get(COOKIE) != cookie or not cookie
            or type(stash.get("tp_handle")) is not int or stash["tp_handle"] < 0
            or stash.get("checkpoint_size") != native_bytes):
        raise ValueError("Draft copy lost its exact page/native checkpoint cookie")
    return stash


def capture_window(ring, job, *, key, cookie, namespace, native_bytes):
    """Blocking raw copies from this job's lease; indices remain logical."""
    import torch
    slot = _lease(ring, job)
    seq = job.sequences[0]
    pos = seq.kv_position
    if (pos <= 0 or pos % PAGE_SIZE or job.recurrent_state.position != pos
            or type(native_bytes) is not int or native_bytes < 0):
        raise ValueError("Target and draft snapshot positions must match exactly")
    stash = _stash(ring, job, key, cookie, native_bytes)
    end = pos // PAGE_SIZE
    indices = tuple(p % ring.ring_pages for p in range(max(0, end - WINDOW_PAGES), end))
    saved = []
    try:
        for src in ring.storage:
            dst = torch.empty((len(indices), *src.shape[1:]), dtype=src.dtype, device="cpu")
            for row, index in enumerate(indices):
                dst[row].copy_(src[slot * ring.ring_pages + index])
            saved.append(dst)
        _lease(ring, job)
        if _stash(ring, job, key, cookie, native_bytes) is not stash:
            raise ValueError("Native snapshot changed during draft capture")
    except BaseException as error:
        raise UnsafeCleanupError("Draft capture or closing ownership failed") from error
    return Pair(pos, key, seq.sequence_ids.torch_slice(0, pos).clone(), indices,
                tuple(saved), cookie, namespace, native_bytes)


def restore_window(ring, job, cp):
    """Called only after exact native target prefix/cookie restoration succeeds."""
    slot = _lease(ring, job)
    seq = job.sequences[0]
    expected = tuple(p % ring.ring_pages for p in
                     range(max(0, cp.position // PAGE_SIZE - WINDOW_PAGES), cp.position // PAGE_SIZE))
    if (seq.kv_position != cp.position or job.recurrent_state.position != cp.position
            or job.cached_pages * PAGE_SIZE != cp.position
            or len(cp.tensors) != len(ring.storage) or tuple(cp.page_indices) != expected
            or len(set(cp.page_indices)) != len(cp.page_indices)
            or any(type(i) is not int or not 0 <= i < ring.ring_pages for i in cp.page_indices)):
        raise ValueError("Native target restoration does not match the paired draft window")
    import torch
    if (cp.prefix.device.type != "cpu"
            or not torch.equal(seq.sequence_ids.torch_slice(0, cp.position), cp.prefix)):
        raise ValueError("Snapshot token prefix differs from this job")
    stash = _stash(ring, job, cp.key, cp.cookie, cp.native_bytes)
    for dst, saved in zip(ring.storage, cp.tensors):
        if (saved.device.type != "cpu" or saved.dtype != dst.dtype
                or tuple(saved.shape) != (len(cp.page_indices), *dst.shape[1:])):
            raise ValueError("Saved draft window schema changed")
    try:
        for dst, saved in zip(ring.storage, cp.tensors):
            for row, index in enumerate(cp.page_indices):
                dst[slot * ring.ring_pages + index].copy_(saved[row])
        _lease(ring, job)
        if _stash(ring, job, cp.key, cp.cookie, cp.native_bytes) is not stash:
            raise ValueError("Native snapshot changed during draft restoration")
    except BaseException as error:
        raise UnsafeCleanupError("Draft restoration or closing ownership failed") from error


def allocate_preserving_lease(ring, job, *, native_allocate, native_deallocate,
                              try_restore, reset_cold):
    """Core transaction; caller must pass unhooked, pinned native methods.

    Returning False from try_restore requests one native cold retry. Using
    Job.deallocate_pages with the B2 hook here would release the lease early and
    is forbidden by the future integration contract. No engine hook is installed.
    """
    ring.check_owner(transfer=True)
    if ring.depth != 1 or job in ring.leases:
        raise ValueError("Allocation requires a fresh job within its native step")
    try:
        ring.claim(job)
        native_allocate(job)
        if not try_restore(job):
            try:
                native_deallocate(job)
            except BaseException as cleanup_error:
                raise UnsafeCleanupError("Cold retry native cleanup uncertain; lease retained") from cleanup_error
            reset_cold(job)
            native_allocate(job)
    except BaseException as error:
        if isinstance(error, UnsafeCleanupError) or getattr(error, "terminal", False):
            raise
        try:
            native_deallocate(job)
        except BaseException as cleanup_error:
            error.add_note(f"Job native allocation cleanup failed: {cleanup_error!r}")
            # Do not recycle this partition while native page/state custody is
            # uncertain. The future caller must close the shared generator.
            raise UnsafeCleanupError("Native allocation cleanup uncertain; lease retained") from error
        ring.release(job)
        raise
