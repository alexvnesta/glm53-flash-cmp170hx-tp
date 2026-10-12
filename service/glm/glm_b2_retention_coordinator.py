"""Uninstalled two-job paired-retention coordinator for the pinned B2 source.

Native callbacks must be the unhooked source-bound methods captured by the
combined ring installer. No prior single-active session installer is composed.
An optional target-only lazy tier is owned by this exact coordinator. Active-host
migration remains excluded.
"""
from collections import Counter
import uuid

from glm_b2_retention_foundation import (
    B2Pairs, Phase, UnsafeCleanupError, SnapshotBudgetExceeded, COOKIE, PAGE_SIZE, WINDOW_PAGES,
    allocate_preserving_lease, capture_window, restore_window, require_settled_replay,
)


class StopSharedGeneration(BaseException):
    """Bypass native per-job Exception handlers after shared custody is lost.

    The outer iterate/cancel bridge converts this to RuntimeError, which the
    actual AsyncGenerator failure sink broadcasts and exposes in health.
    """
    terminal = True


class B2Retention:
    def __init__(self, ring, *, max_bytes, max_checkpoints, native_delete,
                 namespace=None):
        ring.check_owner()
        g = ring.generator
        if (ring.slots != 2 or ring.ring_pages != 32 or ring.leases
                or g.num_remaining_jobs() or getattr(g, 'job_serial', 0)
                or getattr(g, '_glm_b2_retention', None) is not None
                or g.cpu_page_cache is not None or g.pagetable.cpu_tier is not None
                or g.recurrent_cache
                or getattr(g, '_glm_dflash_prefix_cache', None) is not None
                or not callable(native_delete)
                or getattr(g.recurrent_cache, 'model', g.model) is not g.model):
            raise ValueError('B2 paired retention requires a fresh owned two-slot GPU-only ring')
        self.ring = ring
        self.generator = g
        self.native = g.recurrent_cache
        self.native_delete = native_delete
        self.namespace = namespace or uuid.uuid4().hex
        self.pairs = B2Pairs(max_bytes, max_checkpoints, self.namespace, self._release_native)
        self.attached = False
        self.closed = False
        self.failure = None
        self.metrics = Counter()
        # Held between before_stash and capture on exactly one persistent step.
        self.stashes = {}
        self.tier = None
        self.shared = None

    def configure_shared_chat(self, tokens, template, tokenizer, renderer):
        self.check_owner()
        if self.shared is not None:
            raise ValueError('Shared B2 context can bind only once')
        from glm_b2_shared_prefix import B2SharedPrefix
        self.shared = B2SharedPrefix(self, tokens, template, tokenizer, renderer)

    def bind(self):
        self.check_owner()
        if self.attached:
            raise ValueError('B2 paired coordinator is already bound')
        self.ring._paired_owner = self
        self.generator._glm_b2_retention = self
        self.attached = True
        return self

    def check_owner(self, *, transfer=False):
        if self.closed or self.failure is not None or self.pairs.terminal:
            raise StopSharedGeneration('B2 paired coordinator is terminal or closed') from self.failure
        try:
            self.ring.check_owner(transfer=transfer)
        except Exception as error:
            # An ordinary ring proof failure must bypass per-job containment.
            # Existing terminal and deadline/cancellation BaseExceptions pass.
            self._terminal(error)
        if (self.generator.recurrent_cache is not self.native
                or (self.attached and getattr(self.generator, '_glm_b2_retention', None) is not self)):
            self._terminal(RuntimeError('B2 native checkpoint/coordinator owner changed'))

    def _terminal(self, error):
        self.failure = error
        self.pairs.terminal = True
        raise StopSharedGeneration('Shared B2 native or draft custody is uncertain') from error

    def _fence(self):
        self.check_owner()
        try:
            self.generator.model.tp_drain_acks()
        except BaseException as error:
            self._terminal(error)

    def _discard_native(self, key, cookie=None):
        self._fence()
        stash = self.native.get(key)
        if stash is None or (cookie is not None and stash.get(COOKIE) != cookie):
            return
        handle = stash.get('tp_handle')
        if type(handle) is not int or handle < 0:
            self._terminal(ValueError('Cannot release checkpoint without an exact native handle'))
        try:
            self.native.pop(key)
            if not any(v.get('tp_handle') == handle for v in self.native.values()):
                self.generator.model.tp_dispatch_all(self.native_delete, (id(self.native), handle))
                self.metrics['native_releases'] += 1
            self.native.update_total_size()
        except BaseException as error:
            self._terminal(error)

    def _release_native(self, cp):
        self._discard_native(cp.key, cp.cookie)

    def valid(self, cp):
        self._fence()
        stash = self.native.get(cp.key)
        return (cp.namespace == self.namespace and stash is not None
                and stash.get(COOKIE) == cp.cookie and stash.get('position') == cp.position
                and type(stash.get('tp_handle')) is int and stash['tp_handle'] >= 0
                and stash.get('checkpoint_size') == cp.native_bytes
                and self.generator.pagetable.is_resumable(cp.key))

    def prune(self):
        """Invalidate only stale images; other live plans/windows remain owned."""
        self.check_owner()
        images = self.pairs._all()
        stale = {identity: cp for identity, cp in images.items() if not self.valid(cp)}
        for identity in stale:
            self.pairs.completed.pop(identity, None)
        for state in tuple(self.pairs.jobs.values()):
            for role in ('plan', 'prompt', 'decode', 'shared'):
                cp = getattr(state, role)
                if cp is not None and cp.identity in stale:
                    setattr(state, role, None)
        for cp in stale.values():
            try:
                self.pairs._drop_if_unused(cp)
            except UnsafeCleanupError as error:
                self._terminal(error)
        if self.shared is not None:self.shared.prune()
        self.metrics['pruned'] += len(stale)

    def prepare(self, job):
        """Called after native hashing, before appending the job to pending_jobs."""
        import torch
        self.check_owner()
        if (job.generator is not self.generator or job.pagetable is not self.generator.pagetable
                or len(job.sequences) != 1 or job.is_requeued or (job.embeddings and not self.ring.allow_embeddings)
                or job.filters
                or job.banned_strings or job.prefix_token is not None
                or job.orig_max_rq_tokens is not None):
            raise ValueError('B2 paired retention supports plain single-sequence, non-requeued jobs')
        self.prune()
        seq = job.sequences[0]
        ids = seq.sequence_ids.torch()
        plan = next((cp for cp in sorted(self.pairs.completed.values(),
                                         key=lambda cp: cp.position, reverse=True)
                     if cp.position < ids.shape[-1] and self.valid(cp)
                     and torch.equal(ids[:, :cp.position], cp.prefix)), None)
        state = self.pairs.begin(job, plan)
        if self.shared is not None:
            try:self.shared.prepare(job, state)
            except StopSharedGeneration:raise
            except BaseException:
                self.pairs.finish(job, success=False)
                raise
        pages = plan.position // PAGE_SIZE if plan is not None else 0
        seq.max_cached_pages = pages
        job.all_unique_hashes = list(set(seq.page_hashes[:pages]))
        self.metrics['plans' if plan is not None else 'cold_misses'] += 1

    def reset_cold(self, job):
        self.pairs._state(job, {Phase.ACTIVE})
        self.pairs.abandon_plan(job)
        seq = job.sequences[0]
        seq.kv_position = 0
        seq.prefill_complete = False
        seq.max_cached_pages = 0
        job.all_unique_hashes = []
        job.cached_pages = job.cached_tokens = job.total_pages = job.non_sequential_pages = 0
        job.last_recurrent_checkpoint_pos = None
        self.metrics['allocation_cold_fallbacks'] += 1

    def _try_restore(self, job):
        state = self.pairs._state(job, {Phase.ACTIVE})
        cp = state.plan
        try:
            require_settled_replay(job)
        except UnsafeCleanupError as error:
            self._terminal(error)
        if cp is None:
            # A capped-zero cold allocation must never restore unpaired target state.
            return (job.cached_pages == 0 and job.sequences[0].kv_position == 0
                    and job.recurrent_state.position == 0)
        if not self.valid(cp):
            return False
        seq = job.sequences[0]
        if (seq.kv_position != cp.position or job.recurrent_state is None
                or job.recurrent_state.position != cp.position
                or job.cached_pages * PAGE_SIZE != cp.position):
            return False
        try:
            restore_window(self.ring, job, cp)
            self.pairs.restored(job)
        except UnsafeCleanupError as error:
            self._terminal(error)
        if self.shared is not None:self.shared.restored(job, cp)
        self.metrics['hits'] += 1
        self.metrics['restored_tokens'] += cp.position
        return True

    def allocate(self, job, native_allocate, native_deallocate):
        self.check_owner(transfer=True)
        pt=self.generator.pagetable
        def references():
            return (tuple((id(p),p.ref_count) for p in pt.all_pages),
                    tuple(sorted((h,id(p)) for h,p in pt.referenced_pages.items())))
        before=references()
        def checked_deallocate(owned):
            native_deallocate(owned)
            # Sequence.allocate_pages assigns its page list only when the
            # entire native PageTable allocation returns. A mid-allocation
            # fault can otherwise hide already claimed pages from Job cleanup.
            # The single persistent step prevents another job changing refs.
            if references()!=before:
                raise UnsafeCleanupError('Native allocation cleanup left target page references changed')
        self.pairs.activate(job)
        try:
            allocate_preserving_lease(self.ring, job, native_allocate=self._tier_batched(native_allocate),
                native_deallocate=checked_deallocate, try_restore=self._try_restore,
                reset_cold=self.reset_cold)
        except UnsafeCleanupError as error:
            self._terminal(error)
        except StopSharedGeneration:
            raise
        except BaseException:
            # Native transaction established cleanup and released only this lease.
            self.pairs.finish(job, success=False)
            raise

    def _tier_batched(self, native_allocate):
        # Still the unhooked native method: only the target tier's spill/restore copies issued inside one
        # native allocation are queued, then sent as batched rank RPCs in their order before it returns.
        tier = self.tier
        if tier is None:
            return native_allocate
        def allocate(job):
            end = tier.begin_transfer_batch()
            if end is None:
                return native_allocate(job)
            try:
                result = native_allocate(job)
            except BaseException as error:
                end(error)
                raise
            end()
            return result
        return allocate

    def deallocate(self, job, native_deallocate):
        self.check_owner()
        try:
            native_deallocate(job)
        except BaseException as error:
            # Native reap_failed_job catches Exception and would otherwise recycle
            # a suspect slot. This sentinel must escape to the outer iterate sink.
            self._terminal(error)
        self.ring.release(job)

    def _role(self, job):
        seq = job.sequences[0]
        boundary = (len(seq.input_ids) - 1) // PAGE_SIZE * PAGE_SIZE
        return 'prompt' if seq.kv_position <= boundary else 'decode'

    def _cost(self, job):
        seq = job.sequences[0]
        pages = min(seq.kv_position // PAGE_SIZE, WINDOW_PAGES)
        return (sum(pages * t[0].numel() * t.element_size() for t in self.ring.storage)
                + seq.kv_position * seq.sequence_ids.torch().element_size()
                + job.recurrent_state.checkpoint_size)

    def stash(self, job, cache, native_stash, *args, **kwargs):
        """Reserve before native put; capture only a genuine completed checkpoint."""
        self.check_owner(transfer=True)
        if cache is not self.native:
            self._terminal(ValueError('Checkpoint stash cache is not the owned native cache'))
        self.pairs._state(job, {Phase.ACTIVE})
        seq = job.sequences[0]
        pos = seq.kv_position
        interval = kwargs.get('interval', args[0] if args else None)
        if (pos == 0 or job.last_recurrent_checkpoint_pos == pos
                or not job.is_checkpoint_boundary(interval)):
            return native_stash(job, cache, *args, **kwargs)
        try:
            require_settled_replay(job)
        except UnsafeCleanupError as error:
            self._terminal(error)
        if pos % PAGE_SIZE or job.recurrent_state.position != pos:
            self._terminal(ValueError('Checkpoint boundary and recurrent state disagree'))
        key = seq.allocated_pages[pos // PAGE_SIZE - 1].phash
        old = cache.get(key)
        identity = (key, old.get(COOKIE)) if old is not None else None
        cp = self.pairs._all().get(identity)
        role = self._role(job)
        if cp is not None and self.valid(cp):
            # Native put deliberately preserves this exact old target image.
            # Keep its matching draft image even if it belongs to another job.
            try:
                result = native_stash(job, cache, *args, **kwargs)
                if (job.last_recurrent_checkpoint_pos != pos
                        or cache.get(key) is not old or not self.valid(cp)):
                    self._terminal(ValueError('Existing paired native cookie changed during put'))
                self.pairs.reuse(job, role, cp.identity)
            except StopSharedGeneration:
                raise
            except BaseException as error:
                self._terminal(error)
            self.metrics['shared_cookie_reuses'] += 1
            return result
        try:
            reservation = self.pairs.reserve(job, role, self._cost(job))
        except SnapshotBudgetExceeded:
            # No target stash or raw snapshot is created without budget. The
            # active sequence continues on GPU; later requests explicitly miss.
            self.metrics['snapshot_budget_skips'] += 1
            return None
        except UnsafeCleanupError as error:
            self._terminal(error)
        self.stashes[job] = reservation
        if old is not None:
            self._discard_native(key)
        try:
            result = native_stash(job, cache, *args, **kwargs)
            self._fence()
            saved = cache.get(key)
            if (job.last_recurrent_checkpoint_pos != pos or saved is None
                    or saved.get('position') != pos or saved.get('checkpoint_size') != job.recurrent_state.checkpoint_size
                    or type(saved.get('tp_handle')) is not int or saved['tp_handle'] < 0):
                self._terminal(ValueError('New native stash did not publish the exact target checkpoint'))
            cookie = uuid.uuid4().hex
            saved[COOKIE] = cookie
            cp = capture_window(self.ring, job, key=key, cookie=cookie,
                                namespace=self.namespace, native_bytes=saved['checkpoint_size'])
            self.pairs.commit(reservation, cp)
            del self.stashes[job]
            self.metrics['captures'] += 1
            self.prune()
            return result
        except StopSharedGeneration:
            raise
        except BaseException as error:
            self._terminal(error)

    def finish(self, results):
        self.check_owner()
        for result in results:
            job = result.get('job')
            if job not in self.pairs.jobs:
                continue
            if result.get('stage') == 'error' or result.get('eos'):
                try:
                    self.prune()
                    shared_cp = self.pairs.jobs[job].shared
                    success = result.get('stage') != 'error'
                    self.pairs.finish(job, success=success)
                    if self.shared is not None:self.shared.finish(shared_cp, success)
                except UnsafeCleanupError as error:
                    self._terminal(error)

    def cancel(self, job, native_cancel):
        """Caller holds the actual async step lock; native release precedes pairs."""
        self.check_owner()
        if self.shared is not None:self.shared.drop_ticket(job)
        if job not in self.pairs.jobs:
            return native_cancel(job)
        try:
            result = native_cancel(job)
            self.pairs.finish(job, success=False)
        except StopSharedGeneration:
            raise
        except BaseException as error:
            self._terminal(error)
        self.metrics['cancellations'] += 1
        return result

    def stats(self):
        images = self.pairs._all()
        return {'mode': 'b2_paired_retention', 'target_cpu_tier': None if self.tier is None else self.tier.stats(),
                'active_job_records': len(self.pairs.jobs),
                'completed_checkpoints': len(self.pairs.completed),
                'shared_prefix_enabled': self.shared is not None,
                'shared_pending_tickets': 0 if self.shared is None else len(self.shared.tickets),
                'shared_completed_checkpoints': 0 if self.shared is None else len(self.shared.completed),
                'paired_host_bytes': sum(cp.paired_bytes for cp in images.values()),
                'paired_budget_bytes': self.pairs.max_bytes,
                'reserved_snapshot_bytes': sum(r.size for r in self.pairs.reservations.values()),
                'terminal': self.failure is not None or self.pairs.terminal,
                'paired_evictions': self.pairs.evictions,
                'max_checkpoints': self.pairs.max_checkpoints, **self.metrics}

    def retire_after_native_close(self):
        """Native close freed every rank checkpoint; no future slot reuse allowed."""
        if self.native:
            self._terminal(RuntimeError('Native generator close left checkpoint handles alive'))
        if self.shared is not None:self.shared.close()
        self.closed = True
        self.pairs.completed.clear()
        self.pairs.jobs.clear()
        self.pairs.reservations.clear()
        self.pairs.quarantined.clear()
        self.stashes.clear()


def shared_failure(error):
    return RuntimeError('B2 paired retention lost shared native custody: ' + str(error))


def paired_async_generator_class(concurrent_base):
    """Latch a terminal cancel fault through the real async error/503 channel."""
    class PairedAsyncGenerator(concurrent_base):
        def latch_shared(self, error):
            self.error = shared_failure(error)
            coordinator = getattr(self.generator, '_glm_b2_retention', None)
            if coordinator is not None and coordinator.shared is not None:
                coordinator.shared.tickets.clear()
            for job in self.jobs.values():
                job.put_result(self.error)
            self.jobs.clear()
            return self.error

        def enqueue(self, job):
            import asyncio
            if self.error is not None:
                job.put_result(self.error)
                return
            if self._concurrent_closing:
                raise RuntimeError('Concurrent paired generator is closing')
            if job in self._enqueue_tasks or job.job in self.jobs:
                raise ValueError('Async job already belongs to this generator')

            coordinator=getattr(self.generator, '_glm_b2_retention', None)
            shared=None if coordinator is None else coordinator.shared
            if shared is not None:
                try:shared.bind_ticket(job.job)
                except StopSharedGeneration as error:
                    job.put_result(self.latch_shared(error))
                    return

            async def register():
                try:
                    async with self._step_lock:
                        if self._concurrent_closing:
                            raise RuntimeError('Concurrent paired generator is closing')
                        # Skip only the concurrent task scheduler we replace,
                        # preserving the actual native enqueue/error semantics.
                        super(concurrent_base, self).enqueue(job)
                except asyncio.CancelledError:
                    return
                except StopSharedGeneration as error:
                    tracked=self.jobs.get(job.job) is job
                    failure=self.latch_shared(error)
                    # Native enqueue normally registered this job first. If a
                    # prior terminal check failed earlier, still wake its caller.
                    if not tracked:
                        job.put_result(failure)
                except Exception as error:
                    coordinator=getattr(self.generator, '_glm_b2_retention', None)
                    if coordinator is not None and (coordinator.failure is not None or coordinator.pairs.terminal):
                        failure=self.latch_shared(error)
                        job.put_result(failure)
                    else:
                        job.put_result(error)
                finally:
                    if shared is not None:shared.drop_ticket(job.job)
                    self._enqueue_tasks.pop(job,None)
            coroutine=register()
            try:self._enqueue_tasks[job]=asyncio.create_task(coroutine)
            except BaseException:
                coroutine.close()
                if shared is not None:shared.drop_ticket(job.job)
                raise

        async def close(self):
            try:return await super().close()
            finally:
                coordinator=getattr(self.generator, '_glm_b2_retention', None)
                if coordinator is not None and coordinator.shared is not None:
                    coordinator.shared.tickets.clear()

        async def cancel(self, job):
            try:
                try:return await super().cancel(job)
                finally:
                    coordinator=getattr(self.generator, '_glm_b2_retention', None)
                    if coordinator is not None and coordinator.shared is not None:
                        coordinator.shared.drop_ticket(job.job)
            except StopSharedGeneration as error:
                raise self.latch_shared(error) from error
            except RuntimeError as error:
                coordinator = getattr(self.generator, '_glm_b2_retention', None)
                if coordinator is not None and (coordinator.failure is not None or coordinator.pairs.terminal):
                    raise self.latch_shared(error) from error
                raise
    return PairedAsyncGenerator


def bind_b2_retention(generator, max_bytes=1024**3, max_checkpoints=8,
                      native_delete=None):
    """Experimental attachment for the separately opted-in B2 API branch.

    The caller has installed the exact combined ring hooks before loading and
    bound that ring on this fresh generator. The API must choose the paired
    async adapter, rather than using the single-active session installer.
    """
    ring = getattr(generator, '_glm_concurrent_ring', None)
    if ring is None or ring.generator is not generator:
        raise ValueError('A source-bound B2 ring must be installed before paired retention')
    if native_delete is None:
        from exllamav3.cache.recurrent import mp_cache_recurrent_del
        native_delete = mp_cache_recurrent_del
    return B2Retention(ring, max_bytes=max_bytes, max_checkpoints=max_checkpoints,
                       native_delete=native_delete).bind()
