"""Paired retention over upstream's native PairedState checkpoints (P16 stage 2).

Upstream checkpoints a job's target (KDA) state and its drafter's sliding-window state together, as one
PairedState stash under one page hash in one LRU RecurrentCache, and restores the deepest usable pair inside
Job.allocate_pages. Draft custody, the checkpoint budget and eviction are native. This layer keeps the service
policy only: bounded admission records, the terminal latch, the shared system/tools boundary (glm_b2_shared_prefix)
and the batched target RAM tier (glm_b2_cpu_cache). It installs no hooks (glm_native_concurrency does) and owns no
tensors. It replaces glm_b2_retention_foundation and glm_b2_retention_coordinator (stage 1).
"""
from collections import Counter
from dataclasses import dataclass
import uuid

PAGE_SIZE = 256
# /health prompt_cache counters present from start-up (the stage-1 key shapes the soak and qual tooling read)
COUNTERS = ('plans', 'cold_misses', 'hits', 'restored_tokens', 'allocation_cold_fallbacks', 'captures',
            'cancellations', 'shared_cookie_reuses')
SHARED_COUNTERS = ('shared_hits', 'shared_restored_tokens', 'shared_captures', 'shared_capture_skips',
                   'shared_adapted_boundaries', 'shared_boundary_overshoots')


class UnsafeCleanupError(RuntimeError):
    """Native custody is uncertain: latch the shared generator, never reap one job."""
    terminal = True


class StopSharedGeneration(BaseException):
    """Bypass native per-job Exception handlers after shared custody is lost.

    The outer iterate/cancel bridge converts this to RuntimeError, which the
    actual AsyncGenerator failure sink broadcasts and exposes in health.
    """
    terminal = True


def native_handle(stash):
    """The target-side TP checkpoint handle of a native paired stash, or None."""
    side = stash.get("target") if isinstance(stash, dict) else None
    handle = side.get("tp_handle") if isinstance(side, dict) else None
    return handle if type(handle) is int and handle >= 0 else None


def require_settled_replay(job):
    """A retained checkpoint holds committed state on both sides, never a staged speculative pass."""
    pair = job.recurrent_state
    target, draft = getattr(pair, "target", None), getattr(pair, "draft", None)
    if (target is None or draft is None or getattr(target, "replay_failed", True)
            or getattr(target, "last_history", None) != 0
            or getattr(target, "spec_row", "missing") is not None
            or getattr(target, "spec_shape", "missing") is not None
            or type(getattr(target, "parity", None)) is not int or target.parity not in (0, 1)
            or getattr(draft, "last_history", None) != 0 or getattr(draft, "position", None) != target.position):
        raise UnsafeCleanupError("Paired retention requires a settled replay state on both sides")


@dataclass
class JobRecord:
    job: object
    phase: str = "prepared"
    shared_info: object = None


class B2Retention:
    def __init__(self, concurrency, *, namespace=None):
        concurrency.check_owner()
        g = concurrency.generator
        native = g.recurrent_cache
        if (concurrency.slots not in (2,4) or concurrency._paired_owner is not None
                or g.num_remaining_jobs() or getattr(g, 'job_serial', 0)
                or getattr(g, '_glm_b2_retention', None) is not None
                or g.cpu_page_cache is not None or g.pagetable.cpu_tier is not None
                or getattr(g, 'disk_page_cache', None) is not None
                or native is None or len(native) or getattr(native, 'model', None) is not g.model
                or getattr(native, 'pagetable', None) is not g.pagetable
                or getattr(native, 'disk_tier', None) is not None or not g.draft_recurrent):
            raise ValueError('B2 native retention requires a fresh owned two- or four-slot native SWA generator')
        self.concurrency = concurrency
        self.slots = concurrency.slots
        self.generator = g
        self.native = native
        self.namespace = namespace or uuid.uuid4().hex
        self.attached = False
        self.closed = False
        self.closing = False
        self.failure = None
        self.metrics = Counter(dict.fromkeys(COUNTERS, 0))
        self.jobs = {}
        self.tier = None
        self.disk = None
        self.shared = None

    def configure_shared_chat(self, tokens, template, tokenizer, renderer):
        self.check_owner()
        if self.shared is not None:
            raise ValueError('Shared B2 context can bind only once')
        from glm_b2_shared_prefix import B2SharedPrefix
        self.shared = B2SharedPrefix(self, tokens, template, tokenizer, renderer)
        self.metrics.update(dict.fromkeys(SHARED_COUNTERS, 0))

    def bind(self):
        self.check_owner()
        if self.attached:
            raise ValueError('B2 paired coordinator is already bound')
        self.concurrency._paired_owner = self
        self.generator._glm_b2_retention = self
        self.attached = True
        return self

    def check_owner(self, *, transfer=False):
        if self.closed or self.failure is not None:
            raise StopSharedGeneration('B2 paired coordinator is terminal or closed') from self.failure
        try:
            self.concurrency.check_owner(transfer=transfer)
        except Exception as error:
            # An ordinary proof failure must bypass per-job containment.
            # Existing terminal and deadline/cancellation BaseExceptions pass.
            self._terminal(error)
        if (self.concurrency.slots != self.slots
                or self.generator.recurrent_cache is not self.native
                or (not self.closing and getattr(self.generator, 'disk_page_cache', None) is not self.disk)
                or (not self.closing and getattr(self.native, 'disk_tier', None) is not self.disk)
                or (self.attached and getattr(self.generator, '_glm_b2_retention', None) is not self)):
            self._terminal(RuntimeError('B2 native checkpoint/coordinator owner changed'))

    def _terminal(self, error):
        if self.failure is None:
            self.failure = error
        disk = getattr(self.generator, 'disk_page_cache', None)
        if disk is not None:
            disk.abort()
        raise StopSharedGeneration('Shared B2 native custody is uncertain') from error

    def _fence(self):
        self.check_owner()
        try:
            self.generator.model.tp_drain_acks()
        except BaseException as error:
            self._terminal(error)

    def _record(self, job, phase):
        record = self.jobs.get(job)
        if record is None or record.job is not job or record.phase != phase:
            raise ValueError('Job is not in the required B2 lifecycle phase')
        return record

    def resumable(self, key):
        """A checkpoint a new allocation can restore: still in the native cache, its page chain unbroken."""
        return key in self.native and self.generator.pagetable.is_resumable(key)

    def prepare(self, job):
        """After native hashing and the concurrency admission, before the job joins pending_jobs."""
        self.check_owner()
        if job in self.jobs or len(self.jobs) >= self.slots:
            count = 'two' if self.slots == 2 else 'four'
            raise ValueError(f'B2 paired retention admits {count} distinct jobs')
        record = JobRecord(job)
        self.jobs[job] = record
        if self.shared is not None:
            try:
                self.shared.prepare(job, record)
            except StopSharedGeneration:
                raise
            except BaseException:
                self.jobs.pop(job, None)
                raise

    def _custody(self):
        g = self.generator
        pt = g.pagetable
        return (tuple((id(p), p.ref_count) for p in pt.all_pages),
                tuple(sorted((h, id(p)) for h, p in pt.referenced_pages.items())),
                tuple(sorted(g.cache.free_list)), tuple(sorted(g.draft_cache.free_list)))

    def allocate(self, job, native_allocate, native_deallocate):
        """The native allocation claims pages and restores (or creates) the pair; check what it left behind."""
        self.check_owner(transfer=True)
        record = self._record(job, 'prepared')
        g = self.generator
        before = self._custody()
        try:
            self._tier_batched(native_allocate)(job)
        except StopSharedGeneration:
            raise
        except BaseException as error:
            # Job.allocate_pages assigns its page list and its pair only when each step returns. A fault in
            # between can hide claimed pages or a popped state slot from the job's own cleanup.
            try:
                native_deallocate(job)
            except BaseException as cleanup_error:
                failure = UnsafeCleanupError('Native allocation cleanup uncertain')
                failure.__cause__ = cleanup_error
                self._terminal(failure)
            if self._custody() != before:
                failure = UnsafeCleanupError('Native allocation cleanup left pages or state slots changed')
                failure.__cause__ = error
                self._terminal(failure)
            self.jobs.pop(job, None)
            raise
        seq, pair = job.sequences[0], job.recurrent_state
        position = job.cached_pages * PAGE_SIZE
        if (pair is None or pair.target is None or pair.draft is None or pair.draft.cache is not g.draft_cache
                or pair.target.position != position or pair.draft.position != position
                or seq.kv_position != position):
            self._terminal(ValueError('Native allocation restored a pair that disagrees with its cached prefix'))
        record.phase = 'active'
        # The deepest checkpoint the cached K/V prefix covered; a shallower resume is a native fetch fallback.
        cached = next((i for i, p in enumerate(seq.allocated_pages) if p.kv_position != PAGE_SIZE),
                      len(seq.allocated_pages))
        hashes = seq.page_hashes[:min(cached, len(seq.page_hashes))]
        deepest = max((i + 1 for i, h in enumerate(hashes) if h in self.native), default=0)
        self.metrics['plans'] += deepest > 0
        self.metrics['allocation_cold_fallbacks'] += deepest > job.cached_pages
        if position:
            self.metrics['hits'] += 1
            self.metrics['restored_tokens'] += position
        else:
            self.metrics['cold_misses'] += 1
        if self.shared is not None:
            self.shared.allocated(job, record, position)

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

    def stash(self, job, cache, native_stash, *args, **kwargs):
        """Native put of the pair at a checkpoint boundary; a fault in it latches the shared generator (U10)."""
        self.check_owner(transfer=True)
        if cache is not self.native:
            self._terminal(ValueError('Checkpoint stash cache is not the owned native cache'))
        self._record(job, 'active')
        seq = job.sequences[0]
        pos = seq.kv_position
        interval = kwargs.get('interval', args[0] if args else None)
        if pos == 0 or job.last_recurrent_checkpoint_pos == pos or not job.is_checkpoint_boundary(interval):
            return native_stash(job, cache, *args, **kwargs)
        if job.recurrent_state.checkpoint_size > cache.max_size:
            # The upstream budget guard deliberately publishes no stash. The running job stays valid.
            return native_stash(job, cache, *args, **kwargs)
        try:
            require_settled_replay(job)
        except UnsafeCleanupError as error:
            self._terminal(error)
        if pos % PAGE_SIZE or job.recurrent_state.position != pos:
            self._terminal(ValueError('Checkpoint boundary and recurrent state disagree'))
        key = seq.allocated_pages[pos // PAGE_SIZE - 1].phash
        old = cache.get(key)
        try:
            # PairedState.stash dispatches the target side's rank copies first, then copies the draft window to
            # host; RecurrentCache._insert may evict (and release) older pairs before that. A fault anywhere in it
            # can strand a rank-side handle no cache entry refers to: latch rather than reap one job.
            result = native_stash(job, cache, *args, **kwargs)
            self._fence()
        except StopSharedGeneration:
            raise
        except BaseException as error:
            self._terminal(error)
        saved = cache.get(key)
        if job.last_recurrent_checkpoint_pos != pos or saved is None:
            self._terminal(ValueError('Native stash did not publish the checkpoint'))
        if old is not None:
            if saved is not old:
                self._terminal(ValueError('An existing native checkpoint changed during put'))
            # Native put keeps the existing pair under this page hash (an identical prefix): the analogue of B2's
            # shared-cookie reuse.
            self.metrics['shared_cookie_reuses'] += 1
            return result
        draft = saved.get('draft')
        if (saved.get('position') != pos or saved.get('checkpoint_size') != job.recurrent_state.checkpoint_size
                or native_handle(saved) is None or not isinstance(draft, dict) or draft.get('position') != pos
                or saved['target'].get('position') != pos):
            self._terminal(ValueError('New native stash did not publish the exact paired checkpoint'))
        self.metrics['captures'] += 1
        return result

    def finish(self, results):
        self.check_owner()
        for result in results:
            job = result.get('job')
            record = self.jobs.get(job)
            if record is None or not (result.get('stage') == 'error' or result.get('eos')):
                continue
            del self.jobs[job]
            if self.shared is not None:
                self.shared.finish(record, result.get('stage') != 'error')

    def cancel(self, job, native_cancel):
        """Caller holds the actual async step lock; the native release precedes the record."""
        self.check_owner()
        if self.shared is not None:
            self.shared.drop_ticket(job)
        if job not in self.jobs:
            return native_cancel(job)
        try:
            result = native_cancel(job)
            record = self.jobs.pop(job)
            if self.shared is not None:
                self.shared.finish(record, False)
        except StopSharedGeneration:
            raise
        except BaseException as error:
            self._terminal(error)
        self.metrics['cancellations'] += 1
        return result

    def cleared(self, jobs):
        for job in jobs:
            record = self.jobs.pop(job, None)
            if record is not None and self.shared is not None:
                self.shared.finish(record, False)

    def stats(self):
        native, metrics = self.native, getattr(self.native, 'metrics', {})
        evictions, pruned = metrics.get('stash_evictions', 0), metrics.get('stash_pruned', 0)
        shared = self.shared
        return {'mode': 'b2_native_paired', 'target_cpu_tier': None if self.tier is None else self.tier.stats(),
                'active_job_records': len(self.jobs),
                'completed_checkpoints': len(native),
                'shared_prefix_enabled': shared is not None,
                'shared_pending_tickets': 0 if shared is None else len(shared.tickets),
                'shared_completed_checkpoints': 0 if shared is None else len(shared.records),
                'paired_host_bytes': native.current_size,
                'paired_budget_bytes': native.max_size,
                'reserved_snapshot_bytes': 0,
                'terminal': self.failure is not None,
                'paired_evictions': evictions,
                'max_checkpoints': None,
                'kv_only_pages': self.generator.pagetable.metrics.get('alloc_kv_only_pages', 0),
                'snapshot_budget_skips': metrics.get('stash_skipped_budget', 0),
                **self.metrics,
                'native_releases': evictions + pruned, 'pruned': pruned}

    def retire_after_native_close(self):
        """Native close released every checkpoint (both sides, every rank); no future slot reuse allowed."""
        if len(self.native):
            self._terminal(RuntimeError('Native generator close left checkpoint handles alive'))
        if self.shared is not None:
            self.shared.close()
        self.closed = True
        self.jobs.clear()


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
                    if coordinator is not None and coordinator.failure is not None:
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
                if coordinator is not None and coordinator.failure is not None:
                    raise self.latch_shared(error) from error
                raise
    return PairedAsyncGenerator


def bind_b2_retention(generator):
    """Attach B2 to a fresh generator whose native concurrency guard is bound (glm_native_concurrency)."""
    concurrency = getattr(generator, '_glm_native_concurrency', None)
    if concurrency is None or concurrency.generator is not generator:
        raise ValueError('A source-bound native concurrency guard must be bound before paired retention')
    return B2Retention(concurrency).bind()
