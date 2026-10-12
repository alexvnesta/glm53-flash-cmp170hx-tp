"""Experimental per-job DFlash ring partitions for native continuous batching.

Target paging and each job's recurrent handle remain native EXL3. New requests
prefill cold without resetting the shared page table. Importing does not load
Torch or initialize CUDA. This is separate from single-active paired retention.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import inspect
import os
import threading

PAGE_SIZE = 256
# Written from the isolated engine tree after applying the pinned upstream fair
# scheduling patch. Every gate is checked before allocation/hook installation.
ENGINE_SOURCE_PINS = {'exllamav3/util/memory.py': '6f69f43cce4f9d988d65bcc6c038d5f1bb25d4fb7a5b068b600cd6656446fac6',
 'exllamav3/util/pinned_arena.py': '6d8b9aabea21302eadfac75d477833a1c7d6c97a4b6c10df8b3eddf0ee42ac87',
 'exllamav3/generator/cpu_cache.py': 'd3697bcc9f7eb022db1f5a66399c03f9377138c8f7c44a8a29c416ed206ab608',
 'exllamav3/cache/recurrent.py': '43c7480cf82d5432697ae1a78f1f462960564e0b17e40d598d65c03443bdcecd',
 'exllamav3/util/tp_lazy_cache.py': 'bf975d582a4baac230392c150df85eed877c633efe6e9f52523f4a8b10b1b0a4',
 'exllamav3/generator/generator.py': 'add0fa6c149c38eaa55ab7ed6cc43505ca32778d61bac9af8ca63e48b7f11181',
 'exllamav3/generator/job.py': 'b587612432e482d262776afb79a6a007355b2cad3ae4b8aaaf52e567d7c4df8e',
 'exllamav3/generator/pagetable.py': '0cde876b9969df7120ed7137b35bff3040dc3a7613c9f5a9547e58da4a9849ea',
 'exllamav3/model/model_tp.py': '6666de72a3c64ca89c49d9d5a03ab0ec6290a9d83fb101285c11da91510b354c',
 'exllamav3/model/model_tp_fn.py': '6c7a675b50ecfa0b68b167402be6573b470abf5ec43493c63e5c2b53d3655798',
 'exllamav3/cache/cache.py': '7bf22e342cc549385e5b112ddc5a031fd74e22221379abd2fc39795e1916e674',
 'exllamav3/architecture/dflash2.py': '01f92e590899d3b03f83cc2d652137865ce855d3bc6cc3954f15220a3876d00a',
 'exllamav3/modules/arch_specific/dflash2.py': '9381793440fb240f877e07b5f6753d32a592cf9f53fa3a4dab3603d98234823f',
 'exllamav3/util/tp_sharing.py': 'f192529a38088da8856f8150d32a8cd386a54ac9dd8da99055de7a64b649b9d8',
 'exllamav3/cache/recurrent_util.py': '5fb6a69fe7c51f552c9345c6e326826181c14859229a56b64e04391015917a9b',
 'exllamav3/modules/gated_delta_net.py': '1c9b8d4ac352f51c25f81c2591af8081ba9790d5358e2069ae7b0701be344120',
 'exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py': 'a47e3ac514277ca77e73db51ee6306515dc5f1fbb24747dc7d7da0538823c99d',
 'exllamav3/modules/mamba2.py': '465c8b27d0d2a34f9bd2be04d24cf91c9451d5ee1e99e1270be19e8138cf67eb',
 'exllamav3/modules/short_conv.py': 'a9bd611e9bc30ecda794cadc3db78698f8949e6042842f8f45486f872cf3842f',
 'exllamav3/exllamav3_ext/gdn.cu': 'ddbfec35cb1ecf92176d206f695a4476a306683baa8a77a5048fdb4f52a60c87',
 'exllamav3/exllamav3_ext/gdn.cuh': '7bc9a32432e589025082dd616c28acbab09152a6fb98bdd8bcb6ec0c0b1d51a1',
 'exllamav3/exllamav3_ext/bindings.cpp': 'd1b3e620bf9a4ae2679ffa137fad3976439b29cb11b16e69d598b7853bd79b90',
 'exllamav3/exllamav3_ext/graph.cuh': '79102c39181544c5aeb8c536717d01e9c5f7a69f662fa566f19fc90d64b8a63f',
 'exllamav3/exllamav3_ext/libtorch/gated_delta_net.cpp': '3755a3eda10b1499af242cbbeb0208c1a50901baf3d26bef50a9c33a63e25710',
 'exllamav3/exllamav3_ext/libtorch/gated_delta_net.h': 'a26c23b418321be9f8fad3ed502fbf11845ceaefb1e391b71487af3dd6f29859',
 'exllamav3/exllamav3_ext/libtorch/gated_delta_net_bc.h': '9acba74a00cb636075292b791dca453bae979237361b2637f192bad3d6ccf955',
 'exllamav3/ext.py': '893f563b52ece8a0f71bdf71eca4db340e6fa837505570e52a81450cc22d0fc0'}


def verify_engine_sources(root):
    root = Path(root).resolve(strict=True)
    for relative, expected in ENGINE_SOURCE_PINS.items():
        path = root / relative
        if path.resolve(strict=True) != path or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Concurrent engine source mismatch: {relative}")


def _stream(device):
    import torch
    return int(torch.cuda.current_stream(device).cuda_stream)


# exllamav3/tokenizer/mm_embedding.py (hash-pinned with generator.py and job.py by glm_vision.load_tower)
FIRST_MM_EMBEDDING_INDEX = 1000000000


class ConcurrentRing:
    def __init__(self, generator, slots, stream_reader=_stream, allow_embeddings=False, drop_embeddings=False):
        if type(slots) is not int or not 2 <= slots <= 8:
            raise ValueError("Concurrent ring slots must be 2..8")
        if type(allow_embeddings) is not bool:
            raise ValueError("allow_embeddings must be a bool")
        if type(drop_embeddings) is not bool or (drop_embeddings and not allow_embeddings):
            raise ValueError("drop_embeddings must be a bool and needs allow_embeddings")
        g = generator
        cache = g.draft_cache
        if (not g.dflash_draft or not g.model.loaded_tp or g.max_batch_size != slots
                or g.recurrent_cache is None or g.cache.num_slots < slots
                or g.enable_defrag or g.num_remaining_jobs()
                or getattr(g.pagetable, 'cpu_tier', None) is not None
                or getattr(g, '_glm_dflash_prefix_cache', None) is not None
                or getattr(g, '_glm_tp_active_host_coordinator', None) is not None
                or getattr(g.model, '_glm_tp_active_host_coordinator', None) is not None
                or os.environ.get('GLM53_TP_ACTIVE_HOST', '0') not in ('', '0')
                or getattr(cache, 'dflash_ring_slots', None) != slots):
            raise ValueError("Concurrent rings require idle TP, exact batch/state capacity, no defrag or retention")
        self.generator = g
        self._paired_owner = None
        self.model = g.model
        self.target_cache = g.cache
        self.draft_model = g.draft_model
        self.cache = cache
        self.pagetable = g.pagetable
        self.all_pages = g.pagetable.all_pages
        self.pages = tuple(self.all_pages)
        self.cache_serial = g.cache_owner_serial
        self.target_tokens = g.cache.max_num_tokens
        self.draft_tokens = cache.max_num_tokens
        self.native_slots = g.cache.num_slots
        self.slots = slots
        self.ring_pages = cache.dflash_ring_tokens // PAGE_SIZE
        self.storage = tuple(cache.get_all_tensors())
        if (self.ring_pages < (2048 + g.max_chunk_size + 8 + PAGE_SIZE - 1) // PAGE_SIZE
                or not self.storage or any(t is None or t.shape[0] != slots * self.ring_pages
                                           or t.shape[1] != PAGE_SIZE for t in self.storage)):
            raise ValueError("Concurrent draft physical geometry does not match the isolated ring partitions")
        self.stream_reader = stream_reader
        self.storage_schema = self._storage_schema()
        self.devices = tuple(sorted({t.device for t in self.storage}, key=str))
        self.thread = None
        self.streams = None
        self.depth = 0
        self.prefill_job = None
        self.decode_tables = None
        self.leases = {}
        self.free = list(range(slots))
        self.tables = {}
        self.closed = False
        self.peak_leases = 0
        self.batched_draft_rounds = 0
        self.peak_draft_batch = 0
        self.peak_target_batch = 0
        self.batched_verifier_rounds = 0
        self.mixed_prefill_steps = 0
        self.reused_slots = 0
        self.decode_table_rebinds = 0
        self.ever_used = set()
        # --experimental-vision: image jobs carry MMEmbeddings (content-cached, so equal ids mean equal content).
        self.allow_embeddings = allow_embeddings
        # Set a job's embeddings aside once its prompt is prefilled (drop_prefilled_embeddings).
        self.drop_embeddings = drop_embeddings
        self.embeddings_dropped = 0

    def _storage_schema(self):
        return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.data_ptr())
                     for t in self.storage)

    def _ownership_error(self, error):
        paired = self._paired_owner
        if paired is not None:
            paired._terminal(paired.failure or error)
        raise error

    def check_owner(self, transfer=False):
        paired = self._paired_owner
        if paired is not None and (paired.failure is not None or paired.pairs.terminal):
            self._ownership_error(paired.failure or RuntimeError("Paired ring is terminal"))
        try:
            return self._check_owner(transfer=transfer)
        except Exception as error:
            # Keep the unpaired B2 exception path unchanged. Only the exact
            # published paired owner converts proof loss to shared quarantine.
            self._ownership_error(error)

    def _check_owner(self, transfer=False):
        g = self.generator
        if (self.closed or getattr(g, "_glm_b2_retention", None) is not self._paired_owner
                or g.model is not self.model or g.cache is not self.target_cache
                or not g.model.loaded_tp
                or g.draft_model is not self.draft_model or self.cache.model is not self.draft_model
                or g.draft_cache is not self.cache or g.pagetable is not self.pagetable
                or self.pagetable.generator is not g or self.pagetable.cache is not self.target_cache
                or self.pagetable.all_pages is not self.all_pages
                or len(self.all_pages) != len(self.pages)
                or any(a is not b or a.page_index != i for i, (a, b) in enumerate(zip(self.all_pages, self.pages)))
                or g.cache_owner_serial != self.cache_serial
                or getattr(g.cache, 'owner_serial', None) != self.cache_serial
                or g.max_batch_size != self.slots or g.enable_defrag
                or g.cache.max_num_tokens != self.target_tokens
                or self.cache.max_num_tokens != self.draft_tokens
                or g.cache.num_slots != self.native_slots
                or self.cache.dflash_ring_tokens != self.ring_pages * PAGE_SIZE
                or self.cache.dflash_ring_slots != self.slots
                or getattr(self.pagetable, 'cpu_tier', None) is not
                    (None if self._paired_owner is None else self._paired_owner.tier)
                or getattr(g, 'cpu_page_cache', None) is not
                    (None if self._paired_owner is None else self._paired_owner.tier)
                or getattr(g, '_glm_dflash_prefix_cache', None) is not None
                or tuple(map(id, self.cache.get_all_tensors())) != tuple(map(id, self.storage))
                or self._storage_schema() != self.storage_schema
                or getattr(g, '_glm_tp_active_host_coordinator', None) is not None
                or getattr(g.model, '_glm_tp_active_host_coordinator', None) is not None
                or os.environ.get('GLM53_TP_ACTIVE_HOST', '0') not in ('', '0')):
            raise RuntimeError("Concurrent ring owner, cache geometry or storage changed")
        if transfer:
            if self.depth != 1 or self.thread != threading.get_ident():
                raise RuntimeError("Concurrent ring transfer is outside the persistent step owner")
            if tuple(self.stream_reader(d) for d in self.devices) != self.streams:
                raise RuntimeError("Concurrent draft stream changed before ring reuse")

    @contextmanager
    def step(self):
        self.check_owner()
        thread = threading.get_ident()
        if self.thread is None:
            self.thread = thread
            self.streams = tuple(self.stream_reader(d) for d in self.devices)
        if self.thread != thread or self.depth:
            self._ownership_error(RuntimeError("Concurrent generator step owner changed or nested"))
        self.depth = 1
        try:
            self.check_owner(transfer=True)
            yield
        finally:
            self.prefill_job = None
            self.decode_tables = None
            self.depth = 0

    @contextmanager
    def prefill(self, job):
        self.check_owner(transfer=True)
        if self.prefill_job is not None or self.decode_tables is not None:
            raise RuntimeError("Nested concurrent prefill")
        self.prefill_job = job
        try:
            yield
        finally:
            self.prefill_job = None

    @contextmanager
    def decode(self, jobs):
        # Native iterate_gen copies these tables into its compact batch before
        # the verifier forward and passes that same batch to the post-sampling
        # draft update. Sampling may rebind a row in between: hash_completed_page
        # switches a just-completed page to the identical page another active
        # job references and rebuilds block_index_tensor. Prove draft rows
        # against the tables the batch was built from.
        self.check_owner(transfer=True)
        if self.prefill_job is not None or self.decode_tables is not None:
            raise RuntimeError("Nested concurrent decode")
        self.decode_tables = tuple((job, job.sequences[0].block_index_tensor.clone()) for job in jobs)
        try:
            yield
        finally:
            self.decode_tables = None

    def prepare(self, job):
        self.check_owner()
        if (len(job.sequences) != 1 or job.is_requeued or (job.embeddings and not self.allow_embeddings) or job.filters
                or job.banned_strings or job.prefix_token is not None
                or job.orig_max_rq_tokens is not None):
            raise ValueError("Concurrent initial profile supports plain, non-requeued single-sequence jobs")
        # Recompute every newly admitted job. This is a per-sequence fence, not
        # a page-table reset that could invalidate the other live sequence.
        job.sequences[0].max_cached_pages = 0
        job.all_unique_hashes = []

    def claim(self, job):
        self.check_owner(transfer=True)
        if job in self.leases or not self.free:
            raise RuntimeError("Concurrent ring lease capacity/ownership failure")
        slot = self.free.pop(0)
        self.leases[job] = slot
        if slot in self.ever_used:
            self.reused_slots += 1
        self.ever_used.add(slot)
        self.peak_leases = max(self.peak_leases, len(self.leases))

    def release(self, job):
        # Native deallocation follows all draft updates; cancellation owns the
        # async step lock first. No CUDA work occurs here. The next slot user
        # writes on the same fixed worker/device streams after prior queued work.
        self.check_owner()
        slot = self.leases.pop(job, None)
        if slot is not None:
            self.free.append(slot)
            self.free.sort()

    def params(self, params, *, draft_forward=False):
        import torch
        self.check_owner(transfer=True)
        table = params['block_table']
        rows = ([self.prefill_job] if self.prefill_job is not None else
                [job for job in self.generator.active_jobs if job.is_prefill_done()])
        built = self.decode_tables if self.prefill_job is None else None
        if (table.device.type != 'cpu' or table.dtype != torch.int32 or table.dim() != 2
                or table.shape[0] != len(rows) or not rows
                or any(len(job.sequences) != 1 for job in rows)
                or (built is not None and (len(built) != len(rows)
                                           or any(job is not own for job, (own, _) in zip(rows, built))))):
            raise RuntimeError("Draft block-table rows do not match the compact active job order")
        slots = []
        for row, job in enumerate(rows):
            if job not in self.leases:
                raise RuntimeError("Draft row has no live per-job ring lease")
            seq_table = job.sequences[0].block_index_tensor if built is None else built[row][1]
            width = min(table.shape[1], seq_table.shape[1])
            if not torch.equal(table[row, :width], seq_table[0, :width]):
                bad = (table[row, :width] != seq_table[0, :width]).nonzero().flatten()[:4].tolist()
                raise RuntimeError("Draft block table belongs to a different sequence "
                                   f"(row {row} job {getattr(job, 'serial_number', None)} rows "
                                   f"{[getattr(j, 'serial_number', None) for j in rows]} "
                                   f"decode {built is not None} columns {bad})")
            if built is not None:
                self._check_rebind(job, seq_table)
            slots.append(self.leases[job])
        if len(set(slots)) != len(slots):
            raise RuntimeError("Two draft rows alias one ring partition")
        key = (table.shape[1], tuple(slots))
        if key not in self.tables:
            columns = torch.arange(table.shape[1], dtype=torch.int32) % self.ring_pages
            ring = torch.stack([columns + slot * self.ring_pages for slot in slots])
            ring._static_dev_cache = True
            self.tables[key] = ring
        if draft_forward:
            self.peak_draft_batch = max(self.peak_draft_batch, len(rows))
            self.batched_draft_rounds += len(rows) > 1
        return {**params, 'block_table': self.tables[key]}

    def _check_rebind(self, job, built):
        # The only native table change between batch build and draft update is
        # the completed-page dedupe: same length, and each changed column now
        # names the row's own complete page strictly behind its position.
        import torch
        seq = job.sequences[0]
        live = seq.block_index_tensor
        if live.shape == built.shape and torch.equal(live, built):
            return
        changed = [] if live.shape != built.shape else (live[0] != built[0]).nonzero().flatten().tolist()
        if (not changed or any(c >= seq.kv_position // PAGE_SIZE
                               or seq.allocated_pages[c].page_index != live[0, c].item()
                               or seq.allocated_pages[c].kv_position != PAGE_SIZE for c in changed)):
            raise RuntimeError("Draft row block table changed outside completed-page dedupe "
                               f"(job {getattr(job, 'serial_number', None)} columns {changed[:4]})")
        self.decode_table_rebinds += 1

    def stats(self):
        return dict(mode='concurrent_cold' if self._paired_owner is None else 'concurrent_paired', max_sequences=self.slots,
                    ring_tokens_per_sequence=self.ring_pages * PAGE_SIZE,
                    draft_physical_tokens=self.slots * self.ring_pages * PAGE_SIZE,
                    active_ring_leases=len(self.leases), peak_ring_leases=self.peak_leases,
                    peak_draft_batch=self.peak_draft_batch,
                    batched_draft_rounds=self.batched_draft_rounds,
                    peak_target_batch=self.peak_target_batch,
                    batched_verifier_rounds=self.batched_verifier_rounds,
                    mixed_prefill_steps=self.mixed_prefill_steps, reused_slots=self.reused_slots,
                    decode_table_rebinds=self.decode_table_rebinds,
                    paired_retention=self._paired_owner is not None,
                    **({"embeddings_dropped": self.embeddings_dropped} if self.allow_embeddings else {}))

    def drop_prefilled_embeddings(self, jobs):
        """The native decode batch passes every active job's embeddings on every step (generator.iterate_gen),
        which under TP copies them into the worker arena each time. After prefill nothing reads them: GLM-5 has no
        MRoPE, ring jobs are never requeued (prepare refuses it), and only MTP drafting feeds them to a draft model.
        A job keeps them while any image id is still unfed. The list object stays with the content cache."""
        if self.generator.mtp_draft or self.model.caps.get("mrope"):
            return
        for job in jobs:
            if not job.embeddings:
                continue
            unfed = [seq.sequence_ids.torch_slice(seq.kv_position, None) for seq in job.sequences]
            if any(ids.numel() and int(ids.max()) >= FIRST_MM_EMBEDDING_INDEX for ids in unfed):
                continue
            job.embeddings = []
            self.embeddings_dropped += 1

    def close(self):
        self.closed = True
        self.leases.clear()
        self.tables.clear()


def install_concurrent_rings(slots=2, tokens=8192):
    """Install cache factory and instance-gated hooks before model initialization."""
    import torch
    from exllamav3.cache import Cache, CacheLayer_fp16, CacheLayer_quant
    from exllamav3.architecture.dflash2 import DFlash2Model
    from exllamav3.generator.generator import Generator
    from exllamav3.generator.job import Job
    import exllamav3
    verify_engine_sources(Path(exllamav3.__file__).resolve().parent.parent)
    if os.environ.get('GLM53_TP_ACTIVE_HOST', '0') not in ('', '0'):
        raise ValueError("Concurrent rings exclude TP active-host migration")
    if type(slots) is not int or not 2 <= slots <= 8 or tokens != 8192:
        raise ValueError("Concurrent initial profile requires 2..8 isolated 8192-token rings")
    original = Cache.__init__
    if getattr(original, '_glm_concurrent_ring', False):
        if original._glm_concurrent_slots != slots:
            raise ValueError("Concurrent cache policy was already installed with different capacity")
        return
    if getattr(original, '_glm_dflash_ring', False):
        raise ValueError("Single-sequence ring hook is already installed")
    signature = inspect.signature(original)

    class RingFP16(CacheLayer_fp16):
        def __init__(self, config, attention, cache_id, max_num_tokens, **kw):
            super().__init__(config, attention, cache_id, tokens * slots, **kw)

    class RingQ8(CacheLayer_quant):
        def __init__(self, config, attention, cache_id, max_num_tokens, **kw):
            super().__init__(config, attention, cache_id, tokens * slots, **kw)

    def init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        if isinstance(bound.arguments['model'], DFlash2Model):
            model = bound.arguments['model']
            if (bound.arguments['max_num_tokens'] < tokens * slots
                    or any(a.sliding_window != 2047 for a in model.attn_modules)):
                raise ValueError("Concurrent draft requires sufficient physical capacity and all-SWA2048 layers")
            layer = bound.arguments.get('layer_type') or CacheLayer_fp16
            if layer not in (CacheLayer_fp16, CacheLayer_quant):
                raise ValueError("Unsupported concurrent draft cache layer")
            bound.arguments['layer_type'] = RingQ8 if layer is CacheLayer_quant else RingFP16
            bound.arguments['max_batch_size'] = slots
            self.dflash_ring_tokens = tokens
            self.dflash_ring_slots = slots
        return original(*bound.args, **bound.kwargs)
    init._glm_concurrent_ring = True
    init._glm_concurrent_slots = slots
    Cache.__init__ = init

    forward, update = DFlash2Model.forward, DFlash2Model.update_kv_from_target
    def forward_hook(self, input_ids, params, **kwargs):
        cache = params.get('cache')
        manager = getattr(cache, '_glm_concurrent_ring', None)
        if manager is None:
            if getattr(cache, 'dflash_ring_slots', None):
                raise RuntimeError("Concurrent cache has no owning generator")
            return forward(self, input_ids=input_ids, params=params, **kwargs)
        table = params['block_table']
        params['block_table'] = manager.params(params, draft_forward=True)['block_table']
        try:
            return forward(self, input_ids=input_ids, params=params, **kwargs)
        finally:
            params['block_table'] = table
    def update_hook(self, target_hidden, cache, params, lengths=None):
        manager = getattr(cache, '_glm_concurrent_ring', None)
        if manager is None:
            if getattr(cache, 'dflash_ring_slots', None):
                raise RuntimeError("Concurrent cache has no owning generator")
            return update(self, target_hidden=target_hidden, cache=cache, params=params, lengths=lengths)
        return update(self, target_hidden=target_hidden, cache=cache,
                      params=manager.params(params), lengths=lengths)
    DFlash2Model.forward, DFlash2Model.update_kv_from_target = forward_hook, update_hook

    prepare, allocate, prefill, deallocate = Job.prepare_for_queue, Job.allocate_pages, Job.prefill, Job.deallocate_pages
    stash = Job.maybe_stash_recurrent
    cancel, clear_queue = Generator.cancel, Generator.clear_queue
    iterate, iterate_gen, close = Generator.iterate, Generator.iterate_gen, Generator.close
    def prepare_hook(self, g, *args, **kwargs):
        result = prepare(self, g, *args, **kwargs)
        manager = getattr(g, '_glm_concurrent_ring', None)
        if manager is not None:
            manager.prepare(self)
            paired = getattr(g, "_glm_b2_retention", None)
            if paired is not None:
                paired.prepare(self)
        return result
    def allocate_hook(self):
        manager = getattr(self.generator, '_glm_concurrent_ring', None)
        if manager is None:
            return allocate(self)
        paired = getattr(self.generator, "_glm_b2_retention", None)
        if paired is not None:
            return paired.allocate(self, allocate, deallocate)
        manager.claim(self)
        try:
            return allocate(self)
        except BaseException:
            manager.release(self)
            raise
    def prefill_hook(self, results, *args, **kwargs):
        manager = getattr(self.generator, '_glm_concurrent_ring', None)
        if manager is None:
            return prefill(self, results, *args, **kwargs)
        mixed = any(job is not self and job.is_prefill_done() for job in self.generator.active_jobs)
        before = len(results)
        from exllamav3.modules.gated_delta_net import RecurrentReplayError
        try:
            with manager.prefill(self):
                paired = getattr(self.generator, "_glm_b2_retention", None)
                shared = None if paired is None else paired.shared
                result = (prefill(self, results, *args, **kwargs) if shared is None else
                          shared.prefill(self, results, prefill, *args, **kwargs))
        except RecurrentReplayError as error:
            paired = getattr(self.generator, "_glm_b2_retention", None)
            if paired is not None:
                paired._terminal(error)
            raise
        if mixed and any(event.get('stage') == 'prefill' for event in results[before:]):
            manager.mixed_prefill_steps += 1
        return result
    def deallocate_hook(self):
        paired = getattr(self.generator, "_glm_b2_retention", None)
        if paired is not None:
            return paired.deallocate(self, deallocate)
        result = deallocate(self)
        manager = getattr(self.generator, '_glm_concurrent_ring', None)
        if manager is not None:
            manager.release(self)
        return result
    def iterate_hook(self):
        manager = getattr(self, '_glm_concurrent_ring', None)
        if manager is None:
            return iterate(self)
        from glm_b2_retention_coordinator import StopSharedGeneration, shared_failure
        try:
            with manager.step():
                result = iterate(self)
                paired = getattr(self, "_glm_b2_retention", None)
                if paired is not None:
                    paired.finish(result)
                return result
        except StopSharedGeneration as error:
            raise shared_failure(error) from error
    def iterate_gen_hook(self, *args, **kwargs):
        manager = getattr(self, '_glm_concurrent_ring', None)
        if manager is None:
            return iterate_gen(self, *args, **kwargs)
        manager.check_owner(transfer=True)
        jobs = [job for job in self.active_jobs if job.is_prefill_done()]
        rows = len(jobs)
        if manager.drop_embeddings:
            manager.drop_prefilled_embeddings(jobs)
        from exllamav3.modules.gated_delta_net import RecurrentReplayError
        try:
            with manager.decode(jobs):
                result = iterate_gen(self, *args, **kwargs)
        except RecurrentReplayError as error:
            paired = getattr(self, "_glm_b2_retention", None)
            if paired is not None:
                paired._terminal(error)
            raise
        manager.peak_target_batch = max(manager.peak_target_batch, rows)
        manager.batched_verifier_rounds += rows > 1
        return result
    def stash_hook(self, cache, *args, **kwargs):
        paired = getattr(self.generator, "_glm_b2_retention", None)
        if paired is None:
            return stash(self, cache, *args, **kwargs)
        return paired.stash(self, cache, stash, *args, **kwargs)
    def cancel_hook(self, job):
        paired = getattr(self, "_glm_b2_retention", None)
        if paired is None:
            return cancel(self, job)
        return paired.cancel(job, lambda owned_job: cancel(self, owned_job))
    def clear_hook(self):
        paired = getattr(self, "_glm_b2_retention", None)
        if paired is None:
            return clear_queue(self)
        jobs = tuple(self.active_jobs + self.pending_jobs)
        result = clear_queue(self)
        for job in jobs:
            if job in paired.pairs.jobs:
                paired.pairs.finish(job, success=False)
        return result
    def close_hook(self):
        paired = getattr(self, "_glm_b2_retention", None)
        if paired is None:
            # Preserve the original B2 default close behavior exactly.
            try:
                return close(self)
            finally:
                manager = getattr(self, '_glm_concurrent_ring', None)
                if manager is not None:
                    manager.close()
        # The actual async wrapper has settled its uncancellable iteration first.
        # Clear paired arrays only after native close returned with no checkpoints.
        result = close(self)
        paired = getattr(self, "_glm_b2_retention", None)
        if paired is not None:
            paired.retire_after_native_close()
        manager = getattr(self, '_glm_concurrent_ring', None)
        if manager is not None:
            manager.close()
        return result
    Job.prepare_for_queue, Job.allocate_pages = prepare_hook, allocate_hook
    Job.prefill, Job.deallocate_pages = prefill_hook, deallocate_hook
    Job.maybe_stash_recurrent = stash_hook
    Generator.cancel, Generator.clear_queue = cancel_hook, clear_hook
    Generator.iterate, Generator.iterate_gen, Generator.close = iterate_hook, iterate_gen_hook, close_hook


def bind_concurrent_rings(generator, slots, allow_embeddings=False, drop_embeddings=False):
    old_defrag = generator.enable_defrag
    if (generator.num_remaining_jobs() or getattr(generator, '_glm_concurrent_ring', None) is not None
            or getattr(generator.draft_cache, '_glm_concurrent_ring', None) is not None):
        raise ValueError("Concurrent rings require a fresh idle generator")
    generator.enable_defrag = False
    try:
        manager = ConcurrentRing(generator, slots, allow_embeddings=allow_embeddings, drop_embeddings=drop_embeddings)
        generator.draft_cache._glm_concurrent_ring = manager
        generator._glm_concurrent_ring = manager
        return manager
    except BaseException:
        if locals().get("manager") is not None and getattr(generator.draft_cache, "_glm_concurrent_ring", None) is manager:
            del generator.draft_cache._glm_concurrent_ring
        if locals().get("manager") is not None and getattr(generator, "_glm_concurrent_ring", None) is manager:
            del generator._glm_concurrent_ring
        generator.enable_defrag = old_defrag
        raise
