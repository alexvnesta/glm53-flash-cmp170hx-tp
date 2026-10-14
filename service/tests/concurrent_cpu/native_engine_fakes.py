"""CPU fake engine around upstream's real paging, checkpoint and sliding-window state code (P16 stage 2).

Compiled from the pinned engine source with ast (no extension import, no CUDA), each in its own namespace:
- generator/pagetable.py: CachePage, Sequence, PageTable and the page hashing helpers; util/tensor.py: SeqTensor;
- cache/recurrent.py: the whole module (HostPool, host_copy, PairedState, RecurrentCache, the per-rank mp_* functions);
- modules/sliding_attn.py: SWAState, SWALayerState and SlidingAttention.plan_state_rows / write_state_rows /
  commit_state_rows / _decode_state_prep (the drafter's window rings, on CPU tensors);
- cache/cache.py: Cache.get_new_state / new_from_stashed / release_state / reset_states / get_recurrent_layer /
  get_all_recurrent_layers (state slots);
- generator/job.py: Job.prepare_for_queue, prefill, allocate_pages, deallocate_pages, free_recurrent_state,
  maybe_stash_recurrent, is_checkpoint_boundary, hash_completed_page and the batch helpers;
- generator/generator.py: Generator.enqueue, iterate, iterate_start_jobs, recurrent_checkpoint,
  iterate_draftmodel_dflash_gen, iterate_hybrid_gen, iterate_gen, reap_failed_job, cancel, clear_queue,
  on_queue_drained, close, new_recurrent_state, recurrent_state_from_stashed.

Doubles replace model math only:
- the target is a recurrence over token ids (value = step(value, token)); its greedy token is a function of the value,
  so a wrongly restored target state changes the output. Its recurrent state is GDN-shaped (slot, last_history,
  parity, spec_row/spec_shape, a rank-side stash under a TP handle in a per-rank dict);
- the drafter writes K/V rows that encode their absolute position into the real SWA rings (prefill and the accepted
  rows through the real plan/write/commit; the draft block through the real _decode_state_prep). It proposes the
  target's true continuation only when the window it attends to holds exactly the last positions, so a corrupted
  window (U1) shows up as rejected drafts, never as wrong output;
- receive_sample reproduces job.py's token accept and completed-page hashing lines.
"""
import ast
import hashlib
import heapq
import os
import sys
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch

ENGINE = Path(os.environ.get('EXLLAMAV3_ENGINE_ROOT', Path(__file__).resolve().parents[3] / 'engine'))
GLM = Path(__file__).resolve().parents[2] / 'glm'          # the tree's own service modules (hooks under test)
if str(GLM) not in sys.path:
    sys.path.append(str(GLM))
PAGE = 256
VOCAB = 100
K = 7                    # -ndt 7
BLOCK = 8                # DFlash2 block
WINDOW = 2047            # GLM-5.3 DFlash2 drafter: sliding_window 2048, non-causal block
OVERP = 2 * PAGE
DRAFT_LAYERS = 2
TARGET_CHECKPOINT_BYTES = 1 << 20
KEEP_DECORATORS = {'property', 'staticmethod', 'dataclass'}


class _DropRelativeImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return ast.Pass() if node.level else node


def compile_engine(relative, env, *, classes=(), functions=(), assigns=(), methods=None, module=False):
    """Execute selected top-level definitions of one engine file in env (decorators other than property /
    staticmethod / dataclass and function-local relative imports removed; their names come from env)."""
    path = ENGINE / relative
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if module:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, 'id', None) in assigns for t in node.targets):
            body.append(node)
        elif methods and isinstance(node, ast.ClassDef) and node.name in methods:
            wanted = set(methods[node.name])
            node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in wanted]
            if {m.name for m in node.body} != wanted:
                raise RuntimeError(f'{relative}: {node.name} lacks {wanted - {m.name for m in node.body}}')
            node.bases, node.keywords = [], []
            body.append(node)
    found = {n.name for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    missing = (set(classes) | set(functions) | set(methods or ())) - found
    if missing:
        raise RuntimeError(f'{relative}: {sorted(missing)} not found')
    for node in body:
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                child.decorator_list = [d for d in child.decorator_list
                                        if isinstance(d, ast.Name) and d.id in KEEP_DECORATORS]
    mod = ast.Module([ast.ImportFrom('__future__', [ast.alias('annotations')], 0)] + body, [])
    mod = _DropRelativeImports().visit(mod)
    exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), env)
    return env


class Engine:
    """The compiled engine pieces (one namespace per source file, wired as the package imports them)."""

    def __init__(self):
        self.rec = rec = dict(torch=torch, OrderedDict=OrderedDict, PAGE_SIZE=PAGE, malloc_trim=lambda: None)
        compile_engine('exllamav3/cache/recurrent.py', rec, module=True)
        self.swa = swa = dict(torch=torch, PAGE_SIZE=PAGE, host_copy=rec['host_copy'],
                              host_to_device=lambda t, device: t.to(device), override=lambda f: f,
                              mp_cache_recurrent_stash=rec['mp_cache_recurrent_stash'],
                              mp_cache_recurrent_unstash=rec['mp_cache_recurrent_unstash'],
                              new_checkpoint_handle=rec['new_checkpoint_handle'])
        compile_engine('exllamav3/modules/sliding_attn.py', swa, classes=('SWAExportedState', 'SWAState', 'SWALayerState'),
                       methods={'SlidingAttention': ('plan_state_rows', 'write_state_rows', 'commit_state_rows',
                                                     '_decode_state_prep')})
        self.cache = cache = dict(torch=torch, deque=deque)
        compile_engine('exllamav3/cache/cache.py', cache, methods={'Cache': (
            'get_new_state', 'new_from_stashed', 'release_state', 'reset_states', 'get_recurrent_layer',
            'get_all_recurrent_layers')})
        self.pt = pt = dict(torch=torch, PAGE_SIZE=PAGE, hashlib=hashlib, heapq=heapq, deque=deque,
                            defaultdict=defaultdict, pairwise=pairwise, dataclass=dataclass, time=time, os=os,
                            FIRST_MM_EMBEDDING_INDEX=1000000000, to_device=lambda t, d: t.to(d))
        compile_engine('exllamav3/util/tensor.py', pt, classes=('SeqTensor',))
        compile_engine('exllamav3/generator/pagetable.py', pt, classes=('CachePage', 'Sequence', 'PageTable'),
                       functions=('_tensor_blake2b_checksum', '_randomhash', 'is_content_hash'),
                       assigns=('_uniquehash', 'tensor_hash_checksum', 'random_hash'))
        self.job = job = dict(torch=torch, PAGE_SIZE=PAGE, time=time, Sequence=pt['Sequence'], SeqTensor=pt['SeqTensor'],
                              tensor_hash_checksum=pt['tensor_hash_checksum'], random_hash=pt['random_hash'])
        compile_engine('exllamav3/generator/job.py', job, methods={'Job': (
            'prepare_for_queue', '_check_banned_strings', 'current_new_pages_required', 'is_prefill_done',
            'get_max_seq_len', 'get_input_ids_list', 'hash_completed_page', 'prefill', 'allocate_pages',
            'deallocate_pages', 'free_recurrent_state', 'maybe_stash_recurrent', 'is_checkpoint_boundary', 'activate',
            'get_hybrid_draft', 'suffix_match')})
        self.gen = gen = dict(torch=torch, PAGE_SIZE=PAGE, time=time, PairedState=rec['PairedState'],
                              host_pool=rec['host_pool'], mp_host_pool_release=rec['mp_host_pool_release'],
                              cuda_sync_active=lambda: None, malloc_trim=lambda: None,
                              run_pending_swap_sweeps=lambda *a: None,
                              logger=NS(error=lambda *a, **k: None))
        compile_engine('exllamav3/generator/generator.py', gen, methods={'Generator': (
            'num_remaining_jobs', 'enqueue', 'iterate', 'iterate_start_jobs', 'reap_failed_job', 'recurrent_checkpoint',
            'iterate_draftmodel_dflash_gen', 'iterate_hybrid_gen', 'iterate_gen', 'cancel', 'clear_queue',
            'on_queue_drained', 'close', 'new_recurrent_state', 'recurrent_state_from_stashed')})


def step(value, token):
    return (value * 1000003 + int(token) + 1) % 2147483647


def greedy(value):
    return 3 + value % 97


def enc(positions):
    """K/V row content: the absolute position (exact in fp16 below 2^11 per component; +1 keeps rows nonzero)."""
    p = torch.as_tensor(positions, dtype=torch.int64)
    return torch.stack(((p // 1024) + 1, (p % 1024) + 1), dim=-1).to(torch.half)


class Ranks:
    """The target's TP ranks: per-rank recurrent_cache dicts holding the rank-side copy under each checkpoint handle."""

    def __init__(self):
        self.recurrent_cache = {}
        self.caches = {}
        self.dispatched = []
        self.fail = None

    def dispatch(self, fn, args):
        self.dispatched.append(fn.__name__)
        if self.fail is not None and self.fail(fn.__name__, args):
            raise RuntimeError(f'injected rank failure in {fn.__name__}')
        return fn({'recurrent_cache': self.recurrent_cache, 'caches': self.caches,
                   'recurrent_modules': []}, *args)


def rank_stash(ctx, cache_id, handle, slot, position):
    cache = ctx['caches'][cache_id]
    ctx['recurrent_cache'][handle] = (position, cache.values[slot][-1])


def rank_unstash(ctx, cache_id, handle, slot, position):
    cache = ctx['caches'][cache_id]
    stashed_position, value = ctx['recurrent_cache'][handle]
    assert stashed_position == position
    cache.values[slot] = [value]


class Scenario:
    """One generator over the fake models, with the real service hooks installed (glm_native_concurrency)."""

    def __init__(self, engine=None, pages=96, slots=2, hybrid=0, sam=None, periodic=False):
        self.e = e = engine or Engine()
        self.slots, self.pages, self.hybrid, self.sam, self.periodic = slots, pages, hybrid, sam, periodic
        if sam is not None:
            e.job['ext'] = NS(BC_SAM=sam)     # what prepare_for_queue instantiates with -hdt
        self.ranks = Ranks()
        self.rounds = []          # per draft forward: [(serial, window exact)]
        self.widths = []          # verify widths
        self.restores = []        # (serial, position, draft window exact after restore)
        scenario = self
        rec, swa = e.rec, e.swa

        class TargetState:
            """GDN-shaped target state (slot, history, parity, spec rows); rank-side stash under a TP handle."""
            exported = False
            replay_failed = False        # glm_replay_quarantine sets this class attribute on GDNState

            def __init__(self, cache, slot, position, clear=True, stashed=None, test_state=False):
                self.cache, self.slot, self.position = cache, slot, position
                self.last_history = 0
                self.parity = 0
                self.spec_row = self.spec_shape = None
                self.checkpoint_size = TARGET_CHECKPOINT_BYTES
                if clear:
                    cache.values[slot] = [0]
                if stashed is not None:
                    self.unstash(stashed)

            def stash(self):
                handle = rec['new_checkpoint_handle']()
                self.cache.model.tp_dispatch_all(rank_stash, (id(self.cache), handle, self.slot, self.position))
                return {'position': self.position, 'checkpoint_size': self.checkpoint_size, 'tp_handle': handle}

            def unstash(self, stashed):
                assert stashed['position'] == self.position
                self.cache.model.tp_dispatch_all(rank_unstash, (id(self.cache), stashed['tp_handle'], self.slot,
                                                                self.position))

            def rewind(self, num_tokens):
                assert num_tokens >= 0
                history = self.cache.values[self.slot]
                if self.last_history > 0:
                    assert num_tokens <= self.last_history
                    if num_tokens:
                        del history[-num_tokens:]
                self.cache.values[self.slot] = [history[-1]]
                self.position -= num_tokens
                self.last_history = 0
                self.parity = 1 - self.parity
                self.spec_row = self.spec_shape = None

            def rollback_capacity(self):
                return 0

            def free(self):
                self.cache.release_state(self)

            def post_advance(self):
                pass

            def reset(self):
                self.position = 0

        class Target:
            caps = {'recurrent_states': True}
            loaded_tp = True
            output_device = torch.device('cpu')
            config = NS(vocab_size=VOCAB, infer_params=None)

            def __init__(self):
                self.recurrent_state_cls = TargetState

            def tp_dispatch_all(self, fn, args):
                return scenario.ranks.dispatch(fn, args)

            def tp_drain_acks(self):
                pass

            def prefetch_tokens(self, tokens):
                pass

            def _advance(self, input_ids, params, history):
                states = params.get('recurrent_states')
                seqlens = params['cache_seqlens']
                logits = torch.full((input_ids.shape[0], input_ids.shape[1], VOCAB), -100.0)
                for r in range(input_ids.shape[0]):
                    st = states[r]
                    assert st.position == int(seqlens[r]), (st.position, int(seqlens[r]))
                    chain = st.cache.values[st.slot]
                    value = chain[-1]
                    for i, t in enumerate(input_ids[r].tolist()):
                        value = step(value, t)
                        chain.append(value)
                        logits[r, i, scenario.next_token(value, t)] = 100.0
                    if not history:
                        st.cache.values[st.slot] = [value]
                    st.position += input_ids.shape[1]
                    st.last_history = input_ids.shape[1] - 1 if history else 0
                    st.spec_row = r if history else None
                    st.spec_shape = (input_ids.shape[0], input_ids.shape[1], 1) if history else None
                params['export_states'] = [torch.zeros(input_ids.shape[0], input_ids.shape[1], 1)]
                return logits

            def prefill(self, input_ids, params):
                self._advance(input_ids, params, False)

            def forward(self, input_ids, params):
                scenario.widths.append(input_ids.shape[1])
                return self._advance(input_ids, params, params.get('recurrent_history', False))

        class Drafter:
            caps = {'dflash_draft': True, 'recurrent_states': True}
            loaded_tp = False
            config = NS(block_size=BLOCK)
            draft_verifier_params = {}

            def __init__(self):
                self.modules = []
                for i in range(DRAFT_LAYERS):
                    m = swa['SlidingAttention'].__new__(swa['SlidingAttention'])
                    m.kv_state_size = -(-(WINDOW + OVERP) // PAGE) * PAGE
                    m.sliding_window, m.num_kv_heads, m.head_dim = WINDOW, 1, 2
                    m.layer_idx, m.bt_cache, m.device = i, {}, torch.device('cpu')
                    self.modules.append(m)
                self.recurrent_state_cls = swa['SWAState']

            def forward(self, input_ids, params):
                rsg = params['recurrent_states']
                rows = [(st, scenario.window_exact(st)) for st in rsg]
                for m in self.modules:
                    ks, vs = scenario.draft_cache.get_recurrent_layer((m.layer_idx, 0)).get_state_tensors()
                    _, seqlens = m._decode_state_prep(rsg, ks, vs, BLOCK)
                    for st, c in zip(rsg, seqlens.tolist()):
                        ks[st.slot, c:c + BLOCK] = -1.0
                        vs[st.slot, c:c + BLOCK] = -1.0
                for st in rsg:       # advance_recurrent_states with the (bsz, 1) anchor ids
                    st.position += input_ids.shape[1]
                    st.last_history = 0
                    st.post_advance()
                return rows

            def sample_from_state(self, rows, params):
                out, observed = [], []
                for st, exact in rows:
                    job = next(j for j in scenario.g.active_jobs
                               if j.recurrent_state is not None and j.recurrent_state.draft is st)
                    seq = job.sequences[0]
                    target = job.recurrent_state.target
                    anchor = int(seq.sequence_ids.torch()[0, seq.kv_position])
                    value = step(target.cache.values[target.slot][-1], anchor)
                    proposal, t = [], anchor
                    for _ in range(K):
                        t = scenario.next_token(value, t)
                        proposal.append(t if exact else 3 + (t - 3 + 1) % 97)
                        value = step(value, t)
                    out.append([0] + proposal)
                    observed.append((job.serial_number, exact))
                scenario.rounds.append(observed)
                return torch.tensor(out, dtype=torch.long)

            def update_kv_from_target(self, target_hidden, cache, params, lengths=None):
                rsg = params['recurrent_states']
                n = target_hidden[0].shape[1]
                positions = params['cache_seqlens'].tolist()
                plan = self.modules[0].plan_state_rows(rsg, positions, lengths, n)
                k = torch.stack([enc(range(p, p + n)) for p in positions]).unsqueeze(2)   # (bsz, n, 1, 2)
                for m in self.modules:
                    m.write_state_rows(rsg, plan, k, k.clone())
                swa['SlidingAttention'].commit_state_rows(rsg, plan)

        self.Target, self.Drafter, self.TargetState = Target, Drafter, TargetState
        self.model, self.drafter = Target(), Drafter()

        CacheMethods = e.cache['Cache']

        def make_cache(model, layers):
            c = CacheMethods.__new__(CacheMethods)
            c.model, c.num_slots, c.max_history = model, slots, 16
            c.max_num_tokens, c.initialized, c.owner_serial = pages * PAGE, True, 1
            c.recurrent_state_cls = model.recurrent_state_cls
            c.recurrent_layers = layers
            c.free_list = deque(range(slots))
            c.values = {}
            return c

        self.cache = make_cache(self.model, {})
        self.ranks.caches[id(self.cache)] = self.cache
        layers = {}
        for m in self.drafter.modules:
            ls = swa['SWALayerState'](m, slots, 16, 0)
            ls.alloc('cpu')
            layers[(m.layer_idx, 0)] = ls
        self.draft_cache = make_cache(self.drafter, layers)
        self.draft_cache.get_all_tensors = lambda: []

        class Job(e.job['Job']):
            def __init__(self, ids, max_new_tokens, identifier=None):
                ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
                self.sequences = [e.pt['Sequence'](ids, ids)]
                self.max_new_tokens, self.max_rq_tokens = max_new_tokens, None
                self.orig_max_rq_tokens = None
                self.banned_strings, self.prefix_token, self.filters, self.embeddings = [], None, [], []
                self.is_requeued, self.return_top_tokens, self.identifier = False, 0, identifier
                self.recurrent_state = None
                self.last_recurrent_checkpoint_pos = None
                self.cached_pages = self.cached_tokens = self.total_pages = self.non_sequential_pages = 0
                self.new_tokens = 0
                self.time_first_token = self.time_first_prefill = None
                self.checkpoint, self.checkpoint_rewound = None, False
                self.accepted_draft_tokens = self.rejected_draft_tokens = 0
                self.filter_futures, self.logit_masks, self.filters_suspended = [], [], False
                self.max_skips = 4
                self.text = []

            def prepare_logit_mask(self):
                pass

            def prepare_sampling_past_ids(self):
                pass

            def receive_logits(self, logits):
                return torch.argmax(logits, dim=-1), None, None, None

            def receive_sample(self, logits, next_token, next_k_tokens, next_k_probs, next_prob, results, *a):
                # job.py receive_sample: token accept and completed-page hashing, unchanged (no banned strings)
                next_token = next_token.cpu()
                self.new_tokens += 1
                for seq in self.sequences:
                    seq.sequence_ids.append(next_token)
                    page_before = seq.kv_position // PAGE
                    seq.kv_position += 1
                    page_after = seq.kv_position // PAGE
                    if page_after > page_before:
                        self.hash_completed_page(seq, page_before)
                self.text.append(int(next_token.item()))
                eos = self.new_tokens >= self.max_new_tokens
                results.append(dict(job=self, stage='streaming', eos=eos, serial=self.serial_number,
                                    identifier=self.identifier))
                return eos, next_token, False

        class Generator(e.gen['Generator']):
            pass

        self.Job, self.Generator = Job, Generator

    def next_token(self, value, token):
        """The target's greedy token after consuming `token` into state `value`: a function of the whole history
        (hash mode), or of the last token only (periodic mode: the output cycles, so hybrid drafts fire)."""
        return 3 + (5 * int(token)) % 97 if self.periodic else greedy(value)

    # -- observations ----------------------------------------------------------------------------------------------

    def window_exact(self, st):
        """The rows the next draft step attends to hold exactly the positions before st.position."""
        fill = st.position - st.window_beg
        lo = max(0, fill - WINDOW)
        if fill <= 0:
            return True
        want = enc(range(st.window_beg + lo, st.position)).view(-1, 1, 2)
        for ls in self.draft_cache.recurrent_layers.values():
            for t in (ls.k_state, ls.v_state):
                if not torch.equal(t[st.slot, lo:fill], want):
                    return False
        return True

    def expected_value(self, ids):
        value = 0
        for t in ids:
            value = step(value, t)
        return value

    def rank_handles_in_cache(self):
        return {s['target']['tp_handle'] for s in self.g.recurrent_cache.values()}

    def acceptance(self, serial):
        rows = [exact for r in self.rounds for s, exact in r if s == serial]
        return sum(rows), len(rows)

    # -- set-up -------------------------------------------------------------------------------------------------

    @contextmanager
    def installed(self, glm_dir=None):
        """The real service hooks (glm_native_concurrency.install_native_concurrency) on the compiled classes."""
        modules = {n: ModuleType(n) for n in ('exllamav3', 'exllamav3.generator', 'exllamav3.generator.generator',
                                              'exllamav3.generator.job')}
        modules['exllamav3'].__file__ = str(ENGINE / 'exllamav3/__init__.py')
        modules['exllamav3.generator.generator'].Generator = self.Generator
        modules['exllamav3.generator.job'].Job = self.Job
        with patch.dict(sys.modules, modules):
            import glm_native_concurrency
            glm_native_concurrency.install_native_concurrency(self.slots)
            yield glm_native_concurrency

    def generator(self, *, b2=True, shared_tokens=0, fair=(4, 512), max_chunk=2048, recurrent_bytes=5 * 1024**3,
                  allow_embeddings=False, drop_embeddings=False):
        e = self.e
        g = self.Generator.__new__(self.Generator)
        g.__dict__.update(
            model=self.model, cache=self.cache, draft_model=self.drafter, draft_cache=self.draft_cache,
            max_batch_size=self.slots, max_chunk_size=max_chunk, num_draft_tokens=K,
            max_draft_tokens=max(K, self.hybrid), hybrid_draft_tokens=self.hybrid, hybrid_rounds=0,
            active_jobs=[], pending_jobs=[], job_serial=0, enable_defrag=True, cache_owner_serial=1,
            dflash_draft=True, mtp_draft=False, ngram_match_min=0, ngram_corpus=None, visualizer=None,
            disk_page_cache=None, cpu_page_cache=None, fair_gen_rounds=fair[0], fair_chunk_size=fair[1],
            draft_calibrator=None, _draft_conf_round=None, record_draft_stats=False,
            target_recurrent=True, draft_recurrent=True,
            recurrent_checkpoint_interval=2048, recurrent_checkpoint_interval_pp=32768,
            max_total_tokens=self.pages * PAGE, padded_vocab_size=VOCAB, recurrent_cache_size=recurrent_bytes,
            filter_pool=NS(shutdown=lambda wait=False: None), sample_pinned=None,
            draft_input_ids_pinned=torch.empty((self.slots, 1), dtype=torch.long),
            draft_ids_pinned=torch.empty((self.slots, K), dtype=torch.long))
        g._staging = lambda name, rows, width=None, dtype=torch.int32: torch.zeros(
            (rows, width) if width is not None else (rows,), dtype=dtype)
        g.pagetable = e.pt['PageTable'](g, self.cache)
        g.recurrent_cache = e.rec['RecurrentCache'](self.model, recurrent_bytes, self.drafter)
        g.recurrent_cache.pagetable = g.pagetable
        self.cache.reset_states()
        self.draft_cache.reset_states()
        self.g = g
        import glm_native_concurrency
        manager = glm_native_concurrency.bind_native_concurrency(g, self.slots, allow_embeddings=allow_embeddings,
                                                                 drop_embeddings=drop_embeddings)
        coordinator = None
        if b2:
            import glm_b2_native
            coordinator = glm_b2_native.bind_b2_retention(g)
            if shared_tokens:
                coordinator.configure_shared_chat(shared_tokens, 'template', 'tokenizer', 'renderer')
        return g, manager, coordinator

    def admit_shared(self, job, boundary):
        """What the API's prepared_chat + bind_ticket leave for one job (a ticket with the derived boundary)."""
        import glm_b2_shared_prefix as shared
        coordinator = self.g._glm_b2_retention
        tokens = job.sequences[0].input_ids.torch()
        coordinator.shared.tickets[job] = shared.Ticket(tokens, coordinator.shared.context, boundary,
                                                        shared._sha(shared._encoded(tokens[0].tolist())))

    def run(self, jobs, limit=20000, between=None):
        """Enqueue jobs (list of (job, start_round)) and iterate until every job finished."""
        pending = sorted(jobs, key=lambda x: x[1])
        rounds = 0
        while (pending or self.g.num_remaining_jobs()) and rounds < limit:
            while pending and pending[0][1] <= rounds:
                self.g.enqueue(pending.pop(0)[0])
            results = self.g.iterate()
            if between is not None:
                between(rounds, results)
            rounds += 1
        assert rounds < limit, 'scenario did not finish'
        return rounds


def prompt(n, label=0):
    """A deterministic prompt; label selects a different one."""
    return [(7 * i + 11 * label + 5) % 89 + 3 for i in range(n)]


def reference_text(scenario_factory, ids, max_new_tokens):
    """The output of a job run alone, on a fresh generator (no checkpoint to restore)."""
    s = scenario_factory()
    with s.installed():
        s.generator()
        job = s.Job(ids, max_new_tokens)
        s.run([(job, 0)])
        return job.text
