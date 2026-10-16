"""P16 stage 2: the service's native concurrency guard and B2 layer on a fake engine around upstream's real
PageTable / RecurrentCache / PairedState / SWAState (native_engine_fakes.py). No CUDA, no extension import (the hybrid
case uses the arm's native BC_SAM when P16_NATIVE is set).

Each scenario runs the real Generator.iterate loop with the real service hooks installed and checks:
- lossless output against the same prompt run alone on a fresh generator;
- the restored pair: target state (output) and the draft window (every draft round attends to exact rows);
- slot, page and rank-handle custody (no leak, no aliasing) and the /health counters (DESIGN.md B6);
- the terminal latch on stash faults (U10) and on allocation custody loss, with non-latched controls.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'glm'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_engine_fakes as nf
import glm_b2_native as b2
import glm_native_concurrency as nc

ENGINE = nf.Engine()
NATIVE = os.environ.get('P16_NATIVE')


def native_sam():
    if not NATIVE:
        return None
    if NATIVE not in sys.path:
        sys.path.insert(0, NATIVE)
    try:
        import exllamav3_ext
    except ImportError:
        return None
    return getattr(exllamav3_ext, 'BC_SAM', None)


SAM = native_sam()


def scenario(**kw):
    return nf.Scenario(ENGINE, **kw)


REFERENCES = {}


def reference(ids, n, **kw):
    key = (tuple(ids), n, tuple(sorted(kw.items())))
    if key not in REFERENCES:
        REFERENCES[key] = nf.reference_text(lambda: scenario(**kw), ids, n)
    return REFERENCES[key]


class Base(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def custody(self, s, m):
        """No job holds a slot after the scenario; every rank handle is referenced by exactly the native cache."""
        g = s.g
        self.assertEqual(g.num_remaining_jobs(), 0)
        self.assertEqual(sorted(g.cache.free_list), list(range(s.slots)))
        self.assertEqual(sorted(g.draft_cache.free_list), list(range(s.slots)))
        self.assertEqual(m.stats()['active_ring_leases'], 0)
        self.assertEqual(set(s.ranks.recurrent_cache), s.rank_handles_in_cache())
        self.assertFalse(g.pagetable.referenced_pages)


class Resume(Base):
    """Probe S4 on CPU: A crosses decode checkpoints at 4096 and 6144; B = A's prefix + 596 of A's tokens + a
    suffix resumes at 4096 with the exact pair."""

    def run_s4(self, **kw):
        s = scenario(**kw)
        with s.installed():
            g, m, c = s.generator()
            a = s.Job(nf.prompt(3800), 2400)
            s.run([(a, 0)])
            b_ids = (nf.prompt(3800) + a.text[:596] + nf.prompt(100, label=5))
            b = s.Job(b_ids, 300)
            start = len(s.rounds)
            s.run([(b, 0)])
            return s, g, m, c, a, b, b_ids, start

    def test_resume_restores_target_and_draft_window_exactly(self):
        s, g, m, c, a, b, b_ids, start = self.run_s4()
        self.assertEqual(a.text, reference(nf.prompt(3800), 2400))
        self.assertEqual(b.text, reference(b_ids, 300))                      # target restored exactly (lossless)
        self.assertEqual(b.cached_pages * nf.PAGE, 4096)                      # restore-hit at the 4096 checkpoint
        rounds = [exact for r in s.rounds[start:] for serial, exact in r if serial == b.serial_number]
        self.assertTrue(rounds and all(rounds))                                # every draft window exact after resume
        st = c.stats()
        self.assertEqual((st['hits'], st['cold_misses'], st['restored_tokens']), (1, 1, 4096))
        self.assertEqual((st['plans'], st['allocation_cold_fallbacks']), (1, 0))
        self.assertGreaterEqual(st['captures'], 4)                             # 2048, 3584, 4096, 6144 at least
        self.assertEqual(st['completed_checkpoints'], len(g.recurrent_cache))
        self.assertEqual(st['paired_host_bytes'], g.recurrent_cache.current_size)
        self.assertGreater(st['kv_only_pages'], 0)                             # B's cached pages past 4096 replay
        self.assertFalse(st['terminal'])
        self.assertEqual(m.stats()['reused_slots'], 1)
        self.custody(s, m)

    def test_draft_window_of_every_decode_checkpoint_is_exact(self):
        """Each stored pair's draft window equals its live source (the U1 property, through the real put)."""
        s, g, m, c, *_ = self.run_s4()
        layers = list(s.draft_cache.recurrent_layers.values())
        for key, stash in g.recurrent_cache.items():
            draft = stash['draft']
            pos, beg = draft['position'], draft['window_beg']
            want = nf.enc(range(pos - nf.WINDOW, pos)).view(-1, 1, 2)
            for i, layer in enumerate(layers):
                k, v = draft[(i, 0)]
                self.assertTrue(torch.equal(k, want) and torch.equal(v, want), (pos, beg))
            self.assertEqual(stash['target']['position'], pos)

    def test_cold_reuse_off_matches_and_restores_nothing(self):
        """B_cold: the same B prompt on a fresh generator runs cold with the same output."""
        s, g, m, c, a, b, b_ids, _ = self.run_s4()
        s2 = scenario()
        with s2.installed():
            g2, m2, c2 = s2.generator()
            b2job = s2.Job(b_ids, 300)
            s2.run([(b2job, 0)])
        self.assertEqual(b2job.cached_pages, 0)
        self.assertEqual(b2job.text, b.text)
        self.assertEqual(c2.stats()['hits'], 0)


class TwoJobs(Base):
    def test_four_shared_tickets_transition_to_owned_jobs_without_extra_admission(self):
        import glm_b2_shared_prefix as shared
        s=scenario(slots=4)
        with s.installed():
            g,m,c=s.generator(shared_tokens=512)
            jobs=[s.Job(nf.prompt(600,i+1),500) for i in range(5)]
            def bind(j):
                tokens=j.sequences[0].input_ids.torch()
                c.shared.prepared=[shared.Ticket(tokens,c.shared.context,256,
                    shared._sha(shared._encoded(tokens[0].tolist()))),None]
                try:c.shared.bind_ticket(j)
                finally:c.shared.prepared=None
            for j in jobs[:4]:bind(j)
            with self.assertRaisesRegex(ValueError,'capacity'):bind(jobs[4])
            self.assertEqual(len(c.shared.tickets),4)
            for j in jobs[:4]:g.enqueue(j)
            self.assertEqual((len(c.jobs),len(c.shared.tickets)),(4,0))
            s.run([])
            for i,j in enumerate(jobs[:4]):self.assertEqual(j.text,reference(nf.prompt(600,i+1),500))
            self.assertFalse(c.jobs or c.shared.tickets);self.custody(s,m)

    def test_four_distinct_jobs_are_lossless_with_independent_state_and_reuse(self):
        s=scenario(slots=4)
        observed=[]
        with s.installed():
            g,m,c=s.generator()
            jobs=[s.Job(nf.prompt(600,i+1),500) for i in range(4)]
            def observe(_,results):
                held=[(j.recurrent_state.target.slot,j.recurrent_state.draft.slot)
                      for j in g.active_jobs if j.recurrent_state is not None]
                if len(held)==4:observed.append(held)
            s.run([(j,0) for j in jobs],between=observe)
            for i,j in enumerate(jobs):self.assertEqual(j.text,reference(nf.prompt(600,i+1),500))
            self.assertTrue(observed)
            for held in observed:
                self.assertEqual(len({t for t,d in held}),4)
                self.assertEqual(len({d for t,d in held}),4)
            self.assertEqual((m.peak_target_batch,m.peak_draft_batch,m.peak_leases),(4,4,4))
            self.custody(s,m)
            reused=s.Job(nf.prompt(600,7),500);s.run([(reused,0)])
            self.assertEqual(reused.text,reference(nf.prompt(600,7),500))
            self.assertGreater(m.reused_slots,0);self.custody(s,m)

    def test_four_slot_retention_rejects_fifth_admission_and_capacity_drift(self):
        s=scenario(slots=4)
        with s.installed():
            g,m,c=s.generator()
            for i in range(4):g.enqueue(s.Job(nf.prompt(300,i),8))
            with self.assertRaisesRegex(ValueError,'four distinct jobs'):
                g.enqueue(s.Job(nf.prompt(300,8),8))
            self.assertEqual(len(c.jobs),4)
            m.slots=2
            with self.assertRaises(b2.StopSharedGeneration):c.check_owner()

    def test_two_concurrent_distinct_jobs_are_lossless_on_distinct_slots(self):
        s = scenario()
        slots = []
        with s.installed():
            g, m, c = s.generator()
            a, b = s.Job(nf.prompt(3000, 1), 900), s.Job(nf.prompt(2500, 2), 900)

            def observe(_, results):
                held = [(j.recurrent_state.target.slot, j.recurrent_state.draft.slot)
                        for j in g.active_jobs if j.recurrent_state is not None]
                slots.append(held)
            s.run([(a, 0), (b, 3)], between=observe)
        self.assertEqual(a.text, reference(nf.prompt(3000, 1), 900))
        self.assertEqual(b.text, reference(nf.prompt(2500, 2), 900))
        two = [h for h in slots if len(h) == 2]
        self.assertTrue(two)
        for held in two:
            self.assertEqual(len({t for t, _ in held}), 2)
            self.assertEqual(len({d for _, d in held}), 2)
        st = m.stats()
        self.assertEqual((st['peak_target_batch'], st['peak_draft_batch'], st['peak_ring_leases']), (2, 2, 2))
        self.assertGreater(st['batched_verifier_rounds'], 0)
        self.assertGreater(st['batched_draft_rounds'], 0)
        self.assertGreater(st['mixed_prefill_steps'], 0)                       # fair 4/512 while A generates
        self.assertEqual(st['decode_table_rebinds'], 0)
        self.assertTrue(all(exact for r in s.rounds for _, exact in r))
        self.custody(s, m)

    def test_identical_pair_counts_rebind_and_reuses_the_existing_pair(self):
        """P13 scenario: identical prompts decode together; the later job's completed page is deduped onto the
        earlier job's (observed rebind, no draft block table involved) and its put under the same key is a
        native no-op (shared_cookie_reuses)."""
        s = scenario()
        with s.installed():
            g, m, c = s.generator()
            ids = nf.prompt(27)
            a, b = s.Job(ids, 2400), s.Job(ids, 2400)
            s.run([(a, 0), (b, 14)])
        ref = reference(ids, 2400)
        self.assertEqual((a.text, b.text), (ref, ref))
        self.assertGreaterEqual(m.stats()['decode_table_rebinds'], 1)
        self.assertGreaterEqual(c.stats()['shared_cookie_reuses'], 1)
        self.custody(s, m)


class SharedPrefix(Base):
    def test_shared_boundary_capture_then_hit(self):
        s = scenario()
        system = nf.prompt(2100, 9)
        with s.installed():
            g, m, c = s.generator(shared_tokens=65536)
            a = s.Job(system + nf.prompt(700, 1), 64)
            s.admit_shared(a, 2048)
            s.run([(a, 0)])
            self.assertEqual(c.stats()['shared_captures'], 1)
            self.assertEqual(c.stats()['shared_completed_checkpoints'], 1)
            b_ids = system + nf.prompt(900, 2)
            b = s.Job(b_ids, 64)
            s.admit_shared(b, 2048)
            s.run([(b, 0)])
        st = c.stats()
        self.assertEqual(b.cached_pages * nf.PAGE, 2048)
        self.assertEqual((st['shared_hits'], st['shared_restored_tokens']), (1, 2048))
        self.assertEqual(b.text, reference(b_ids, 64))
        self.assertTrue(all(exact for r in s.rounds for _, exact in r))
        self.custody(s, m)

    def test_adapted_boundary_with_a_shorter_common_prefix(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator(shared_tokens=65536)
            a = s.Job(nf.prompt(2600, 3) + nf.prompt(500, 1), 32)
            s.admit_shared(a, 2560)
            s.run([(a, 0)])
            b_ids = nf.prompt(1100, 3) + nf.prompt(2000, 4)          # shares 1100 tokens with A: 1024 page-aligned
            b = s.Job(b_ids, 32)
            s.admit_shared(b, 2560)
            s.run([(b, 0)])
        st = c.stats()
        self.assertEqual(st['shared_adapted_boundaries'], 1)
        self.assertEqual(st['shared_captures'], 2)                    # A at 2560, B at its adapted 1024
        self.assertEqual(b.text, reference(b_ids, 32))
        self.custody(s, m)


class Eviction(Base):
    def test_lru_and_stranded_eviction_release_both_sides(self):
        draft_bytes = nf.DRAFT_LAYERS * 2 * nf.WINDOW * 2 * 2
        budget = 3 * (nf.TARGET_CHECKPOINT_BYTES + draft_bytes) + 1000
        s = scenario(pages=40)
        with s.installed():
            g, m, c = s.generator(recurrent_bytes=budget)
            a = s.Job(nf.prompt(3000, 1), 3000)
            s.run([(a, 0)])
            b = s.Job(nf.prompt(4000, 2), 2000)                       # evicts A's pages: A's pairs strand
            s.run([(b, 0)])
        st = c.stats()
        self.assertGreater(st['paired_evictions'], 0)
        self.assertEqual(st['native_releases'], st['paired_evictions'] + st['pruned'])
        self.assertLessEqual(g.recurrent_cache.current_size, budget)
        self.assertLessEqual(len(g.recurrent_cache), 3)
        self.assertEqual(b.text, reference(nf.prompt(4000, 2), 2000, pages=40))
        self.custody(s, m)


class TerminalLatch(Base):
    """U10: a fault inside the native paired put can strand the target's rank-side copy (its handle is dispatched
    before the draft window is copied). Under B2 it latches the shared generator; a non-B2 control shows the
    stranded handle the latch exists for."""

    def fault(self, s, b2_bound):
        """Fail the draft-window copy of the second paired put: the prompt's last-page checkpoint (2816), which
        native Job.prefill takes inside the per-job containment of iterate()."""
        calls = {'n': 0, 'failed': 0}
        original = s.e.swa['SWALayerState'].stash

        def failing(layer, slot, fill):
            calls['n'] += 1
            if calls['n'] == 3:            # two sliding layers per put: the first layer of put #2
                calls['failed'] += 1
                raise MemoryError('injected draft stash fault')
            return original(layer, slot, fill)
        results = None
        with s.installed():
            g, m, c = s.generator(b2=b2_bound)
            a = s.Job(nf.prompt(3000, 1), 600)
            g.enqueue(a)
            error = None
            with patch.object(s.e.swa['SWALayerState'], 'stash', failing):
                for _ in range(400):
                    try:
                        results = g.iterate()
                    except BaseException as caught:
                        error = caught
                        break
                    if any(r.get('stage') == 'error' for r in results) or not g.num_remaining_jobs():
                        break
            return g, m, c, a, error, results, calls

    def test_draft_stash_fault_latches_b2(self):
        s = scenario()
        g, m, c, a, error, _, calls = self.fault(s, True)
        self.assertEqual(calls['failed'], 1)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn('lost shared native custody', str(error))
        self.assertTrue(c.stats()['terminal'])
        self.assertIsInstance(c.failure, MemoryError)
        stranded = set(s.ranks.recurrent_cache) - s.rank_handles_in_cache()
        self.assertEqual(len(stranded), 1)                         # the target copy the failed put dispatched
        with self.assertRaises(b2.StopSharedGeneration):
            c.prepare(s.Job(nf.prompt(300), 8))

    def test_control_without_b2_reaps_one_job_and_strands_the_rank_copy(self):
        s = scenario()
        g, m, c, a, error, results, calls = self.fault(s, False)
        self.assertIsNone(error)                                   # native reap_failed_job contained it
        self.assertTrue(any(r.get('stage') == 'error' and r['job'] is a for r in results))
        stranded = set(s.ranks.recurrent_cache) - s.rank_handles_in_cache()
        self.assertEqual(len(stranded), 1)                         # leaked, and the generator keeps going

    def test_target_rank_stash_fault_latches_b2(self):
        s = scenario()
        s.ranks.fail = lambda name, args: name == 'rank_stash'
        with s.installed():
            g, m, c = s.generator()
            a = s.Job(nf.prompt(3000, 1), 64)
            g.enqueue(a)
            with self.assertRaisesRegex(RuntimeError, 'lost shared native custody'):
                for _ in range(400):
                    g.iterate()
        self.assertTrue(c.stats()['terminal'])

    def test_unsettled_pair_is_refused_before_put(self):
        """A draft side left with a staged pass at a boundary is refused before the native put."""
        s = scenario()
        update = s.drafter.update_kv_from_target

        def staged(target_hidden, cache, params, lengths=None):
            update(target_hidden, cache, params, lengths)
            for st in params['recurrent_states']:
                if st.position == 2048:
                    st.last_history = 1
        s.drafter.update_kv_from_target = staged
        with s.installed():
            g, m, c = s.generator()
            g.enqueue(s.Job(nf.prompt(3000, 1), 64))
            with self.assertRaisesRegex(RuntimeError, 'lost shared native custody'):
                for _ in range(400):
                    g.iterate()
        self.assertIsInstance(c.failure, b2.UnsafeCleanupError)
        self.assertEqual(len(g.recurrent_cache), 0)


class AllocationCustody(Base):
    def test_pair_creation_fault_returns_target_slot_and_next_job_runs(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator()
            original = s.draft_cache.get_new_state
            def failing():
                raise AssertionError('injected: no draft slot')
            s.draft_cache.get_new_state = failing
            custody = c._custody()
            g.enqueue(s.Job(nf.prompt(600), 8))
            results = g.iterate()
            self.assertTrue(any(r.get('stage')=='error' and 'no draft slot' in str(r.get('error')) for r in results))
            self.assertIsNone(c.failure)
            self.assertEqual(c._custody(), custody)
            s.draft_cache.get_new_state = original
            s.run([(s.Job(nf.prompt(600, 2), 8), 0)])
            self.assertIsNone(c.failure)

    def test_restored_pair_that_disagrees_with_its_prefix_is_terminal(self):
        """A restore whose draft side does not sit at the resumed position (a broken engine contract) latches."""
        s = scenario()
        with s.installed():
            g, m, c = s.generator()
            s.run([(s.Job(nf.prompt(2600), 16), 0)])
            restore = s.draft_cache.new_from_stashed
            def shifted(stashed, position):
                state = restore(stashed, position)
                state.position += 256
                return state
            s.draft_cache.new_from_stashed = shifted
            g.enqueue(s.Job(nf.prompt(2600) + nf.prompt(300, 4), 8))
            with self.assertRaisesRegex(RuntimeError, 'lost shared native custody'):
                g.iterate()
        self.assertRegex(repr(c.failure), 'disagrees with its cached prefix')

    def test_contained_allocation_fault_reaps_only_that_job(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator()
            pt = g.pagetable
            original = type(pt).allocate_pages
            def failing(self, *a, **k):
                raise ValueError('injected page allocation refusal')
            type(pt).allocate_pages = failing
            try:
                job = s.Job(nf.prompt(600), 8)
                g.enqueue(job)
                results = g.iterate()
            finally:
                type(pt).allocate_pages = original
            self.assertTrue(any(r.get('stage') == 'error' and r['job'] is job for r in results))
            self.assertFalse(c.stats()['terminal'])
            self.assertEqual(c.stats()['active_job_records'], 0)
            ok = s.Job(nf.prompt(600, 3), 16)
            s.run([(ok, 0)])
        self.assertEqual(ok.text, reference(nf.prompt(600, 3), 16))
        self.custody(s, m)

    def test_leaked_draft_slot_is_detected_at_the_next_decode(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator()
            g.enqueue(s.Job(nf.prompt(300), 64))
            g.iterate()
            s.draft_cache.free_list.popleft()                         # a slot no job holds
            with self.assertRaisesRegex(RuntimeError, 'lost shared native custody'):
                for _ in range(10):
                    g.iterate()
        self.assertIn('slot custody', str(c.failure))


class Admission(Base):
    def test_unsupported_jobs_refused_at_enqueue_and_third_job_refused(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator()
            for change in (dict(banned_strings=['x']), dict(filters=[object()]), dict(prefix_token=1),
                           dict(is_requeued=True), dict(orig_max_rq_tokens=10), dict(embeddings=[object()])):
                job = s.Job(nf.prompt(300), 8)
                job.__dict__.update(change)
                if 'banned_strings' in change:
                    job._check_banned_strings = lambda strings: None
                with self.subTest(change=list(change)), self.assertRaises((ValueError, AssertionError)):
                    g.enqueue(job)
            self.assertEqual(c.stats()['active_job_records'], 0)
            g.enqueue(s.Job(nf.prompt(300), 8))
            g.enqueue(s.Job(nf.prompt(300, 1), 8))
            with self.assertRaisesRegex(ValueError, 'two distinct jobs'):
                g.enqueue(s.Job(nf.prompt(300, 2), 8))

    def test_swa_full_and_unbound_shapes_refused(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator(b2=False)
            g._glm_native_concurrency = None
            g.draft_recurrent = False                                   # what -swa_full gives
            with self.assertRaises(ValueError):
                nc.NativeConcurrency(g, 2)
            g.draft_recurrent = True
            with self.assertRaises(ValueError):
                nc.NativeConcurrency(g, 3)                              # slots != -ambs
        with self.assertRaises(ValueError):
            b2.bind_b2_retention(g)


class HealthShape(Base):
    PAIRED_FIELDS = ("plans", "cold_misses", "hits", "restored_tokens", "allocation_cold_fallbacks", "captures",
                     "native_releases", "pruned", "snapshot_budget_skips", "cancellations", "paired_evictions",
                     "completed_checkpoints", "paired_host_bytes", "reserved_snapshot_bytes", "active_job_records",
                     "terminal")

    def test_prompt_cache_and_concurrent_ring_keys(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator(shared_tokens=65536)
            s.run([(s.Job(nf.prompt(600), 16), 0)])
            pc, ring = c.stats(), m.stats()
        missing = [k for k in self.PAIRED_FIELDS if k not in pc and k != 'cancellations']
        self.assertEqual(missing, [])
        for key in ('mode', 'target_cpu_tier', 'paired_budget_bytes', 'max_checkpoints', 'shared_prefix_enabled',
                    'shared_pending_tickets', 'shared_completed_checkpoints'):
            self.assertIn(key, pc)
        self.assertEqual((pc['mode'], pc['snapshot_budget_skips'], pc['reserved_snapshot_bytes'], pc['max_checkpoints']),
                         ('b2_native_paired', 0, 0, None))
        self.assertEqual(pc['paired_budget_bytes'], 5 * 1024**3)
        self.assertEqual(ring['mode'], 'native_swa')
        for key in ('max_sequences', 'active_ring_leases', 'peak_ring_leases', 'peak_draft_batch',
                    'batched_draft_rounds', 'peak_target_batch', 'batched_verifier_rounds', 'mixed_prefill_steps',
                    'reused_slots', 'decode_table_rebinds', 'paired_retention'):
            self.assertIn(key, ring)
        self.assertEqual((ring['draft_state_rows'], ring['draft_window']), (2560, 2047))
        self.assertEqual(ring['draft_state_bytes'], nf.DRAFT_LAYERS * 2 * 2 * 2560 * 1 * 2 * 2)


class Close(Base):
    def test_close_releases_every_pair_and_retires(self):
        s = scenario()
        with s.installed():
            g, m, c = s.generator()
            s.run([(s.Job(nf.prompt(3000), 64), 0)])
            self.assertGreater(len(g.recurrent_cache), 0)
            g.close()
        self.assertEqual(s.ranks.recurrent_cache, {})
        self.assertTrue(c.closed and m.closed)
        with self.assertRaises(b2.StopSharedGeneration):
            c.check_owner()


@unittest.skipIf(SAM is None, 'native BC_SAM unavailable (set P16_NATIVE)')
class Hybrid(Base):
    """Upstream hybrid drafting (H=15, 16-row verifies) writes up to 16 accepted rows per round into the window rings
    (the n = 16 plan_state_rows path, including the page shift)."""

    def run_jobs(self, hybrid, resume=True):
        s = scenario(hybrid=hybrid, sam=SAM, periodic=True)
        with s.installed():
            g, m, c = s.generator()
            a = s.Job(nf.prompt(3800, 1), 2400)
            s.run([(a, 0)])
            b = s.Job(nf.prompt(3800, 1) + a.text[:596] + nf.prompt(100, 2), 600)
            s.run([(b, 0)])
        return s, g, m, c, a, b

    def test_hybrid_rounds_lossless_and_windows_exact(self):
        s0, *_, a0, b0 = self.run_jobs(0)
        s, g, m, c, a, b = self.run_jobs(15)
        self.assertEqual((a.text, b.text), (a0.text, b0.text))
        self.assertGreater(g.hybrid_rounds, 0)
        self.assertIn(16, s.widths)
        self.assertLess(len(s.widths), len(s0.widths))
        self.assertEqual(b.cached_pages * nf.PAGE, 4096)
        self.assertTrue(all(exact for r in s.rounds for _, exact in r))
        self.custody(s, m)


if __name__ == '__main__':
    unittest.main(verbosity=2)
