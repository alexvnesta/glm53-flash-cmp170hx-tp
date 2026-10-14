"""P16 stage 2: the native concurrency guard (glm_native_concurrency) contracts, CPU only.

Replaces test_ring.py / test_install_hooks.py (the stage-1 ConcurrentRing partitions, block-table remap, leases,
stream proof and cache factory are deleted). The guard keeps: the start-up shape checks, the owner proof, the
persistent step/thread owner, the one admission check, slot-custody and observation counters, the P14c drop, the
engine source pins and the instance-gated hook installation.
"""
import os
import sys
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'glm'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_native_concurrency as nc
import native_engine_fakes as nf

ENGINE = Path(os.environ.get('EXLLAMAV3_ENGINE_ROOT', ROOT.parent / 'engine'))


class Layer:
    """SWALayerState-shaped draft window ring geometry (2560 rows, window 2047, two slots)."""
    def __init__(self, slots=2):
        self.max_batch_size = slots
        self.module = NS(kv_state_size=2560, sliding_window=2047)

    def storage_size(self):
        return 2 * self.max_batch_size * 2560 * 8 * 128 * 2


def generator(slots=2):
    """A generator double the real NativeConcurrency accepts (TP target, recurrent DFlash2 drafter)."""
    target = NS(num_slots=2, owner_serial=1, max_num_tokens=32768, free_list=deque([0, 1]))
    draft_model = NS()
    draft = NS(model=draft_model, num_slots=slots, recurrent_layers={(i, 0): Layer(slots) for i in range(5)},
               free_list=deque(range(slots)))
    g = NS(dflash_draft=True, draft_recurrent=True, model=NS(loaded_tp=True, caps={}), cache=target,
           draft_model=draft_model, draft_cache=draft, max_batch_size=2, recurrent_cache=NS(), enable_defrag=False,
           cache_owner_serial=1, max_chunk_size=2048, active_jobs=[], pending_jobs=[], mtp_draft=False)
    g.num_remaining_jobs = lambda: len(g.active_jobs) + len(g.pending_jobs)
    g.pagetable = NS(generator=g, cache=target, cpu_tier=None, all_pages=[NS(page_index=i) for i in range(128)])
    return g


class Job:
    def __init__(self, g, draft_slot=0, target_slot=0, prefilled=True):
        self.generator, self.pagetable = g, g.pagetable
        self.sequences = [NS(block_index_tensor=torch.zeros((1, 4), dtype=torch.int32), max_cached_pages=None)]
        self.is_requeued = False; self.embeddings = []; self.filters = []; self.banned_strings = []
        self.prefix_token = None; self.orig_max_rq_tokens = None; self.prefilled = prefilled
        self.recurrent_state = NS(target=NS(slot=target_slot), draft=NS(slot=draft_slot, cache=g.draft_cache))

    def is_prefill_done(self):
        return self.prefilled


class Contracts(unittest.TestCase):
    def setUp(self):
        self.g = generator()
        self.m = nc.NativeConcurrency(self.g, 2)

    def test_start_up_shape_refusals(self):
        changes = [('draft_recurrent', False), ('dflash_draft', False), ('max_batch_size', 3),
                   ('enable_defrag', True), ('recurrent_cache', None)]
        for name, value in changes:
            g = generator(); setattr(g, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                nc.NativeConcurrency(g, 2)
        for change in (lambda g: setattr(g.model, 'loaded_tp', False), lambda g: setattr(g.draft_cache, 'num_slots', 3),
                       lambda g: g.draft_cache.recurrent_layers.clear(),
                       lambda g: setattr(g.draft_cache.recurrent_layers[(0, 0)], 'max_batch_size', 3),
                       lambda g: setattr(g.draft_cache.recurrent_layers[(1, 0)].module, 'kv_state_size', 2304),
                       lambda g: setattr(g.pagetable, 'cpu_tier', object()),
                       lambda g: g.active_jobs.append(object()),
                       lambda g: setattr(g.model, '_glm_tp_active_host_coordinator', object())):
            g = generator(); change(g)
            with self.subTest(change=change), self.assertRaises(ValueError):
                nc.NativeConcurrency(g, 2)
        with patch.dict(os.environ, {'GLM53_TP_ACTIVE_HOST': '1'}), self.assertRaises(ValueError):
            nc.NativeConcurrency(generator(), 2)
        for slots in (1, 9, True):
            with self.assertRaises(ValueError):
                nc.NativeConcurrency(generator(), slots)

    def test_geometry_is_read_from_the_native_rings(self):
        st = self.m.stats()
        self.assertEqual((st['mode'], st['draft_state_rows'], st['draft_window']), ('native_swa', 2560, 2047))
        self.assertEqual(st['draft_state_bytes'], 5 * Layer().storage_size())
        self.assertEqual((st['active_ring_leases'], st['max_sequences'], st['paired_retention']), (0, 2, False))
        self.assertNotIn('embeddings_dropped', st)

    def test_owner_and_geometry_changes_refused(self):
        changes = [lambda g: setattr(g, 'model', NS(loaded_tp=True)),
                   lambda g: setattr(g.cache, 'owner_serial', 2),
                   lambda g: setattr(g, 'cache_owner_serial', 2),
                   lambda g: setattr(g.pagetable, 'all_pages', list(g.pagetable.all_pages)),
                   lambda g: g.pagetable.all_pages.__setitem__(3, NS(page_index=3)),
                   lambda g: setattr(g, 'enable_defrag', True),
                   lambda g: setattr(g, 'draft_model', NS()),
                   lambda g: setattr(g, 'draft_recurrent', False),
                   lambda g: setattr(g.draft_cache, 'recurrent_layers', dict(g.draft_cache.recurrent_layers)),
                   lambda g: setattr(g.draft_cache, 'num_slots', 3),
                   lambda g: setattr(g, 'recurrent_cache', NS()),
                   lambda g: setattr(g.model, '_glm_tp_active_host_coordinator', object()),
                   lambda g: setattr(g, '_glm_tp_active_host_coordinator', object()),
                   lambda g: setattr(g.pagetable, 'cpu_tier', object()),
                   lambda g: setattr(g, '_glm_b2_retention', object())]
        for change in changes:
            self.setUp(); change(self.g)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.m.check_owner()

    def test_transfer_outside_the_step_or_thread_refused(self):
        with self.assertRaises(RuntimeError):
            self.m.check_owner(transfer=True)
        with self.m.step():
            self.m.check_owner(transfer=True)
        errors = []
        def foreign():
            try:
                with self.m.step(): pass
            except RuntimeError as e: errors.append(e)
        t = threading.Thread(target=foreign); t.start(); t.join(); self.assertEqual(len(errors), 1)

    def test_nesting_refused_and_failure_restores_depth(self):
        with self.m.step():
            with self.assertRaises(RuntimeError):
                with self.m.step(): pass
        with self.assertRaises(ValueError):
            with self.m.step():
                with self.m.prefill(object()):
                    raise ValueError('prefill failure')
        self.assertEqual(self.m.depth, 0); self.assertIsNone(self.m.prefill_job)
        with self.m.step():
            with self.m.prefill(object()):
                with self.assertRaises(RuntimeError):
                    with self.m.prefill(object()): pass

    def test_admission_is_one_merged_check(self):
        self.m.prepare(Job(self.g))
        for change in (dict(is_requeued=True), dict(filters=[object()]), dict(banned_strings=['x']),
                       dict(prefix_token=1), dict(orig_max_rq_tokens=5), dict(embeddings=[object()]),
                       dict(generator=object()), dict(pagetable=object())):
            job = Job(self.g); job.__dict__.update(change)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.m.prepare(job)
        job = Job(self.g); job.sequences.append(job.sequences[0])
        with self.assertRaises(ValueError):
            self.m.prepare(job)
        vision = nc.NativeConcurrency(generator(), 2, allow_embeddings=True)
        job = Job(vision.generator); job.embeddings = [object()]
        vision.prepare(job)

    def test_slot_custody_and_reuse_observations(self):
        g, m = self.g, self.m
        a, b = Job(g, 0, 0), Job(g, 1, 1)
        for job in (a, b):
            g.cache.free_list.popleft(); g.draft_cache.free_list.popleft(); g.active_jobs.append(job)
        m.allocated(a); m.allocated(b)
        self.assertEqual((m.stats()['active_ring_leases'], m.stats()['peak_ring_leases']), (2, 2))
        g.active_jobs.remove(a); g.cache.free_list.append(0); g.draft_cache.free_list.append(0)
        c = Job(g, 0, 0); g.cache.free_list.popleft(); g.draft_cache.free_list.popleft(); g.active_jobs.append(c)
        m.allocated(c)
        self.assertEqual(m.stats()['reused_slots'], 1)
        for corrupt in ('alias', 'leak', 'foreign'):
            self.setUp(); g, m = self.g, self.m
            a, b = Job(g, 0, 0), Job(g, 1, 1)
            for job in (a, b):
                g.cache.free_list.popleft(); g.draft_cache.free_list.popleft(); g.active_jobs.append(job)
            if corrupt == 'alias': b.recurrent_state.draft.slot = 0
            if corrupt == 'leak': g.draft_cache.free_list.clear(); g.active_jobs.remove(b)
            if corrupt == 'foreign': b.recurrent_state.draft.cache = NS()
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(RuntimeError, 'slot custody'):
                m.check_slots()

    def test_observed_rebinds_and_batch_peaks(self):
        a, b = Job(self.g), Job(self.g, 1, 1)
        tables = self.m.snapshot_tables([a, b])
        b.sequences[0].block_index_tensor = torch.ones((1, 4), dtype=torch.int32)      # native dedupe rebuilt it
        self.m.observe_decode([a, b], tables, drafted=True)
        self.m.observe_decode([a], self.m.snapshot_tables([a]), drafted=False)
        st = self.m.stats()
        self.assertEqual(st['decode_table_rebinds'], 1)
        self.assertEqual((st['peak_target_batch'], st['batched_verifier_rounds']), (2, 1))
        self.assertEqual((st['peak_draft_batch'], st['batched_draft_rounds']), (2, 1))

    def test_p14c_drop_prefilled_embeddings(self):
        m = nc.NativeConcurrency(generator(), 2, allow_embeddings=True, drop_embeddings=True)
        ids = torch.tensor([[1, 2, nc.FIRST_MM_EMBEDDING_INDEX, 3, 4]])
        def job(kv):
            j = Job(m.generator); j.embeddings = [object()]
            j.sequences = [NS(kv_position=kv, sequence_ids=NS(torch_slice=lambda a, b: ids[:, a:b]))]
            return j
        fed, unfed = job(4), job(2)
        m.drop_prefilled_embeddings([fed, unfed])
        self.assertEqual((fed.embeddings, len(unfed.embeddings)), ([], 1))
        self.assertEqual(m.stats()['embeddings_dropped'], 1)
        with self.assertRaises(ValueError):
            nc.NativeConcurrency(generator(), 2, drop_embeddings=True)

    def test_bind_disables_defrag_and_failure_restores_it(self):
        g = generator(); g.enable_defrag = True
        m = nc.bind_native_concurrency(g, 2)
        self.assertFalse(g.enable_defrag); self.assertIs(g._glm_native_concurrency, m)
        with self.assertRaises(ValueError):
            nc.bind_native_concurrency(g, 2)
        g = generator(); g.enable_defrag = True; g.draft_recurrent = False       # -swa_full
        with self.assertRaises(ValueError):
            nc.bind_native_concurrency(g, 2)
        self.assertTrue(g.enable_defrag); self.assertFalse(hasattr(g, '_glm_native_concurrency'))

    def test_source_pins_exact_engine(self):
        nc.verify_engine_sources(ENGINE)
        with tempfile.TemporaryDirectory() as td, self.assertRaises(FileNotFoundError):
            nc.verify_engine_sources(td)
        self.assertEqual(len(nc.ENGINE_SOURCE_PINS), 29)
        for key in ('exllamav3/modules/sliding_attn.py', 'exllamav3/architecture/dflash.py',
                    'exllamav3/modules/arch_specific/dflash.py', 'exllamav3/cache/recurrent.py'):
            self.assertIn(key, nc.ENGINE_SOURCE_PINS)


class Installation(unittest.TestCase):
    def test_hooks_are_idempotent_and_capacity_conflicts_refuse(self):
        s = nf.Scenario(nf.Engine())
        with s.installed():
            hook = s.Job.prepare_for_queue
            nc.install_native_concurrency(2)
            self.assertIs(s.Job.prepare_for_queue, hook)
            with self.assertRaises(ValueError):
                nc.install_native_concurrency(3)

    def test_unbound_generator_runs_the_native_methods(self):
        """Hooks installed, no guard bound: upstream behaviour, nothing checked or counted."""
        s = nf.Scenario(nf.Engine())
        with s.installed():
            g, m, c = s.generator(b2=False)
            del g._glm_native_concurrency
            job = s.Job(nf.prompt(300), 32)
            s.run([(job, 0)])
        self.assertEqual(job.text, nf.reference_text(lambda: nf.Scenario(nf.Engine()), nf.prompt(300), 32))
        self.assertEqual(m.stats()['peak_target_batch'], 0)

    def test_pins_refuse_the_stage_1_engine_shape(self):
        """A tree whose sliding_attn.py lacks the U1 stash fix (stage 1) is refused before any hook."""
        import shutil
        with tempfile.TemporaryDirectory() as td:
            for relative in nc.ENGINE_SOURCE_PINS:
                (Path(td) / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ENGINE / relative, Path(td) / relative)
            target = Path(td) / 'exllamav3/modules/sliding_attn.py'
            target.write_text(target.read_text().replace('l.stash(self.slot, fill)', 'l.stash(self.slot, self.position)'))
            with self.assertRaisesRegex(ValueError, 'sliding_attn.py'):
                nc.verify_engine_sources(td)


if __name__ == '__main__':
    unittest.main()
