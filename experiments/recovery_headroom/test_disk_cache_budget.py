"""Execute real disk-cache methods with CPU byte/TP-I/O boundaries, never model math."""
import ast
from collections import deque
import heapq
import multiprocessing
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RELATIVE = 'exllamav3/generator/disk_cache.py'
METHODS = {'write_page', 'write_state', '_make_room', '_evict_one', '_drop', '_enqueue',
           '_rooted_elsewhere', '_build_order', '_reclaim_root', '_state_written'}


class TokenBytes:
    def contiguous(self): return self
    def numpy(self): return self
    def tobytes(self): return bytes(16)
    def clone(self): return self


def load(source, class_name='DiskPageCache', methods=METHODS):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.body = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name in methods]
    assert {m.name for m in cls.body} == methods
    cls.bases, cls.keywords = [], []
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, 'id', None) in ('PAGE_MAGIC', '_HEADER', '_STRAY_AGE') for t in n.targets)]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    env = {'deque': deque, 'heapq': heapq, 'os': os, 'shutil': shutil, 'struct': struct, 'time': time,
           'mp_disk_write': object()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *constants, cls], type_ignores=[])),
                 'actual_pinned_disk_cache', 'exec'), env)
    return env[class_name]


def disk(source):
    cls = load(source); d = cls.__new__(cls)
    d._lock = threading.RLock(); d._cv = threading.Condition(d._lock)
    d.pages = {}; d.states = {}; d.total_size = 0; d.max_size = 100000; d.root_size = 100000
    d.pending_pages = set(); d.pending_states = {}; d._queue = deque(); d._tp_waits = []
    d._order = deque(); d._order_pops = 0; d._order_rebuild = 64; d._deferred_release = {}
    d.cpu_tier = set(); d.pagetable = NS(get_live_page=lambda h: None)
    d.metrics = {'evictions': 0, 'skipped_mm': 0, 'state_candidates_dropped': 0}
    d.image_size = 16; d._tmp_suffix = '.tmp'; d.tp_groups = []; d.write_disabled = False
    d._can_write = Mock(return_value=True); d.drain_ready_states = Mock()
    d._spaced = Mock(return_value=True); d._sides = Mock(return_value=({}, {}))
    d._state_layout = Mock(return_value={'markers': None}); d._write_failed = Mock()
    d._page_path = Mock(return_value='/not-created/page'); d._create_page_file = Mock(return_value=128)
    return d


def baseline_hang(source, entered):
    d = disk(source); d.max_size = 0; d.total_size = 8
    d.states[b'x'] = {'size': 8, 'position': 256, 'last_access': 1}; d.cpu_tier.add(b'x')
    with d._lock:
        entered.set()
        d._make_room(1)


class DiskBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (REPO/'engine-src'/RELATIVE).read_text()
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name); path = root/RELATIVE; path.parent.mkdir(parents=True); path.write_text(cls.original)
        subprocess.run(['git', 'apply', '--check', str(HERE/'0002-disk-cache-budget-progress.patch')], cwd=root, check=True)
        subprocess.run(['git', 'apply', str(HERE/'0002-disk-cache-budget-progress.patch')], cwd=root, check=True)
        cls.fixed = path.read_text()

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_retained_baseline_negative_control_really_spins(self):
        context = multiprocessing.get_context('spawn'); entered = context.Event()
        child = context.Process(target=baseline_hang, args=(self.original, entered)); child.start()
        try:
            self.assertTrue(entered.wait(5), 'baseline did not enter its actual method')
            child.join(.2)
            self.assertTrue(child.is_alive(), 'baseline did not reproduce the captured no-progress loop')
        finally:
            if child.is_alive(): child.terminate()
            child.join(5)

    def test_rooted_state_with_zero_or_small_positive_share_returns_without_eviction(self):
        for quota in (0, 4):
            d = disk(self.fixed); d.max_size = quota; d.total_size = 8
            d.states[b'x'] = {'size': 8, 'position': 256, 'last_access': 1}; d.cpu_tier.add(b'x')
            with d._lock: self.assertFalse(d._make_room(1))
            self.assertEqual(d.total_size, 8); self.assertIn(b'x', d.states)
            acquired = []
            def check():
                ok = d._lock.acquire(timeout=.2); acquired.append(ok)
                if ok: d._lock.release()
            thread = threading.Thread(target=check); thread.start(); thread.join(1)
            self.assertEqual(acquired, [True])

    def test_oversized_write_does_not_evict_committed_data(self):
        d = disk(self.fixed); d.max_size = 4; d.total_size = 4
        d.pages[b'p'] = {'size': 4, 'prev_hash': None, 'last_access': 1}
        with d._lock: self.assertFalse(d._make_room(5))
        self.assertIn(b'p', d.pages); self.assertFalse(d._queue)

    def test_pending_reservations_are_not_overcommitted(self):
        d = disk(self.fixed); d.max_size = d.total_size = 8; d.pending_pages.add(b'pending')
        with d._lock: self.assertFalse(d._make_room(1))
        self.assertEqual(d.pending_pages, {b'pending'}); self.assertEqual(d.total_size, 8)

    def test_actual_lru_tree_still_evicts_leaf_before_root(self):
        d = disk(self.fixed); d.max_size = 16; d.total_size = 24
        for i, (key, prev) in enumerate(((b'a', None), (b'b', b'a'), (b'c', b'b'))):
            d.pages[key] = {'size': 8, 'prev_hash': prev, 'last_access': i}
        with d._lock: self.assertTrue(d._make_room(8))
        self.assertEqual(set(d.pages), {b'a'}); self.assertEqual(d.total_size, 8)
        self.assertEqual(d.metrics['evictions'], 2)

    def test_protected_sibling_root_reclaim_reports_zero_share_without_hanging(self):
        d = disk(self.fixed); d.root_size = 8; d.total_size = 1
        d.states[b'x'] = {'size': 1, 'position': 256, 'last_access': 1}; d.cpu_tier.add(b'x')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); own = root/'own'; own.mkdir(); sibling = root/'retained'; sibling.mkdir()
            (sibling/'manifest.json').write_text('{}'); (sibling/'index.sqlite').write_bytes(bytes(32))
            d.root_dir = str(root); d.dir = str(own)
            d._dir_size = lambda path: sum(p.stat().st_size for p in Path(path).iterdir() if p.is_file())
            d._reclaim_root()
            self.assertEqual(d.max_size, 0); self.assertTrue((sibling/'index.sqlite').exists())
            self.assertIn(b'x', d.states)

    def test_rejected_page_and_state_do_no_file_or_rank_io(self):
        d = disk(self.fixed); d.max_size = 0; model = NS(tp_dispatch_all=Mock())
        d.tp_groups = [(model, [], [0, 1])]
        self.assertFalse(d.write_page(b'p', None, TokenBytes(), 0, []))
        self.assertFalse(d.write_state(b's', {'checkpoint_size': 1, 'position': 256}))
        model.tp_dispatch_all.assert_not_called(); d._create_page_file.assert_not_called()
        d._state_layout.assert_not_called(); self.assertFalse(d.pending_pages or d.pending_states or d._queue)
        self.assertEqual(d.total_size, 0); self.assertEqual(d.metrics['skipped_budget'], 2)

    def test_page_reservation_precedes_tp_dispatch_and_is_queued_once(self):
        d = disk(self.fixed)
        def dispatch(*args):
            self.assertIn(b'p', d.pending_pages); self.assertGreater(d.total_size, 0)
        model = NS(tp_dispatch_all=Mock(side_effect=dispatch)); d.tp_groups = [(model, [], [0, 1])]
        self.assertTrue(d.write_page(b'p', None, TokenBytes(), 0, []))
        model.tp_dispatch_all.assert_called_once(); self.assertEqual(len(d._queue), 1)
        self.assertEqual(d.pending_pages, {b'p'}); self.assertEqual(d._queue[0][6], d.total_size)

    def test_file_creation_error_releases_reservation_without_tp_copy(self):
        d = disk(self.fixed); model = NS(tp_dispatch_all=Mock()); d.tp_groups = [(model, [], [0, 1])]
        d._create_page_file.side_effect = OSError('synthetic disk failure')
        self.assertFalse(d.write_page(b'p', None, TokenBytes(), 0, []))
        self.assertEqual(d.total_size, 0); self.assertFalse(d.pending_pages or d._queue)
        model.tp_dispatch_all.assert_not_called(); d._write_failed.assert_called_once()

    def test_rank_transport_failure_is_not_swallowed_or_reserved_forever(self):
        d = disk(self.fixed); error = BrokenPipeError('synthetic rank loss')
        d.tp_groups = [(NS(tp_dispatch_all=Mock(side_effect=error)), [], [0, 1])]
        with self.assertRaises(BrokenPipeError) as raised: d.write_page(b'p', None, TokenBytes(), 0, [])
        self.assertIs(raised.exception, error); self.assertEqual(d.total_size, 0)
        self.assertFalse(d.pending_pages or d._queue); d._write_failed.assert_not_called()

    def test_state_reservation_and_layout_failure_accounting(self):
        d = disk(self.fixed); stash = {'checkpoint_size': 8, 'position': 256}
        def layout(*args):
            self.assertIs(d.pending_states[b's'], stash); self.assertEqual(d.total_size, 4104)
            raise OSError('synthetic layout failure')
        d._state_layout.side_effect = layout
        self.assertFalse(d.write_state(b's', stash)); self.assertEqual(d.total_size, 0)
        self.assertFalse(d.pending_states or d._queue); d._write_failed.assert_called_once()

    def test_actual_cpu_store_retains_ram_entry_when_disk_budget_refuses(self):
        d = disk(self.fixed); d.max_size = 0
        source = (REPO/'engine-src/exllamav3/generator/cpu_cache.py').read_text()
        cls = load(source, 'CPUPageCache', {'store'}); tier = cls.__new__(cls)
        tier.entries = {}; tier._new_slot = Mock(return_value=0); tier.segments = []; tier.slot_views = []
        tier.tp = False; tier.disk_tier = d; tier._copy_events = Mock(return_value=[]); tier.slot_events = {}
        tier.metrics = {'pushes': 0, 'dedup_hits': 0}
        page = NS(phash=b'p', prev_hash=None, sequence=TokenBytes(), page_index=0, persistable=True)
        tier.store(page, serial=1); tier.store(page, serial=2)
        self.assertEqual(tier.entries[b'p']['slot'], 0); self.assertEqual(tier.metrics, {'pushes': 1, 'dedup_hits': 1})
        self.assertFalse(d.pending_pages or d._queue); self.assertEqual(d.metrics['skipped_budget'], 1)


if __name__ == '__main__': unittest.main()
