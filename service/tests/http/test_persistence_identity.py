import ast
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from glm_persistence import persistence_options, profile_identity, identity_digest, source_digest, persistence_health


class ProfileContracts(unittest.TestCase):
    def test_health_distinguishes_root_budget_from_exhausted_namespace_share(self):
        disk = NS(_lock=threading.RLock(), fingerprint='synthetic', root_size=50*1024**3,
                  max_size=0, total_size=8, pages={}, states={b'x': object()},
                  pending_pages=set(), pending_states={}, write_disabled=False,
                  metrics={'skipped_budget': 3})
        report = persistence_health(disk)
        self.assertEqual(report['budget_bytes'], 50*1024**3)
        self.assertEqual(report['namespace_budget_bytes'], 0)
        self.assertEqual(report['metrics']['skipped_budget'], 3)
        self.assertFalse(report['writes_disabled'])

    def test_options_require_owned_ram_tier_and_bounded_disk(self):
        args = NS(disk_cache_dir='/tmp/model-cache', disk_cache_size=50, persistent_cache_min_free_gib=16)
        with self.assertRaises(ValueError):
            persistence_options(args, NS(enabled=False, target_cpu_bytes=0))
        retention = NS(enabled=True, target_cpu_bytes=8*1024**3)
        self.assertEqual(persistence_options(args, retention).budget_bytes, 50*1024**3)
        for bad in (0, -1, float('nan'), float('inf'), True, 257):
            args.disk_cache_size = bad
            with self.assertRaises(ValueError):
                persistence_options(args, retention)
        args.disk_cache_size = 50
        args.disk_cache_dir = 'relative'
        with self.assertRaises(ValueError):
            persistence_options(args, retention)

    def test_identity_changes_with_steering_native_precision_and_source(self):
        with tempfile.TemporaryDirectory() as tmp, patch('glm_persistence.version', return_value='test-version'):
            root = Path(tmp)
            (root/'kernel.py').write_text('x=1\n')
            steering = {'enabled': True, 'alpha': 2, 'layers': 44, 'width': 16384,
                        'vector_sha256': 'a'*64, 'base_revision': 'b'*40}
            def make(**kw):
                return profile_identity(root, kw.get('native', 'c'*64), kw.get('steering', steering),
                                        kw.get('dtype', 'fp16'), {'EXL3_TEST': '1', 'HF_TOKEN': 'PRIVATE'})
            first = identity_digest(make())
            for changed in ({'native':'d'*64}, {'dtype':'fp32'},
                            {'steering':{**steering,'alpha':1}},
                            {'steering':{**steering,'vector_sha256':'e'*64}},
                            {'steering':{**steering,'layers':43}}):
                self.assertNotEqual(first, identity_digest(make(**changed)))
            self.assertNotIn('PRIVATE', json.dumps(make()))
            (root/'kernel.py').write_text('x=2\n')
            self.assertNotEqual(first, identity_digest(make()))

    def test_actual_upstream_disk_fingerprint_includes_profile(self):
        root = Path(__file__).resolve().parents[3]/'engine-src'
        source = root/'exllamav3/generator/disk_cache.py'
        tree = ast.parse(source.read_text())
        fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='cache_fingerprint']
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        env = {'PAGE_SIZE':256, '__version__':'test', 'hashlib':hashlib, 'json':json,
               'model_identity':lambda config:'model', '_layer_key':lambda k,l:k}
        env['DISK_CACHE_FORMAT_VERSION'] = next(ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='DISK_CACHE_FORMAT_VERSION' for t in n.targets))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), 'exec'), env)
        cache = NS(model=NS(config=object()), layers={})
        fp = env['cache_fingerprint']
        self.assertEqual(fp([cache], [], 4096), fp([cache], [], 4096, None))
        self.assertNotEqual(fp([cache], [], 4096, {'alpha':2})[0], fp([cache], [], 4096, {'alpha':1})[0])


if __name__=='__main__':
    unittest.main()
