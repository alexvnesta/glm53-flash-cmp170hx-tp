"""Chunk geometry contracts from the actual API helper, without loading CUDA.

The old lifespan/CLI fakes targeted the retired MTP service and raised missing-
global errors after P16. Native lifecycle coverage lives in concurrent_cpu and
in the GPU qualification window; this module tests the retained chunk contract.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

API = Path(__file__).resolve().parents[1] / 'glm' / 'glm_api.py'
source = ast.parse(API.read_text())
helper = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'prefill_chunk_size')
namespace = {'PREFILL_PAGE_SIZE': 256}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(API), 'exec'), namespace)
prefill_chunk_size = namespace['prefill_chunk_size']


class Contracts(unittest.TestCase):
    def test_default_uses_loader_chunk(self):
        for n in (256, 512, 2048):
            with self.subTest(n=n):
                self.assertEqual(prefill_chunk_size(SimpleNamespace(chunk_size=n)), n)

    def test_smaller_cap_does_not_change_loader_geometry(self):
        args = SimpleNamespace(chunk_size=2048, prefill_chunk_size=512)
        self.assertEqual(prefill_chunk_size(args), 512)
        self.assertEqual(args.chunk_size, 2048)

    def test_equal_cap_is_valid(self):
        self.assertEqual(prefill_chunk_size(SimpleNamespace(chunk_size=2048, prefill_chunk_size=2048)), 2048)

    def test_bad_loader_chunks_are_rejected(self):
        for n in (None, True, False, 0, -256, 257, 2048.0, '2048'):
            with self.subTest(n=n), self.assertRaises(ValueError):
                prefill_chunk_size(SimpleNamespace(chunk_size=n))

    def test_bad_caps_are_rejected(self):
        for n in (True, False, 0, -256, 257, 512.0, '512', 4096):
            with self.subTest(n=n), self.assertRaises(ValueError):
                prefill_chunk_size(SimpleNamespace(chunk_size=2048, prefill_chunk_size=n))


if __name__ == '__main__': unittest.main()
