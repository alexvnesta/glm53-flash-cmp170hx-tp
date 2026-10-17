"""Check the TP CPU export path without requiring Torch or CUDA on CI."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from prepare_active_kv import mapped_source
from prepare_dynamic_kv import managed_source


class CPUExport(unittest.TestCase):
    def test_metadata_allocation_never_enters_managed_cuda_pool(self):
        source = Path(__file__).resolve().parents[2]/'engine-src/exllamav3/cache/mla.py'
        tree = ast.parse(managed_source(mapped_source(source.read_text())))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='CacheLayer_MLA_quant')
        alloc = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='alloc')
        alloc.decorator_list = []
        for arg in alloc.args.args:
            arg.annotation = None
        module = ast.Module(body=[alloc], type_ignores=[])
        def zeros(shape, *, dtype, device):
            self.assertEqual(device, 'cpu')
            return (tuple(shape), dtype, device)
        torch = SimpleNamespace(device=lambda d:SimpleNamespace(type=d), zeros=zeros, int='int32', half='fp16')
        scope = {'torch':torch}
        exec(compile(ast.fix_missing_locations(module), '<actual-transformed-alloc>', 'exec'), scope)
        layer = SimpleNamespace(qshape=(1,256,128), sshape=(1,256,16), shape_r=None,
                                shape_i=None, shape_p=None, kidx_shadow_on=False)
        scope['alloc'](layer, 'cpu')
        self.assertEqual(layer.qk, ((1,256,128),'int32','cpu'))
        self.assertEqual(layer.sk, ((1,256,16),'fp16','cpu'))
        self.assertIsNone(layer._active_managed)
        self.assertIsNone(layer._active_host_arena)


if __name__ == '__main__':
    unittest.main()
