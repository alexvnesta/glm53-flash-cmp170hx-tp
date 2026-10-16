import ast
import unittest
import tempfile
import hashlib
from pathlib import Path
from types import SimpleNamespace

from prepare_nccl_broadcast import OLD, FLAG_ANCHOR, REL, patched, prepare
from prepare_row16 import verify


class BroadcastPolicyTests(unittest.TestCase):
    def backend(self, mode):
        source=patched(FLAG_ANCHOR+'\nclass TPBackendNCCL:\n'+OLD)
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        calls=[]
        ns={'torch':SimpleNamespace(Tensor=object),'_nccl_broadcast':mode,
            'dist':SimpleNamespace(broadcast=lambda *a,**k:calls.append(('nccl',a,k)))}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'variant','exec'),ns)
        obj=ns['TPBackendNCCL']();obj.active_devices=[4,2]
        obj.fallback=SimpleNamespace(broadcast=lambda *a:calls.append(('native',a,{})))
        return obj,calls

    def test_threshold_uses_bytes_and_source_device_maps_to_rank(self):
        obj,calls=self.backend('bulk')
        tensor=lambda n,width:SimpleNamespace(numel=lambda:n,element_size=lambda:width)
        for n,width,expected in ((1024,2,'native'),(1025,2,'nccl'),(257,8,'nccl'),(0,8,'native')):
            obj.broadcast(tensor(n,width),2)
            self.assertEqual(calls[-1][0],expected)
            if expected=='nccl':self.assertEqual(calls[-1][2],{'src':1,'async_op':False})

    def test_default_preserves_native_and_unknown_source_fails(self):
        obj,calls=self.backend('native')
        t=SimpleNamespace(numel=lambda:8192,element_size=lambda:4)
        obj.broadcast(t,4);self.assertEqual(calls[-1][0],'native')
        obj,calls=self.backend('bulk')
        with self.assertRaises(ValueError):obj.broadcast(t,7)
        self.assertEqual(calls,[])

    def test_preparation_attests_new_source_and_leaves_original_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=root/'original';path=engine/REL
            path.parent.mkdir(parents=True)
            path.write_text(FLAG_ANCHOR+'\nclass TPBackendNCCL:\n'+OLD)
            original=path.read_bytes()
            manifest=root/'engine.sha256'
            manifest.write_text(hashlib.sha256(original).hexdigest()+'  '+REL+'\n')
            output=root/'candidate'
            result=prepare(engine,manifest,output)
            verify(engine,manifest);verify(output/'engine',output/'engine.sha256')
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(set(result['changed_sources']),{REL})
            self.assertFalse(result['native_rebuild_required'])
            with self.assertRaisesRegex(ValueError,'Destination'):
                prepare(engine,manifest,output)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'identity'):
                prepare(engine,manifest,root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_invalid_mode_and_source_drift_fail(self):
        with self.assertRaisesRegex(ValueError,'anchors'):
            patched(FLAG_ANCHOR+'\n'+OLD.replace('self.fallback','self.other'))
        source=patched(FLAG_ANCHOR+'\nclass TPBackendNCCL:\n'+OLD)
        with self.assertRaisesRegex(ValueError,'native or bulk'):
            exec(source,{'os':SimpleNamespace(environ={'EXL3_TP_NCCL_BROADCAST':'typo'})})


if __name__=='__main__':unittest.main()
