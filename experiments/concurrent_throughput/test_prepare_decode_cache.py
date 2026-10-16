import hashlib
from pathlib import Path
import tempfile
import unittest
from prepare_decode_cache import prepare,patch_unit,patch_loads,MACRO,INSTANCES,ANCHOR,GEOMETRY
from prepare_decode_instructions import HEADER,UNIT,DERIVED,TEMPLATE,CALL,DECODE_CALL,patch_header
from prepare_row16 import verify


class CachePreparationTests(unittest.TestCase):
    def test_full_preparation_custody_and_down_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=root/'base';engine.mkdir()
            instances=MACRO+'MoeCoopKernel a_k##K() {\n'+ANCHOR.replace('<3>','<K>')+'\n}\nMoeCoopKernel b_k##K() { return down<K>(); }\n'
            sources={HEADER:'namespace exl3_moe_coop_ns {\n'+TEMPLATE+'() {\n'+DECODE_CALL+'\n}\n'+CALL+' 0);\n',
                UNIT:'#include "../exl3_moe_coop_kernel.cuh"\nEXL3_MOE_COOP_INST_DEF(3)\n',INSTANCES:instances,GEOMETRY:'#define MOE_COOP_WNT 2 // tiles\n',
                'exllamav3/exllamav3_ext/libtorch/mlp.h':'#define MAX_BSZN 32 // capacity\n'}
            for rel,s in sources.items():
                p=engine/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
            manifest=root/'base.sha256';manifest.write_text(''.join(hashlib.sha256(s.encode()).hexdigest()+'  '+n+'\n' for n,s in sorted(sources.items())))
            output=root/'candidate';r=prepare(engine,manifest,output)
            verify(engine,manifest);verify(output/'engine',output/'engine.sha256')
            self.assertEqual(r['kind'],'routed_a_decode_l1')
            unit=(output/'engine'/UNIT).read_text();header=(output/'engine'/DERIVED).read_text()
            self.assertIn('b_k3() { return down<3>(); }',unit)
            self.assertIn('smem - smem_stage_bytes<3>()',unit)
            self.assertIn('if (cb == 2)',unit)
            self.assertIn('cudaFuncAttributePreferredSharedMemoryCarveout, 33',unit)
            self.assertIn('major != 8 || minor != 0',unit)
            self.assertIn('ld.global.L1::evict_last.u16',header)
            self.assertNotIn('__ldg',header)
            self.assertEqual((output/'engine'/INSTANCES).read_text(),instances)
            self.assertEqual((output/'engine'/HEADER).read_text(),sources[HEADER])
            with self.assertRaisesRegex(ValueError,'Destination'):prepare(engine,manifest,output)

    def test_changed_anchors_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'Instance anchor'):patch_unit('changed','changed')
        with self.assertRaisesRegex(ValueError,'helper anchor'):patch_loads('changed')


if __name__=='__main__':unittest.main()
