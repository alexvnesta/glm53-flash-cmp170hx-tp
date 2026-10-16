import hashlib
from pathlib import Path
import tempfile
import unittest
from prepare_row16 import verify
from prepare_routed_moe import prepare,patch_header,TEMPLATE,REG,CALL,HEADER,DERIVED,UNIT

class RoutedPreparationTests(unittest.TestCase):
    def test_only_k3_uses_derived_header_and_original_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=root/'base';engine.mkdir()
            sources={HEADER:TEMPLATE+'() {\n'+REG+'\n}\n'+CALL+' 0);\ngemv_tile<bits, cb, WIDE, HALF>(down);\n',
                UNIT:'#include "../exl3_moe_coop_kernel.cuh"\nEXL3_MOE_COOP_INST_DEF(3)\n',
                'exllamav3/exllamav3_ext/libtorch/mlp.h':'#define MAX_BSZN 32 // cap\n'}
            for name,source in sources.items():
                p=engine/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
            manifest=root/'base.sha256';manifest.write_text(''.join(hashlib.sha256(s.encode()).hexdigest()+'  '+n+'\n' for n,s in sorted(sources.items())))
            receipt=prepare(engine,manifest,root/'candidate');verify(engine,manifest)
            verify(root/'candidate/engine',root/'candidate/engine.sha256')
            self.assertEqual((root/'candidate/engine'/HEADER).read_text(),sources[HEADER])
            derived=(root/'candidate/engine'/DERIVED).read_text()
            self.assertIn('GATE_UP && bits == 3 && cb == 2',derived)
            self.assertIn('gemv_tile<bits, cb, WIDE, HALF>(down)',derived)
            self.assertEqual(set(receipt['changed_sources']),{UNIT,DERIVED})
            self.assertIsNone(receipt['changed_sources'][DERIVED]['before'])
            with self.assertRaisesRegex(ValueError,'Destination'):prepare(engine,manifest,root/'candidate')
            (engine/HEADER).write_text('drift')
            with self.assertRaisesRegex(ValueError,'identity'):prepare(engine,manifest,root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_prefetch_keeps_register_decode_and_fold_cadence(self):
        source=TEMPLATE+'() {\n'+REG+'\n}\n'+CALL+' 0);\nconstexpr int FOLD = 4;\n'
        result=patch_header(source,'prefetch8')
        self.assertIn(REG,result)
        self.assertIn('GATE_UP && bits == 3 && cb == 2 ? 8',result)
        self.assertIn('constexpr int FOLD = 4;',result)
        with self.assertRaisesRegex(ValueError,'Unknown'):patch_header(source,'other')

    def test_changed_anchor_refused_before_output(self):
        with self.assertRaisesRegex(ValueError,'anchor'):patch_header(TEMPLATE+'\n'+REG)

    def test_retile_is_guarded_and_leaves_down_call_unmodified(self):
        source=TEMPLATE+'() {\n'+REG+'''\n    constexpr int TWORDS = HALF ? 1 : 2;
    constexpr int WN = tile_wn<WIDE>();
    constexpr int WKK = tile_wk<WIDE>();
}\n'''+CALL+' 0);\ngemv_tile<bits, cb, WIDE, HALF>(down);\n'
        result=patch_header(source,'retile')
        self.assertIn('__CUDA_ARCH__ == 800',result)
        self.assertIn('GATE_UP && bits == 3 && cb == 2 && WIDE && !HALF',result)
        self.assertIn('constexpr int WNT = RETILE ? 1 :',result)
        self.assertIn('constexpr int COLS = WNT * 16;',result)
        self.assertIn('constexpr int WN = RETILE ? 8 :',result)
        self.assertIn('constexpr int WKK = RETILE ? 2 :',result)
        self.assertIn('gemv_tile<bits, cb, WIDE, HALF>(down)',result)
        with self.assertRaisesRegex(ValueError,'Retile anchor'):
            patch_header(source.replace('    constexpr int WN = tile_wn<WIDE>();',''), 'retile')

if __name__=='__main__':unittest.main()
