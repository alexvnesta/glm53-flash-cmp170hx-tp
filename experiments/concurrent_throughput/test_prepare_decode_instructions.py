import hashlib
from pathlib import Path
import tempfile
import unittest
from prepare_row16 import verify
from prepare_decode_instructions import VABS_SOURCE,prepare,patch_header,table_header,codebook_bits,TEMPLATE,CALL,DECODE_CALL,HEADER,UNIT,DERIVED,TABLE


class LookupPreparationTests(unittest.TestCase):
    def test_original_sources_preserved_and_only_k3_derives_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); engine=root/'base'; engine.mkdir()
            source='namespace exl3_moe_coop_ns {\n'+TEMPLATE+'() {\n'+DECODE_CALL+'\n}\n'+CALL+' 0);\n'
            source+='gemv_tile<bits, cb, WIDE, HALF>(down);\nconstexpr int FOLD = 4;\n'
            originals={HEADER:source,UNIT:'#include "../exl3_moe_coop_kernel.cuh"\n',
                'exllamav3/exllamav3_ext/libtorch/mlp.h':'#define MAX_BSZN 32 // capacity\n'}
            for rel,text in originals.items():
                path=engine/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            manifest=root/'base.sha256'
            manifest.write_text(''.join(hashlib.sha256(s.encode()).hexdigest()+'  '+n+'\n' for n,s in sorted(originals.items())))
            candidate=root/'candidate';receipt=prepare(engine,manifest,candidate)
            verify(engine,manifest);verify(candidate/'engine',candidate/'engine.sha256')
            self.assertEqual((candidate/'engine'/HEADER).read_text(),source)
            derived=(candidate/'engine'/DERIVED).read_text()
            self.assertIn('gemv_tile<bits, cb, WIDE, HALF>(down)',derived)
            self.assertIn('constexpr int FOLD = 4;',derived)
            self.assertIn('__CUDA_ARCH__ == 800',derived)
            self.assertIn('if constexpr (GATE_UP && cb == 2 && !HALF)',derived)
            self.assertEqual(set(receipt['changed_sources']),{UNIT,DERIVED,TABLE})
            self.assertEqual(receipt['table_bytes'],131072)
            self.assertEqual((candidate/'engine'/TABLE).read_text(),table_header())
            with self.assertRaisesRegex(ValueError,'Destination'):prepare(engine,manifest,candidate)
            (engine/HEADER).write_text('drift')
            with self.assertRaisesRegex(ValueError,'identity'):prepare(engine,manifest,root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_table_is_finite_and_reproducible(self):
        values=[codebook_bits(i) for i in range(65536)]
        self.assertTrue(all(x & 0x7c00 != 0x7c00 for x in values))
        # Fingerprint fixed input arithmetic; GPU exhaustive oracle establishes equivalence.
        self.assertEqual(hashlib.sha256(table_header().encode()).hexdigest(),
            '1b97a885181dd3f9c54f2e6b75e0f0ed3419e063e41900bf84abb949712793f2')

    def test_vabsdiff_preserves_index_width_and_has_no_lookup_reads(self):
        source='namespace exl3_moe_coop_ns {\n'+TEMPLATE+'() {\n'+DECODE_CALL+'\n}\n'+CALL+' 0);\n'
        patched=patch_header(source,'vabsdiff')
        self.assertIn('exl3_mul1_vabsdiff::decode_pair(w0, w1)',patched)
        self.assertIn('dq8_mul1_vabsdiff(awv',patched)
        self.assertNotIn('__ldg',patched)
        self.assertIn('x0 &= 0xffffu; x1 &= 0xffffu;',VABS_SOURCE)
        self.assertEqual(VABS_SOURCE.count('vabsdiff4.u32.u32.u32.add'),2)
        with self.assertRaisesRegex(ValueError,'Unknown'):patch_header(source,'unknown')

    def test_changed_anchor_refused(self):
        with self.assertRaisesRegex(ValueError,'anchor'):patch_header(TEMPLATE+'\n'+CALL)


if __name__=='__main__':unittest.main()
