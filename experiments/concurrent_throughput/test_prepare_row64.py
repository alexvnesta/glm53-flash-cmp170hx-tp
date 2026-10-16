import json
import unittest
import prepare_row16
import prepare_row32
import prepare_row64
import test_prepare_row16
from build_row64 import UNITS

class ExpansionTests(unittest.TestCase):
    def setUp(self):
        fixture=test_prepare_row16.PreparationTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        prepare_row16.prepare(fixture.engine,fixture.output,fixture.manifest)
        self.base=fixture.output.parent/'row32'
        prepare_row32.prepare(fixture.output/'engine',self.base,fixture.output/'engine.sha256')
        for rel,edits in prepare_row64.EDITS.items():
            if rel.endswith('mlp.h'):continue
            p=self.base/'engine'/rel;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('\n'.join(edits)+'\n')
        self.manifest=self.base/'engine.sha256'
        self.manifest.write_text(''.join(f'{prepare_row16.sha(p)}  {p.relative_to(self.base/"engine")}\n'
            for p in sorted((self.base/'engine').rglob('*')) if p.is_file()))
        self.target=self.base.parent/'row64'

    def test_preserves_base_and_updates_both_sort_keys_with_the_slot_bound(self):
        before=self.manifest.read_text()
        r=prepare_row64.prepare(self.base/'engine',self.target,self.manifest)
        prepare_row16.verify(self.base/'engine',self.manifest)
        prepare_row16.verify(self.target/'engine',self.target/'engine.sha256')
        self.assertEqual(before,self.manifest.read_text())
        self.assertEqual(r['decode_max_rows'],64)
        self.assertEqual(r['max_routed_slots'],512)
        self.assertEqual(len(r['changed_sources']),3)
        text=(self.target/'engine'/next(k for k in prepare_row64.EDITS if k.endswith('.cuh'))).read_text()
        self.assertNotIn('<< 8',text);self.assertEqual(text.count('<< 9'),2)
        self.assertEqual(len(UNITS),15)
        self.assertEqual(len({str(p) for p in UNITS}),15)

    def test_changed_sort_anchor_fails_before_creating_output(self):
        rel=next(k for k in prepare_row64.EDITS if k.endswith('.cuh'))
        p=self.base/'engine'/rel;p.write_text(p.read_text().replace('(eu << 8) | u','different-key'))
        self.manifest.write_text(self.manifest.read_text().replace(
            next(s.split()[0] for s in self.manifest.read_text().splitlines() if s.endswith(rel)),prepare_row16.sha(p)))
        with self.assertRaisesRegex(ValueError,'Source anchor changed'):
            prepare_row64.prepare(self.base/'engine',self.target,self.manifest)
        self.assertFalse(self.target.exists())

if __name__=='__main__':unittest.main()
