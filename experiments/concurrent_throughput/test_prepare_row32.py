import unittest

import prepare_row16
import prepare_row32
import test_prepare_row16


class ExpansionTests(unittest.TestCase):
    def test_row32_preserves_attested_row16_and_only_changes_capacity(self):
        fixture = test_prepare_row16.PreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        prepare_row16.prepare(fixture.engine,fixture.output,fixture.manifest)
        target = fixture.output.parent/'row32'
        manifest = fixture.output/'engine.sha256'
        result = prepare_row32.prepare(fixture.output/'engine',target,manifest)
        prepare_row16.verify(fixture.output/'engine',manifest)
        prepare_row16.verify(target/'engine',target/'engine.sha256')
        self.assertEqual(result['decode_max_rows'],32)
        self.assertEqual(list(result['changed_sources']),['exllamav3/exllamav3_ext/libtorch/mlp.h'])
        with self.assertRaisesRegex(ValueError,'row16 capacity'):
            prepare_row32.prepare(target/'engine',target.parent/'row64',target/'engine.sha256')
        self.assertFalse((target.parent/'row64').exists())


if __name__=='__main__':
    unittest.main()
