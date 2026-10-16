import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare_row16 import prepare, verify
from build_row16 import compile_command


class PreparationTests(unittest.TestCase):
    def test_repeated_compile_flag_and_depfile_stay_inside_candidate(self):
        rel = Path("exllamav3/exllamav3_ext/bindings.cpp")
        command = ["c++", "-MF", "/old/build/" + str(rel.with_suffix(".o.d")),
                   "-I/old/engine/exllamav3", "-c", "-c", "/old/engine/" + str(rel),
                   "-o", "/old/build/" + str(rel.with_suffix(".o"))]
        rewritten = compile_command(command, rel, Path("/new/engine"), Path("/new/build"))
        self.assertFalse(any("/old/" in arg for arg in rewritten))
        self.assertEqual(rewritten[rewritten.index("-MF") + 1],
                         "/new/build/" + str(rel.with_suffix(".o.d")))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.engine, self.output = root / "base", root / "variant"
        self.engine.mkdir()
        fixtures = {
            "exllamav3/modules/mlp.py": "MAX_BSZN = 8  # dense\n",
            "exllamav3/modules/block_sparse_mlp.py": "MAX_BSZN = 8  # experts\n",
            "exllamav3/exllamav3_ext/libtorch/mlp.h": "#define MAX_BSZN 8 // capacity\n",
            "exllamav3/exllamav3_ext/bindings.cpp": "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n{\n}\n",
            "unchanged.txt": "original source\n",
        }
        for rel, text in fixtures.items():
            p = self.engine / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        self.manifest = root / "qualified.sha256"
        self.manifest.write_text("".join(
            hashlib.sha256(text.encode()).hexdigest() + "  " + rel + "\n"
            for rel, text in sorted(fixtures.items())
        ))

    def test_preserves_base_and_makes_a_verifiable_variant_requiring_new_native(self):
        original = self.manifest.read_text()
        receipt = prepare(self.engine, self.output, self.manifest)
        verify(self.engine, self.manifest)
        verify(self.output / "engine", self.output / "engine.sha256")
        self.assertEqual(self.manifest.read_text(), original)
        self.assertEqual(len(receipt["changed_sources"]), 4)
        self.assertEqual(receipt["status"], "unqualified_candidate")
        self.assertEqual((self.output / "engine/unchanged.txt").read_text(), "original source\n")
        self.assertIn("ext.decode_max_rows", (self.output / "engine/exllamav3/modules/mlp.py").read_text())

    def test_source_drift_is_rejected_before_creating_output(self):
        (self.engine / "unchanged.txt").write_text("drift")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            prepare(self.engine, self.output, self.manifest)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "preserve"
        marker.write_text("other work")
        with self.assertRaises(ValueError):
            prepare(self.engine, self.output, self.manifest)
        self.assertEqual(marker.read_text(), "other work")


if __name__ == "__main__":
    unittest.main()
