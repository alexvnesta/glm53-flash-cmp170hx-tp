import json
import unittest
import sys

from code_oracle import validate

GOOD = '''def compress_runs(values):
    result = []
    for value in values:
        if result and result[-1][0] == value:
            result[-1][1] += 1
        else:
            result.append([value, 1])
    return result
'''


class CodeOracleTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Runtime oracle requires Linux resource limits")
    def test_checks_empty_negative_repeated_and_separated_runs(self):
        self.assertTrue(validate(json.dumps({"code": GOOD})))
        self.assertFalse(validate(json.dumps({"code": "def compress_runs(values):\n    return []\n"})))

    def test_imports_reflection_and_module_level_code_are_refused(self):
        for code in ("import os\n" + GOOD, "print('x')\n" + GOOD,
                     "def compress_runs(values):\n    return values.__class__\n",
                     "def compress_runs(values):\n    return __import__('os')\n"):
            with self.subTest(code=code):
                self.assertFalse(validate(json.dumps({"code": code})))


if __name__ == "__main__":
    unittest.main()
