import unittest

from gpu_oracles import expand_oracle, oracle_code


class OracleSourceTests(unittest.TestCase):
    def test_standalone_oracle_needs_no_pytest_runtime_or_test_decorators(self):
        source = '''import pytest

def run_case():
    out = torch.full((8, H_out), 0)
    ctr = smax * (I // 128) + 8 * (Ho // 128)

@pytest.mark.parametrize("value", [1])
def test_other(value):
    pass
'''
        namespace = {}
        exec(oracle_code(source, "fixture"), namespace)
        self.assertTrue(callable(namespace["run_case"]))
        self.assertNotIn("pytest", namespace)
        self.assertNotIn("test_other", namespace)

    def test_only_expands_run_case_even_when_another_test_repeats_counter_layout(self):
        source = '''def run_case():
    out = torch.full((8, H_out), 0)
    ctr = smax * (I // 128) + 8 * (Ho // 128)

def other_test():
    ctr = smax * (I // 128) + 8 * (Ho // 128)
'''
        expanded = expand_oracle(source)
        self.assertIn("torch.full((max(8, bsz), H_out),", expanded)
        self.assertEqual(expanded.count("max(8, bsz) * (Ho // 128)"), 1)
        self.assertIn("def other_test():\n    ctr = smax * (I // 128) + 8 * (Ho // 128)", expanded)


if __name__ == "__main__":
    unittest.main()
