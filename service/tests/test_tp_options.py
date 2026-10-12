"""Service option and actual API AST contracts, no engine import."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glm_tp_options", ROOT / "glm/glm_tp_options.py")
options = importlib.util.module_from_spec(spec); spec.loader.exec_module(options)


def args(**kwargs):
    values = dict(experimental_dflash2_tp=False, tensor_parallel=False, draft_model_dir="/ABS/draft",
                  mtp=False, dflash_session_cache=False, target_cpu_cache_gib=0, cpu_cache_size=0)
    values.update(kwargs); return SimpleNamespace(**values)


class ServiceOptionsTests(unittest.TestCase):
    def test_normal_layer_split_is_unchanged_and_disabled(self):
        self.assertIs(options.tp_dflash_options(args()), False)

    def test_tp_dflash_requires_explicit_true(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            options.tp_dflash_options(args(tensor_parallel=True))
        self.assertIs(options.tp_dflash_options(args(tensor_parallel=True, experimental_dflash2_tp=True)), True)

    def test_reject_non_boolean(self):
        for value in (None, "1", 1, 0):
            with self.assertRaisesRegex(ValueError, "bool"):
                options.tp_dflash_options(args(experimental_dflash2_tp=value))

    def test_opt_in_cannot_enable_wrong_or_generic_tier_profiles(self):
        for change in ({"mtp": True}, {"tensor_parallel": False}, {"tensor_parallel": 1},
                       {"draft_model_dir": None}, {"cpu_cache_size": 1}):
            a = args(tensor_parallel=True, experimental_dflash2_tp=True)
            vars(a).update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                options.tp_dflash_options(a)

    def test_health_uses_actual_loaded_gate(self):
        g = SimpleNamespace(model=SimpleNamespace(loaded_tp=True), dflash_draft=True,
                            draft_model=SimpleNamespace(config=SimpleNamespace(experimental_tensor_parallel=True)))
        self.assertTrue(options.tp_dflash_health(g))
        for key, value in (("model", SimpleNamespace(loaded_tp=False)), ("dflash_draft", False),
                           ("draft_model", None), ("draft_model", SimpleNamespace(config=SimpleNamespace(experimental_tensor_parallel=1)))):
            old = getattr(g, key); setattr(g, key, value)
            self.assertFalse(options.tp_dflash_health(g)); setattr(g, key, old)

    def test_actual_lifespan_validates_before_loader(self):
        tree = ast.parse((ROOT / "glm/glm_api.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
        statements = [ast.unparse(n) for n in fn.body]
        check = next(i for i, s in enumerate(statements) if "tp_dflash_options(args)" in s)
        allocation = next(i for i, s in enumerate(statements) if "model_init.init(" in s)
        self.assertLess(check, allocation)
        self.assertIn("model_init.add_args(parser, cache=True, add_draft_model_args=True", (ROOT / "glm/glm_api.py").read_text())


if __name__ == "__main__": unittest.main()
