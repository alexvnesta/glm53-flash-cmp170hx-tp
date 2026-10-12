"""Portable CPU contracts plus execution of the actual upstream truncation method."""
import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "glm"))
from glm_dflash_options import DFlashOptions, generator_kwargs, summarize_draft_stats

ENGINE = Path(os.environ["EXLLAMAV3_ENGINE_ROOT"]).resolve()


def load_method(name):
    tree = ast.parse((ENGINE / "exllamav3/generator/generator.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Generator")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {"torch": torch, "PAGE_SIZE": 256, "cuda_sync_active": lambda: None,
          "time": SimpleNamespace(time=lambda: 0.0)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "generator_method", "exec"), ns)
    return ns[name]


class OptionsTests(unittest.TestCase):
    def test_fixed_defaults(self):
        self.assertEqual(generator_kwargs({}), {"dynamic_draft_tokens": False,
            "draft_confidence": 0.4, "record_draft_stats": False})

    def test_explicit_adaptive_and_stats(self):
        self.assertEqual(generator_kwargs({"GLM53_DYNAMIC_DRAFT": "1",
            "GLM53_RECORD_DRAFT_STATS": "true", "GLM53_DRAFT_CONFIDENCE": ".7"}),
            {"dynamic_draft_tokens": True, "draft_confidence": .7, "record_draft_stats": True})

    def test_stats_independent_of_adaptive(self):
        self.assertTrue(generator_kwargs({"GLM53_RECORD_DRAFT_STATS": "1"})["record_draft_stats"])
        self.assertFalse(generator_kwargs({"GLM53_RECORD_DRAFT_STATS": "1"})["dynamic_draft_tokens"])

    def test_invalid_switches(self):
        for value in ("yes", "", "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generator_kwargs({"GLM53_DYNAMIC_DRAFT": value})

    def test_invalid_confidence(self):
        for value in ("0", "1", "nan", "inf", "-0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generator_kwargs({"GLM53_DRAFT_CONFIDENCE": value})

    def test_geometry_is_not_reconfigured(self):
        for key, value in (("num_draft_tokens", 6), ("draft_block_size", 16),
                           ("max_q_size", 9), ("max_batch_size", 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                DFlashOptions(adaptive=True).generator_kwargs(**{key: value})
        self.assertEqual(set(generator_kwargs({})),
            {"dynamic_draft_tokens", "draft_confidence", "record_draft_stats"})

    def test_upstream_constructor_accepts_existing_kwargs(self):
        tree = ast.parse((ENGINE/"exllamav3/generator/generator.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=="Generator")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=="__init__")
        self.assertTrue(set(generator_kwargs({})).issubset({a.arg for a in fn.args.args}))

    def test_stats_denominator_and_nonmutation(self):
        rows = [(4, 7, 3), (8, 3, 3), (9, 0, 0)]
        original = list(rows)
        self.assertEqual(summarize_draft_stats(rows), {"rounds":3, "proposed":10,
            "accepted":6, "acceptance":.6})
        self.assertEqual(rows, original)

    def test_empty_stats_not_zero_acceptance(self):
        self.assertIsNone(summarize_draft_stats([])["acceptance"])

    def test_stats_refuses_invalid_observations(self):
        for row in ((1,7,8), (-1,7,0), (1,-1,0), (1,7,-1), (1,7,True), (1,7)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                summarize_draft_stats([row])


class UpstreamTruncationTests(unittest.TestCase):
    def run_round(self, scores, adaptive=True):
        observed = {}
        sequence = SimpleNamespace(block_index_tensor=torch.zeros((1,16),dtype=torch.long), kv_position=0)
        job = SimpleNamespace(is_prefill_done=lambda:True, get_max_seq_len=lambda:1,
            sequences=[sequence], time_first_token=1.0,
            get_input_ids_list=lambda:[torch.tensor([[42]])])
        def forward(*, input_ids, params):
            observed["params"] = params
            return torch.zeros((1,8,4))
        def sample(state, params):
            if params.get("export_draft_conf"):
                params["draft_conf"] = torch.tensor([[0.0]+scores])
            return torch.arange(8,dtype=torch.long)[None,:]
        draft = SimpleNamespace(config=SimpleNamespace(block_size=8), forward=forward,
                                sample_from_state=sample)
        generator = SimpleNamespace(active_jobs=[job], num_draft_tokens=7, draft_model=draft,
            draft_cache=object(), draft_calibrator=SimpleNamespace(threshold=lambda:5.0) if adaptive else None,
            draft_input_ids_pinned=torch.zeros((1,1),dtype=torch.long),
            draft_ids_pinned=torch.zeros((1,7),dtype=torch.long),
            _staging=lambda name,*shape:torch.zeros(shape,dtype=torch.long))
        output = load_method("iterate_draftmodel_dflash_gen")(generator, [])
        self.assertFalse(torch.cuda.is_initialized())
        return output, generator, observed

    def test_adaptive_cuts_at_first_low_confidence(self):
        output, generator, observed = self.run_round([9,8,4,9,9,9,9])
        self.assertEqual(output.tolist(), [[1,2]])
        self.assertEqual(generator._draft_conf_round["window"], 2)
        self.assertEqual(generator._draft_conf_round["conf"].shape, (1,7))
        self.assertTrue(observed["params"]["export_draft_conf"])

    def test_zero_window_uses_existing_target_only_round(self):
        output, generator, _ = self.run_round([4,9,9,9,9,9,9])
        self.assertIsNone(output)
        self.assertEqual(generator._draft_conf_round["window"], 0)

    def test_fixed_mode_keeps_k7(self):
        output, generator, observed = self.run_round([0]*7, adaptive=False)
        self.assertEqual(output.tolist(), [[1,2,3,4,5,6,7]])
        self.assertIsNone(generator._draft_conf_round)
        self.assertNotIn("export_draft_conf", observed["params"])

    def test_real_calibrator_burn_in_and_threshold(self):
        spec = importlib.util.spec_from_file_location("calibrator_cpu",
            ENGINE/"exllamav3/generator/draft_confidence.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cal = module.DraftConfidenceCalibrator(.5, burn_in=4, min_count=2)
        self.assertEqual(cal.threshold(), -float("inf"))
        for score, accepted in ((9,True),(9,True),(3,False),(3,False)):
            cal.add_label(score, accepted)
        self.assertEqual(cal.threshold(), 9)
        self.assertEqual(cal.estimate(9), 1.0)
        self.assertEqual(cal.estimate(3), 0.0)


if __name__ == "__main__":
    unittest.main()
