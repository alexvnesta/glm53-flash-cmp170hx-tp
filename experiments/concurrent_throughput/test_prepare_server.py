from pathlib import Path
import unittest

from prepare_server import configuration_environment


class ServerEnvironmentTests(unittest.TestCase):
    def test_preserves_profile_flags_and_isolates_paths_without_credentials(self):
        raw = [b"EXL3_MLA_KIDX_RING=32", b"EXL3_BC_ATTN=1",
               b"EXL3_DSA_QC_STAGE_MIN_R=9", b"GLM53_DYNAMIC_DRAFT=0",
               b"NCCL_SOCKET_IFNAME=lo", b"TRITON_CACHE_DIR=/base/triton-cache",
               b"HF_TOKEN=private", b"ANTHROPIC_API_KEY=private"]
        env = configuration_environment(raw, Path("/base"), Path("/trial"))
        self.assertEqual(env, {"EXL3_MLA_KIDX_RING": "32", "EXL3_BC_ATTN": "1",
                               "EXL3_DSA_QC_STAGE_MIN_R": "9", "GLM53_DYNAMIC_DRAFT": "0",
                               "NCCL_SOCKET_IFNAME": "lo", "TRITON_CACHE_DIR": "/trial/triton-cache"})

    def test_unreviewed_engine_flag_fails_before_an_unmatched_comparison(self):
        with self.assertRaisesRegex(ValueError, "Unreviewed model configuration"):
            configuration_environment([b"EXL3_NEW_SETTING=private-value"], Path("/base"), Path("/trial"))


if __name__ == "__main__":
    unittest.main()
