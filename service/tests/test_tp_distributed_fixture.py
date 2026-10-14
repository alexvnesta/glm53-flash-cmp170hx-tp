"""Portable synthetic-fixture contracts; token counts are recorded provenance.

The standard-library checks do not independently tokenize the model template.
Pinned tokenizer/template calibration and runtime usage are separate evidence.
"""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'
VALUES = ('MICA-4827','FERN-9031','OPAL-1764','REED-6258')
LABELS = ('MARKER_START','MARKER_FIRST_MIDDLE','MARKER_SECOND_MIDDLE','MARKER_END')


class FixtureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.new=json.loads((FIXTURES/'tp_distributed_markers_v2.json').read_text())
        cls.old=json.loads((FIXTURES/'tp_distributed_markers_v1_invalid_geometry.json').read_text())
        cls.case=cls.new['case'];cls.req=cls.case['request']
        cls.user=next(m['content'] for m in cls.req['messages'] if m['role']=='user')

    def test_versions_are_distinct(self):
        self.assertEqual(self.new['fixture_revision'],'distributed-high-markers-v2')
        self.assertEqual(self.case['id'],'distributed_high_markers_v2')
        self.assertEqual(self.old['case']['id'],'distributed_high_markers')

    def test_genuine_large_corpus_preserved(self):
        self.assertGreater(len(self.user),350000)
        self.assertGreater(self.user.count('\n'),4000)

    def test_unique_values_and_labels(self):
        for label,value in zip(LABELS,VALUES):
            with self.subTest(label=label):
                self.assertEqual(self.user.count(label+'='),1)
                self.assertEqual(self.user.count(value),1)
        self.assertEqual(self.case['oracle'],{'kind':'ordered_markers','values':list(VALUES)})

    def test_character_extent_is_distributed(self):
        locations=[self.user.index(v)/len(self.user) for v in VALUES]
        self.assertLess(locations[0],.01)
        self.assertGreater(locations[1],.25);self.assertLess(locations[1],.40)
        self.assertGreater(locations[2],.60);self.assertLess(locations[2],.75)
        self.assertGreater(locations[3],.98)

    def test_recorded_token_extent_is_distributed(self):
        positions=self.new['provenance']['marker_positions']
        self.assertEqual([p['token_position'] for p in positions],[52,44440,88845,133257])
        self.assertEqual([p['value'] for p in positions],list(VALUES))
        for p in positions:self.assertAlmostEqual(p['fraction'],p['token_position']/133299)

    def test_cpu_and_owner_token_counts_and_runtime_gates(self):
        self.assertEqual(self.new['provenance']['cpu_rendered_prompt_tokens'],133299)
        self.assertEqual(self.new['provenance']['owner_endpoint_tokenize_tokens'],133299)
        self.assertEqual(self.case['expected_prompt_tokens'],133299)
        self.assertEqual(self.case['expected_prompt_token_range'],[110000,140000])

    def test_request_settings_preserved(self):
        self.assertEqual({k:v for k,v in self.req.items() if k!='messages'},
            {'model':'GLM-5.3-Flash','temperature':0,'top_p':1,'seed':170,
             'max_tokens':128,'ignore_eos':False,'stream':False,'reasoning_effort':'low','store':False})
        self.assertEqual([m['role'] for m in self.req['messages']],['system','user'])

    def test_historical_bad_range_remains_bad(self):
        self.assertEqual(self.old['observed_prompt_tokens'],1476)
        low,high=self.old['case']['expected_prompt_token_range']
        self.assertFalse(low<=self.old['observed_prompt_tokens']<=high)
        self.assertEqual(self.old['status'],'historical_geometry_failure_preserved')
        self.assertLess(len(self.old['case']['request']['messages'][0]['content']),4000)

    def test_source_hash_shape_and_pinned_tokenizer(self):
        sources=self.new['provenance']['source_hashes']
        for source in sources.values():
            self.assertRegex(source['sha256'],r'^[0-9a-f]{64}$')
            self.assertGreater(source['bytes'],0)
        self.assertEqual(sources['tokenizer']['sha256'],'19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d')

    def test_fixtures_have_no_private_paths_or_native_receipts(self):
        for name in ('tp_distributed_markers_v2.json','tp_distributed_markers_v1_invalid_geometry.json'):
            text=(FIXTURES/name).read_text()
            for forbidden in ('/Users/','/data2/','/home/','/mnt/','GPU-','InvocationID'):
                self.assertNotIn(forbidden,text)

    def test_pending_summary_does_not_claim_finished_replay(self):
        # The completed audit update changes this status after bounded live proof.
        value=json.loads((ROOT/'docs/qualification_summary.json').read_text())['full384_multi_session_successor']
        self.assertIn(value['status'],('awaiting_complete_replay_audit','bounded_full384_multi_session_lifecycle_qualified'))
        if value['status']=='awaiting_complete_replay_audit':self.assertFalse(value['overall_qualification_claimed'])
        self.assertFalse(value['known64k_profile_semantically_supported'])


if __name__=='__main__':unittest.main(verbosity=2)
