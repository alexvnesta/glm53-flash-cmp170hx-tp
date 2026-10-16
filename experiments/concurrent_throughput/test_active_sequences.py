import hashlib
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from prepare_active_sequences import capacity_argv, service_overlay
from prepare_row16 import sha
import sustained


class ActiveSequenceTests(unittest.TestCase):
    def test_reviewed_service_overlay_requires_original_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';source=Path(tmp)/'source';base.mkdir();source.mkdir()
            (base/'guard.py').write_text('two slots\n');(source/'guard.py').write_text('owned slots\n')
            pins={'guard.py':sha(base/'guard.py')}
            with patch.dict('prepare_active_sequences.SERVICE_BASE_SHA256',pins,clear=True):
                receipt=service_overlay(base,source)
                self.assertEqual(receipt['guard.py']['after'],sha(source/'guard.py'))
                (base/'guard.py').write_text('unexpected change\n')
                with self.assertRaisesRegex(ValueError,'drift'):service_overlay(base,source)

    def test_only_capacity_and_allocation_parity_change(self):
        argv=['python','api.py','-ambs','2','--concurrent-max-sequences','2',
              '-cs','655360','--context-cap','392960','-ndt','7','--hybrid-draft-tokens','15']
        actual=capacity_argv(argv,4)
        self.assertEqual([(a,b) for a,b in zip(argv,actual) if a!=b],[('2','4'),('2','4')])
        self.assertEqual(argv[3],'2')
        for bad in (argv+['-ambs','2'], ['-ambs','4','--concurrent-max-sequences','2']):
            with self.assertRaises(ValueError): capacity_argv(bad,4)

    def test_fixture_is_identical_to_prior_matched_runs(self):
        historical=json.loads((Path(__file__).parent/'moe_fp32_2026-10-16.json').read_text())
        digest=hashlib.sha256(json.dumps(sustained.fixtures(24),sort_keys=True).encode()).hexdigest()
        # Find the frozen fixture identity regardless of report wrapper fields.
        self.assertIn(digest,json.dumps(historical))

    def test_four_active_evidence_is_distinct_from_two_active(self):
        rows=[{'agent':a,'usage':{'completion_tokens':100},'correct':True,'finish_s':1} for a in range(4)]
        samples=[{'seconds':s,'agent_tokens':[n]*4,'active_sequences':4,'waiting':0}
                 for s,n in ((5,10),(10,90))]
        d=sustained.summarize(rows,samples,4,15,active_sequences=4)
        self.assertEqual(d['steady']['fraction_capacity_active'],1)
        self.assertEqual(d['steady']['fraction_two_active'],0)
        self.assertEqual(d['steady']['peak_active_sequences'],4)
        self.assertAlmostEqual(sum(a['delivered_tok_s'] for a in d['agents']),d['aggregate_tok_s'])


if __name__=='__main__': unittest.main()
