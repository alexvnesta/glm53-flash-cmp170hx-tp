import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

FILE=Path(__file__).resolve().parents[1]/'glm/glm_weightless_options.py'
spec=importlib.util.spec_from_file_location('glm_weightless_options',FILE)
options=importlib.util.module_from_spec(spec);spec.loader.exec_module(options)


class WeightlessStartupTests(unittest.TestCase):
    def test_disabled_requires_no_engine_or_gpu(self):
        self.assertEqual(options.runtime_status([0,1],{}),{'enabled':False})

    def test_current_parent_and_worker_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proc=root/'proc';receipts=root/'receipts';receipts.mkdir()
            env={'GLM_WEIGHTLESS_VECTOR':'vector','GLM_WEIGHTLESS_VALIDATE':'1',
                 'GLM_WEIGHTLESS_RECEIPTS':str(receipts),'GLM_WEIGHTLESS_SHA256':'a'*64,
                 'GLM_WEIGHTLESS_BASE_REVISION':'b'*40}
            projector=SimpleNamespace(alpha=2.0,vectors=dict.fromkeys(range(1,45)))
            modules={'exllamav3.modules':SimpleNamespace(transformer=SimpleNamespace(_weightless_projector=projector))}
            def receipt(pid,device):
                (receipts/f'numerics-{pid}.json').write_text(json.dumps({'pid':pid,'device':device,'passed':True,'cases':[{}]*18}))
            with patch.dict(sys.modules,modules):
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    options.runtime_status([0,1],env,proc)
                receipt(os.getpid(),'cuda:0')
                receipt(999998,'cuda:1')  # Stale PID cannot certify a new startup.
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    options.runtime_status([0,1],env,proc)
                child=proc/'999997';child.mkdir(parents=True)
                (child/'status').write_text(f'PPid:\t{os.getpid()}\n')
                receipt(999997,'cuda:1')
                status=options.runtime_status([0,1],env,proc)
                self.assertEqual(status['numerics_verified_devices'],['cuda:0','cuda:1'])
                self.assertEqual(status['layers'],44)


if __name__=='__main__':unittest.main()
