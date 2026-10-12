"""Run admission and actual ASGI ownership contracts without importing a model."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    repo=Path(__file__).resolve().parents[1]
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=''
    env['HIP_VISIBLE_DEVICES']=''
    env['PYTHONPATH']=str(repo/'glm')
    program="""import sys, unittest, json
assert 'torch' not in sys.modules
suite=unittest.TestSuite()
loader=unittest.TestLoader()
for name in ('test_admission.py','test_response_cleanup.py'):
    suite.addTests(loader.discover('tests/http',pattern=name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),'torch_imported':'torch' in sys.modules}))
raise SystemExit(not result.wasSuccessful() or 'torch' in sys.modules)
"""
    checked=subprocess.run([sys.executable,'-c',program],cwd=repo,env=env)
    if checked.returncode:return checked.returncode
    return subprocess.run([sys.executable,str(repo/'scripts/check_response_lifetime_ast.py')],cwd=repo,env=env).returncode


if __name__=='__main__':raise SystemExit(main())
