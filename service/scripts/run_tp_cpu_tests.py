"""TP CPU contracts on explicit current engine source, with CUDA hidden."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine-root',type=Path,required=True)
    args=parser.parse_args()
    source=args.engine_root.resolve(strict=True)
    if not (source/'exllamav3/generator/generator.py').is_file():
        parser.error('Supply an ExLlama engine source checkout')
    env=os.environ.copy()
    env['EXLLAMAV3_ENGINE_ROOT']=str(source)
    env['CUDA_VISIBLE_DEVICES']=''
    env['HIP_VISIBLE_DEVICES']=''
    env['GLM53_CPU_DUPLICATE_RECYCLE']='0'
    program="""import unittest,json,torch
before=torch.cuda.is_initialized()
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_tp_*.py'))
after=torch.cuda.is_initialized()
print(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),'cuda_initialized_before':before,'cuda_initialized_after':after}))
raise SystemExit(not result.wasSuccessful() or before or after)
"""
    return subprocess.run([sys.executable,'-c',program],cwd=Path(__file__).resolve().parents[1],env=env).returncode


if __name__=='__main__':raise SystemExit(main())
