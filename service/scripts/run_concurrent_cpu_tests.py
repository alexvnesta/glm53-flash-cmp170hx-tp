"""Run concurrency contracts against an explicit engine tree without CUDA."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.engine_root.resolve(strict=True)
    if not (root / 'exllamav3/generator/generator.py').is_file():
        parser.error('Supply the pinned experimental engine source tree')
    env = os.environ.copy()
    env.update(EXLLAMAV3_ENGINE_ROOT=str(root), CUDA_VISIBLE_DEVICES='',
               HIP_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    program = """import json,torch,unittest
before=torch.cuda.is_initialized()
suite=unittest.defaultTestLoader.discover('tests/concurrent_cpu',pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=torch.cuda.is_initialized()
print(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),cuda_initialized_before=before,cuda_initialized_after=after)))
raise SystemExit(not result.wasSuccessful() or result.testsRun==0 or before or after)
"""
    return subprocess.run([sys.executable, '-c', program], env=env,
                          cwd=Path(__file__).resolve().parents[1]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
