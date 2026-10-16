"""NVTX row-count instrumentation for an exclusive diagnostic API process.

Usage: python profile_rows_bootstrap.py PID_FILE API_PY [API arguments...]
Run under Nsight with the same attested engine/native environment as the API.
This adds timing ranges without changing the source tree. Diagnostic rates
are separate from the unprofiled frozen throughput cohort."""
import functools
import os
from pathlib import Path
import runpy
import sys

pidfile,api,*args=sys.argv[1:]
Path(pidfile).write_text(str(os.getpid())+'\n')
import torch
from exllamav3.generator.generator import Generator
original=Generator.iterate_gen
@functools.wraps(original)
def traced(self,results,draft_tokens=None):
    sequences=sum(len(job.sequences) for job in self.active_jobs if job.is_prefill_done())
    width=1 if draft_tokens is None else draft_tokens.shape[-1]+1
    torch.cuda.nvtx.range_push(f'TARGET_ROWS_{sequences*width}_SEQS_{sequences}_WIDTH_{width}')
    try:return original(self,results,draft_tokens)
    finally:torch.cuda.nvtx.range_pop()
Generator.iterate_gen=traced
sys.argv=[api,*args]
runpy.run_path(api,run_name='__main__')
