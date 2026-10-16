"""Exclusive two-rank content and ordering oracle for the actual TP backend.

Run with the prepared engine/native PYTHONPATH. Parent refuses occupied GPUs;
children are bounded and only child processes from this invocation are killed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from prepare_row16 import verify


def worker(a):
    import torch
    from exllamav3.model.model_tp_backend import TPBackendNCCL
    torch.cuda.set_device(a.rank)
    backend=TPBackendNCCL(a.rank,[0,1],0,'tcp://127.0.0.1:'+str(a.port),a.rank==0,a.uuid)
    rows=[]
    try:
        for dtype in (torch.uint8,torch.int64,torch.float16,torch.bfloat16,torch.float32):
            width=torch.empty((),dtype=dtype).element_size()
            # The retained native LL path requires even byte counts.
            for size in (max(2,width),2048-max(2,width),2048,2048+width,131072,1048576):
                n=size//width
                for src in (0,1):
                    # Integer-valued finite data has an exact representation in every dtype.
                    expected=((torch.arange(n,dtype=torch.int64)+17*src)%101).to(dtype)
                    value=expected.to(a.rank) if a.rank==src else torch.full((n,),-1 if dtype!=torch.uint8 else 255,dtype=dtype,device=a.rank)
                    backend.broadcast(value,src)
                    torch.cuda.synchronize()
                    assert torch.equal(value.cpu().view(torch.uint8),expected.view(torch.uint8))
                    # Interleave reductions with native-small and NCCL-large broadcasts.
                    reduced=torch.full((257,),a.rank+1,dtype=torch.float32,device=a.rank)
                    backend.all_reduce(reduced)
                    assert torch.equal(reduced,torch.full_like(reduced,3))
                    rows.append({'dtype':str(dtype),'bytes':size,'source_device':src,'bit_exact':True})
        a.output.write_text(json.dumps({'rank':a.rank,'passed':True,'cases':rows},indent=2)+'\n')
    finally:backend.close()


def main(a):
    verify(a.profile/'engine',a.profile/'engine.sha256')
    native,=(a.profile/'native').glob('exllamav3_ext*.so')
    sha=hashlib.sha256(native.read_bytes()).hexdigest()
    assert sha==json.loads((a.profile/'build-receipt.json').read_text())['native_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU owners remain'
    a.output.mkdir(parents=True,exist_ok=False)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'PYTHONPATH':str(a.profile/'native')+':'+str(a.profile/'engine'),
         'EXL3_TP_NCCL_BROADCAST':'bulk','NCCL_SOCKET_IFNAME':'lo','NCCL_DEBUG':'INFO'}
    ident='glm_broadcast_'+uuid.uuid4().hex
    children=[];logs=[];receipt={'passed':False,'native_sha256':sha,'ranks':[]}
    try:
        for rank in (0,1):
            log=(a.output/f'rank{rank}.log').open('w');logs.append(log)
            children.append(subprocess.Popen([sys.executable,__file__,'--rank',str(rank),'--port',str(port),'--uuid',ident,
                '--output',str(a.output/f'rank{rank}.json')],env=env,stdout=log,stderr=subprocess.STDOUT))
        until=time.monotonic()+120
        while any(p.poll() is None for p in children):
            if time.monotonic()>until:raise TimeoutError('Two-rank oracle exceeded 120 seconds')
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Rank failed; inspect logs')
            time.sleep(.2)
        assert all(p.returncode==0 for p in children)
        receipt['ranks']=[json.loads((a.output/f'rank{r}.json').read_text()) for r in (0,1)]
        receipt['passed']=all(r['passed'] for r in receipt['ranks'])
    finally:
        for p in children:
            if p.poll() is None:p.kill()
            p.wait(timeout=10)
        for f in logs:f.close()
        receipt['returncodes']=[p.returncode for p in children]
        (a.output/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':receipt['passed'],'cases':sum(len(r['cases']) for r in receipt['ranks'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=(0,1));p.add_argument('--port',type=int);p.add_argument('--uuid')
    a=p.parse_args()
    if a.rank is None:
        if not a.profile:p.error('--profile required for parent')
        main(a)
    else:worker(a)
