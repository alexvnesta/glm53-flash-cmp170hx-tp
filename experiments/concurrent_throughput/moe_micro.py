"""Exclusive-GPU screening of existing cooperative-MoE geometry controls.

CUDA graph replay timings are kernel screening data, not server token rates.
Uses upstream random EXL3 projections at GLM dimensions and production scratch
capacity; routed cases include foreign experts as in expert-parallel sharding.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from gpu_oracles import oracle_code
from prepare_row16 import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.profile.resolve(strict=True)
    verify(root/'engine', root/'engine.sha256')
    so, = (root/'native').glob('exllamav3_ext*.so')
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    if digest != json.loads((root/'build-receipt.json').read_text())['native_sha256']:
        raise ValueError('Native identity changed')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Exclusive GPU window required')
    sys.path[:0] = [str(root/'native'),str(root/'engine'),str(root/'engine/tests')]
    import torch
    import exllamav3_ext as ext
    assert ext.decode_max_rows==32 and torch.cuda.device_count()==2
    source = root/'engine/tests/kernels/moe/test_moe_coop.py'
    ns = {'__name__':'moe_screening','__file__':str(source)}
    exec(oracle_code(source.read_text(), str(source)),ns)
    Proj = ns['Proj']
    report = {'passed':False,'native_sha256':digest,'oracle_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'flags':{k:os.environ.get(k) for k in ('EXL3_MOE_COOP_WIDE','EXL3_MOE_COOP_KSPLIT')},'cases':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        for device in (0,1):
            torch.cuda.set_device(device)
            for kind,bits,experts,topk,capacity in (('routed',3,32,8,256),('shared',5,1,1,32)):
                gen = torch.Generator().manual_seed(20261016)
                H,I=4096,2048
                gate,up,down = [Proj(k,n,bits,2,False,gen,experts,device) for k,n in ((H,I),(H,I),(I,H))]
                had_g,had_u = [torch.empty((capacity,H),dtype=torch.half,device=device) for _ in range(2)]
                gu_g,gu_u = [torch.empty((capacity,1,I),dtype=torch.float,device=device) for _ in range(2)]
                act = torch.empty((capacity,1,I),dtype=torch.half,device=device)
                d = torch.empty((capacity,1,H),dtype=torch.float,device=device)
                out = torch.empty((32,H),dtype=torch.float,device=device)
                ctr = torch.zeros(capacity*(I//128)+32*(H//128)+2*capacity+3,dtype=torch.int,device=device)
                for rows in (8,16,32):
                    x = (torch.randn((rows,H),generator=gen)*.05).half().to(device)
                    global_e = experts*2 if kind=='routed' else 1
                    sel = torch.stack([torch.randperm(global_e,generator=gen)[:topk] for _ in range(rows)]).to(device)
                    rw = torch.rand((rows,topk),generator=gen).half().to(device)
                    minimum,maximum = (experts//2,experts//2+experts) if kind=='routed' else (-1,-1)
                    def launch():
                        ext.exl3_moe_coop(x,sel,rw,minimum,maximum,H,*gate.tabs,*up.tabs,*down.tabs,
                            None,None,None,bits,bits,bits,False,True,0,10.,True,
                            had_g,had_u,gu_g,gu_u,act,d,ctr,out,None,None)
                    launch();torch.cuda.synchronize();first=out[:rows].clone()
                    launch();torch.cuda.synchronize()
                    assert torch.isfinite(first).all() and torch.equal(first,out[:rows])
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        for _ in range(5):launch()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):launch()
                    times=[]
                    for _ in range(5):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        for _ in range(100):graph.replay()
                        end.record();end.synchronize();times.append(start.elapsed_time(end)/100)
                    row={'device':device,'kind':kind,'bits':bits,'rows':rows,'local_experts':experts,
                         'topk':topk,'scratch_slots':capacity,'median_ms':statistics.median(times),'repeats_ms':times}
                    report['cases'].append(row);print(json.dumps(row),flush=True)
                    del graph
        report['passed']=True
    finally:args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
