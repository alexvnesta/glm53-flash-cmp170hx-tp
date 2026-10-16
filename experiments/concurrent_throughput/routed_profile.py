"""Exclusive synthetic hot-kernel capture for Nsight Compute.

Profiles one warmed row32, 3-bit/mul1 routed launch on GPU0. Random EXL3
projections and 146 local / 288 global experts approximate a production shard;
this is diagnostic hardware-counter evidence, not serving tok/s or actual
model routing. Run the real serving fixture separately without a profiler.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from gpu_oracles import oracle_code
from prepare_row16 import verify

def main(a):
    verify(a.profile/'engine',a.profile/'engine.sha256')
    so,=(a.profile/'native').glob('exllamav3_ext*.so');digest=hashlib.sha256(so.read_bytes()).hexdigest()
    assert digest==json.loads((a.profile/'build-receipt.json').read_text())['native_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Exclusive GPU window required'
    sys.path[:0]=[str(a.profile/'native'),str(a.profile/'engine'),str(a.profile/'engine/tests')]
    import torch
    import exllamav3_ext as ext
    assert torch.cuda.get_device_capability(0)==(8,0) and ext.decode_max_rows==32
    path=a.profile/'engine/tests/kernels/moe/test_moe_coop.py';ns={'__name__':'routed_profile','__file__':str(path)}
    exec(oracle_code(path.read_text(),str(path)),ns);Proj=ns['Proj'];gen=torch.Generator().manual_seed(20261016)
    device=0;H,I,capacity,rows,experts,topk=4096,2048,256,32,146,8
    gate,up,down=[Proj(k,n,3,2,False,gen,experts,device) for k,n in ((H,I),(H,I),(I,H))]
    had_g,had_u=[torch.empty((capacity,H),dtype=torch.half,device=device) for _ in range(2)]
    gu_g,gu_u=[torch.empty((capacity,1,I),dtype=torch.float,device=device) for _ in range(2)]
    act=torch.empty((capacity,1,I),dtype=torch.half,device=device);d=torch.empty((capacity,1,H),dtype=torch.float,device=device)
    out=torch.empty((rows,H),dtype=torch.float,device=device)
    ctr=torch.zeros(capacity*(I//128)+rows*(H//128)+2*capacity+3,dtype=torch.int,device=device)
    x=(torch.randn((rows,H),generator=gen)*.05).half().to(device)
    sel=torch.stack([torch.randperm(288,generator=gen)[:topk] for _ in range(rows)]).to(device)
    rw=torch.rand((rows,topk),generator=gen).half().to(device)
    def launch():
        ext.exl3_moe_coop(x,sel,rw,0,146,H,*gate.tabs,*up.tabs,*down.tabs,None,None,None,
            3,3,3,False,True,0,10.,True,had_g,had_u,gu_g,gu_u,act,d,ctr,out,None,None)
    for _ in range(20):launch()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range('routed_sample'):launch()
    torch.cuda.synchronize();assert torch.isfinite(out).all()
    print(json.dumps({'passed':True,'native_sha256':digest,'rows':rows,'local_experts':experts,'global_experts':288,'device':device,'selected_local_slots':int(((sel>=0)&(sel<146)).sum()),'units':'Hardware counters only; no server token rate'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);main(p.parse_args())
