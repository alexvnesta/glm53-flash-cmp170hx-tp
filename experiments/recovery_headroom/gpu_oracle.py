"""Private, small KDA layer graph/eager comparison. Requires exclusive GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def child(root, output):
    sys.path[:0] = [str(root/'native'), str(root/'engine'), str(root/'engine/tests')]
    import torch
    import exllamav3_ext as ext
    from testlib.exl3 import rand_trellis, rand_scale
    results = {}
    with torch.inference_mode():
        for dev in (0, 1):
            torch.cuda.set_device(dev)
            assert torch.cuda.get_device_capability(dev) == (8, 0)
            d = torch.device('cuda', dev)
            gen = torch.Generator().manual_seed(20261019)
            def rand(shape, dtype=torch.half):
                return (torch.randn(shape, generator=gen)*0.05).to(device=d, dtype=dtype)
            H, nk, nv, hk, hv = 256, 2, 2, 128, 128
            F = 2*nk*hk+nv*hv
            def linear(k, n):
                return ext.BC_LinearEXL3(rand_trellis(k,n,3,gen,d), rand_scale(k,gen,d),
                    rand_scale(n,gen,d), 3., None, False, True,
                    torch.empty((8,16,k),device=d,dtype=torch.half))
            qkv, out = linear(H,F), linear(nv*hv,H)
            norm = ext.BC_GatedRMSNorm(torch.ones(hv,device=d,dtype=torch.bfloat16),1e-6,0.,1,False,1)
            bc = ext.BC_GatedDeltaNetSplit(qkv,out,rand((nv,H)),rand((hk,H)),rand((nv*hk,hk)),
                rand((hv,H)),rand((nv*hv,hv)),rand((nv*hk,),torch.bfloat16),
                rand((nv,),torch.float),0.,nk,nv,hk,hv,rand((F,4),torch.bfloat16),None,norm,1.)
            for B,S,hist in [(1,1,False),(1,7,True),(2,7,True),(4,8,True),(4,16,True),(3,15,True),(2,1,False)]:
                def empty(shape,dtype=torch.float): return torch.empty(shape,device=d,dtype=dtype)
                bc.configure_slot_kda(B,S,hist,empty((B,S,F)),empty((B,S,nv,hv)),empty((B,S,nv)),
                    empty((B,S,hk)),empty((B,S,nv*hk)),empty((B,S,hv)),empty((B,S,nv),torch.bfloat16),
                    empty((B,S,nv,hk)),empty((B,F,S),torch.bfloat16),empty((B,S,F),torch.bfloat16),
                    empty((B,S,nv,hv),torch.bfloat16),empty((B,S,nv*hv),torch.half),
                    empty((B,S,H),torch.half),empty((B,S,nv*hv),torch.half))
                conv0=rand((B,F,4+16),torch.bfloat16)
                state0=rand((2*B,17,nv,hk,hv),torch.half)
                slots=torch.arange(B,device=d,dtype=torch.int32)
                scan=slots*2+1 if hist else slots*2
                source=slots*2 if hist else None
                x0=rand((B,S,H))
                runs=[]
                # First use eager; second capture or refusal; later calls replay or remain eager.
                for rep in range(4):
                    x=x0.clone();y=empty((B,S,H));conv=conv0.clone();state=state0.clone()
                    bc.run_bszN(x,y,conv,state,slots,scan,hist,source)
                    torch.cuda.synchronize(dev)
                    assert torch.isfinite(y).all() and torch.isfinite(state).all()
                    runs.append([y.cpu(),conv.cpu(),state.cpu()])
                for run in runs[1:]:
                    for actual,expected in zip(run,runs[0]):
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                results[f'{dev}:{B}:{S}:{hist}']=runs[-1]
    torch.save(results,output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true')
    a=p.parse_args();root=a.candidate.resolve()
    if a.child: return child(root,a.output)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU owner is present')
    binary,=(root/'native').glob('exllamav3_ext*.so')
    sha=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert sha==json.loads((root/'build-receipt.json').read_text())['native_sha256']
    # Verify the complete prepared source manifest before importing native code.
    for line in (root/'engine.sha256').read_text().splitlines():
        digest,relative=line.split('  ',1)
        assert hashlib.sha256((root/'engine'/relative).read_bytes()).hexdigest()==digest,relative
    a.output.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'native_sha256':sha,'scope':'Small synthetic KDA graph/eager; no full-model fit claim'}
    try:
        for mode,floor in [('graphs','0'),('forced-eager','1048576')]:
            env=os.environ.copy();env['EXL3_KDA_GRAPH_MIN_FREE_MB']=floor
            env['PYTHONPATH']=str(root/'native')+':'+str(root/'engine')
            with (a.output/(mode+'.log')).open('w') as log:
                subprocess.run([sys.executable,__file__,'--child','--candidate',str(root),'--output',str(a.output/(mode+'.pt'))],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
        import torch
        graph=torch.load(a.output/'graphs.pt',weights_only=True)
        eager=torch.load(a.output/'forced-eager.pt',weights_only=True)
        assert graph.keys()==eager.keys()
        for key in graph:
            for x,y in zip(graph[key],eager[key]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        messages=(a.output/'forced-eager.log').read_text().count('new graph slot remains eager')
        assert messages==len(graph), (messages,len(graph))
        assert 'new graph slot remains eager' not in (a.output/'graphs.log').read_text()
        report.update(passed=True,cases=len(graph),within_process_runs=4,comparison='bit exact output, conv state and recurrent state',guard_refusals=messages)
    finally:
        (a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
