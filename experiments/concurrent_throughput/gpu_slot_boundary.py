"""Supplemental 512-slot dense-oracle boundary; exclusive GPUs, no model server."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from gpu_oracles import oracle_code
from prepare_row16 import verify

def main(a):
    verify(a.profile/'engine',a.profile/'engine.sha256')
    so,=(a.profile/'native').glob('exllamav3_ext*.so')
    digest=hashlib.sha256(so.read_bytes()).hexdigest()
    assert digest==json.loads((a.profile/'build-receipt.json').read_text())['native_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    sys.path[:0]=[str(a.profile/'native'),str(a.profile/'engine'),str(a.profile/'engine/tests')]
    import torch
    import exllamav3_ext as ext
    assert ext.decode_max_rows==64 and torch.cuda.device_count()==2
    path=a.profile/'engine/tests/kernels/moe/test_moe_coop.py'
    ns={'__name__':'slot512_dense_oracle','__file__':str(path)}
    exec(oracle_code(path.read_text(),str(path)),ns)
    report={'passed':False,'native_sha256':digest,'source_oracle_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'oracle':'Unchanged upstream 3e-3 dense tolerance; one upstream zero routing weight, all picks local.',
            'cases':[]}
    try:
        for index in (0,1):
            device=torch.device(f'cuda:{index}')
            assert torch.cuda.get_device_capability(device)==(8,0)
            ns['run_case'](device,20261080,3,3,3,2,bsz=64,limit=10.0,gu_f32=True,
                           H=4096,Hi=4096,I=2048,Ho=4096,H_out=4096,
                           E=8,topk=8,min_e=-1,max_e=-1,empty_rows=False)
            report['cases'].append({'device':index,'rows':64,'slots':512,'all_picks_local':True,'passed':True})
        report['passed']=True
    finally:a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases'])}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
