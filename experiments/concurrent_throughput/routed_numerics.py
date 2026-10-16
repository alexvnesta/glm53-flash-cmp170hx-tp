"""Full-width dense-oracle checks and bit hashes for baseline/candidate comparison.

Uses upstream's unchanged 3e-3 tolerance and bit reproducibility checks. Adding
an AST return exposes the already-checked output without changing arithmetic.
Compare the JSON output hashes across binaries; no response repair is involved.
"""
import argparse,ast,hashlib,json,subprocess,sys
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
    path=a.profile/'engine/tests/kernels/moe/test_moe_coop.py'
    ns={'__name__':'routed_dense_oracle','__file__':str(path)}
    exec(oracle_code(path.read_text(),str(path)),ns)
    # Preserve the original oracle, then get its checked output from the same code object.
    from gpu_oracles import expand_oracle
    tree=ast.parse(expand_oracle(path.read_text()))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    fn.body.append(ast.Return(value=ast.Name(id='o1',ctx=ast.Load())))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),ns)
    receipt={'passed':False,'native_sha256':digest,'source_oracle_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':[]}
    try:
        for device in (0,1):
            assert torch.cuda.get_device_capability(device)==(8,0)
            for rows in (1,2,4,7,8,9,16,17,24,31,32)+((33,48,63,64) if ext.decode_max_rows==64 else ()):
                out=ns['run_case'](torch.device(f'cuda:{device}'),20261016+rows,3,3,3,2,
                    bsz=rows,H=4096,Hi=4096,I=2048,Ho=4096,H_out=4096,E=8,topk=8,
                    min_e=1,max_e=9,empty_rows=True,limit=10.,gu_f32=True)
                raw=out.cpu().numpy().tobytes()
                receipt['cases'].append({'device':device,'rows':rows,'output_sha256':hashlib.sha256(raw).hexdigest(),'passed':True})
        receipt['passed']=True
    finally:a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':receipt['passed'],'cases':len(receipt['cases'])}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
