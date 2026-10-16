"""Prepare an unqualified 64-row MoE variant from the attested row32 engine.

Four hybrid-15 verifiers need 64 rows and 512 routed slots at GLM top-k 8.
The slot-sort key needs nine index bits; increasing capacity alone is unsafe.
Never compiles, loads a model, or changes a service.
"""
import argparse
import json
from pathlib import Path
import shutil
from prepare_row16 import sha,verify

PREFIX='exllamav3/exllamav3_ext/'
EDITS={
    PREFIX+'libtorch/mlp.h':{'#define MAX_BSZN 32 ':'#define MAX_BSZN 64 '},
    PREFIX+'quant/exl3_moe_coop.cu':{
        'slots_max <= 256':'slots_max <= 512',
        'scratch must hold 1..256 slots':'scratch must hold 1..512 slots'},
    PREFIX+'quant/exl3_moe_coop_kernel.cuh':{
        'constexpr int MAX_SLOTS = 256;                  // MAX_BSZN * top_k bound (top_k <= 32)':
        'constexpr int MAX_SLOTS = 512;                  // 64 rows * GLM top_k 8; scratch checked by launcher\n'
        'static_assert(MAX_SLOTS <= THREADS, "slot sort requires one thread per slot");',
        '(my_e << 8) | t':'(my_e << 9) | t',
        '(eu << 8) | u':'(eu << 9) | u'},
}

def prepare(engine,output,manifest):
    engine=engine.resolve(strict=True);output=output.resolve()
    if output.exists() or output==engine or engine in output.parents:
        raise ValueError('Destination must be new and outside the qualified engine')
    expected=verify(engine,manifest);changed={}
    for rel,edits in EDITS.items():
        text=(engine/rel).read_text()
        for old,new in edits.items():
            if text.count(old)!=1:raise ValueError('Source anchor changed: '+rel)
            text=text.replace(old,new)
        changed[rel]=text
    for name in ('mlp','block_sparse_mlp'):
        if (engine/f'exllamav3/modules/{name}.py').read_text().count('MAX_BSZN = ext.decode_max_rows  #')!=1:
            raise ValueError('Python must read native capacity')
    if (engine/(PREFIX+'bindings.cpp')).read_text().count('m.attr("decode_max_rows") = MAX_BSZN;')!=1:
        raise ValueError('Native must export decode_max_rows')
    output.mkdir(parents=True)
    target=output/'engine'
    shutil.copytree(engine,target,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    for rel,text in changed.items():(target/rel).write_text(text)
    hashes={name:sha(target/name) for name in expected}
    (output/'engine.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(hashes.items())))
    receipt={'status':'unqualified_candidate','source_files':len(hashes),
        'base_manifest_sha256':sha(manifest),'decode_max_rows':64,'max_routed_slots':512,
        'changed_sources':{rel:{'before':expected[rel],'after':hashes[rel]} for rel in changed},
        'native_rebuild_required':True,'gpu_tests_run':False,'serving_changed':False,
        'hypothesis':'Keep four hybrid-15 verifiers on the cooperative decode path without changing MMA tile geometry'}
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.engine,a.output,a.manifest)))
