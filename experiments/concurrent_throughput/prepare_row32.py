"""Expand the attested row16 variant to cover two hybrid-15 verifiers.

Only the native capacity header changes. Python already reads the capacity
exported by the extension. Requires a native rebuild and exclusive GPU tests.
"""
import argparse
import json
from pathlib import Path
import shutil

from prepare_row16 import sha, verify


def prepare(engine, output, manifest):
    engine = engine.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output == engine or engine in output.parents:
        raise ValueError('Destination must be new and outside the qualified engine')
    expected = verify(engine, manifest)
    rel = 'exllamav3/exllamav3_ext/libtorch/mlp.h'
    old, new = '#define MAX_BSZN 16 ', '#define MAX_BSZN 32 '
    if (engine/rel).read_text().count(old) != 1:
        raise ValueError('Expected the attested row16 capacity header')
    for name in ('mlp', 'block_sparse_mlp'):
        if (engine/f'exllamav3/modules/{name}.py').read_text().count('MAX_BSZN = ext.decode_max_rows  #') != 1:
            raise ValueError('Python must read native decode_max_rows')
    if (engine/'exllamav3/exllamav3_ext/bindings.cpp').read_text().count('m.attr("decode_max_rows") = MAX_BSZN;') != 1:
        raise ValueError('Native must export decode_max_rows')
    output.mkdir(parents=True)
    target = output/'engine'
    shutil.copytree(engine, target, ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    (target/rel).write_text((target/rel).read_text().replace(old,new))
    hashes = {name:sha(target/name) for name in expected}
    (output/'engine.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(hashes.items())))
    receipt = {'status':'unqualified_candidate','source_files':len(hashes),
        'base_manifest_sha256':sha(manifest),'changed_sources':{rel:{'before':expected[rel],'after':hashes[rel]}},
        'decode_max_rows':32,'native_rebuild_required':True,'gpu_tests_run':False,'serving_changed':False,
        'hypothesis':'Keep two hybrid-15 requests (32 verification rows) on the fused small-row path'}
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.engine,args.output,args.manifest)))
