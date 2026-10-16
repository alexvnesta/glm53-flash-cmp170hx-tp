"""Clone an attested two-slot profile for a private four-active-sequence trial.

Reuses the exact engine and native extension. A reviewed paired-cache service
overlay is applied identically to both arms. Keeps cache/context/drafting
settings; only allocation parity (-ambs) and active capacity differ between
arms. Does not launch a service. Wider hybrid verification may use the generic path.
"""
import argparse
import json
from pathlib import Path
import shutil

from prepare_row16 import sha, verify
from prepare_server import configuration_environment

SERVICE_BASE_SHA256 = {
    'glm_b2_options.py':'3681cdc3b03476ba0406fe4a1f8838f472b952f5967501b2088fbcf2a3203aef',
    'glm_b2_native.py':'506e67870dede352a98d5ad6f38312c1e4f354255594df10d6f2355ad458f489',
    'glm_b2_cpu_cache.py':'664dcda4792f7245b819a7b9927c74e5d2b52dc7a4834ee575ab8c76b8e70531',
    'glm_b2_shared_prefix.py':'a17760ec191b418fbbc1c1fdb6c3b895632056b244028fddb134b4715d79ff4c',
}


def service_overlay(base, source):
    """Validate the original two-slot guards before copying a reviewed overlay."""
    changes={}
    for name,before in SERVICE_BASE_SHA256.items():
        if sha(base/name)!=before:
            raise ValueError('Paired-cache service source drift: '+name)
        after=sha(source/name)
        if after==before:raise ValueError('Four-slot service extension missing: '+name)
        changes[name]={'before':before,'after':after}
    return changes


def capacity_argv(argv, sequences):
    if type(sequences) is not int or sequences not in (2, 4):
        raise ValueError('Only matched two- and four-slot trials are supported')
    result = list(argv)
    for aliases in (('-ambs', '--autosplit_max_batch_size'), ('--concurrent-max-sequences',)):
        found = [i for i, value in enumerate(result) if value in aliases]
        if len(found) != 1 or result[found[0]+1] != '2':
            raise ValueError('Expected exactly one two-slot allocation/capacity flag')
        result[found[0]+1] = str(sequences)
    return result


def prepare(base, output, pid, sequences, port=8013, service_source=None):
    base = base.resolve(strict=True); output = output.resolve()
    if output.exists() or output == base or base in output.parents:
        raise ValueError('Destination must be new and outside the production profile')
    verify(base/'engine', base/'engine.sha256')
    if sequences==4 and service_source is None:
        raise ValueError('Four slots require the reviewed paired-cache service overlay')
    changes=service_overlay(base/'service/glm',service_source) if service_source else {}
    native, = (base/'native').glob('exllamav3_ext*.so')
    build = json.loads((base/'build-receipt.json').read_text())
    if sha(native) != build['native_sha256']:
        raise ValueError('Native identity mismatch')
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if str(base/'service/glm/glm_api.py') not in argv:
        raise ValueError('PID does not own the attested profile')
    if argv[argv.index('--port')+1] == str(port):
        raise ValueError('Require a different private loopback port')
    argv = capacity_argv(argv, sequences)
    raw = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    env = configuration_environment(raw, base, output)
    output.mkdir(parents=True)
    for name in ('engine', 'service'):
        shutil.copytree(base/name, output/name,
                        ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    for name in changes:
        shutil.copy2(service_source/name,output/'service/glm'/name)
    for name in ('engine.sha256', 'build-receipt.json'):
        shutil.copy2(base/name, output/name)
    (output/'native').mkdir(); shutil.copy2(native, output/'native'/native.name)
    env['GLM_WEIGHTLESS_RECEIPTS'] = str(output/'receipts/numerics')
    if not env.get('PYTHONPATH', '').startswith(str(output/'native')+':'):
        raise ValueError('Native import path does not lead with the isolated clone')
    for name in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
        if name not in env: continue
        dst = Path(env[name]); src = Path(env[name].replace(str(output),str(base)))
        if src == dst or output not in dst.parents:
            raise ValueError('Compilation caches must be isolated')
        if src.exists(): shutil.copytree(src,dst)
    argv = [v.replace(str(base/'service'),str(output/'service')) for v in argv]
    argv[argv.index('--port')+1] = str(port)
    argv[argv.index('--disk_cache_dir')+1] = str(output/'disk-cache')
    receipt = {'status':'prepared_not_launched','production_pid':pid,
        'active_sequences':sequences,'native_sha256':sha(native),
        'engine_manifest_sha256':sha(base/'engine.sha256'),
        'native_rebuild_required':False,'serving_changed':False,'changed_service_sources':changes,
        'argv':argv,'env':env,'cwd':str(output)}
    path = output/'launch-plan.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n'); path.chmod(0o600)
    return {k:v for k,v in receipt.items() if k not in ('argv','env')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--active-sequences',type=int,choices=(2,4),required=True)
    p.add_argument('--service-source',type=Path)
    args=p.parse_args()
    print(json.dumps(prepare(args.base,args.output,args.pid,args.active_sequences,
                             service_source=args.service_source)))
