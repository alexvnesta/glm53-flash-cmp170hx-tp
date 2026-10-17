"""Prepare a private CUDA demand-paged KV candidate; never changes hosting."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from prepare_active_kv import prepare, sha


def managed_source(text):
    start = text.index('class CacheLayer_MLA_quant(')
    prefix, body = text[:start], text[start:]
    begin = body.index('        # Experimental active backing.')
    end = body.index('        self.v = torch.zeros', begin)
    body = body[:begin] + '''        self._active_host_arena = None
        self._active_managed = None
        dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        if dev.type == "cuda" and self.qshape:
            from ..util.managed_kv import ManagedKV
            owner = ManagedKV(self.qshape, self.sshape, dev)
            self._active_managed = owner
            self.qk, self.sk = owner.q, owner.s
            print(f"ACTIVE_KV_MANAGED device={dev} bytes={owner.bytes}", flush=True)
        else:
            self.qk = torch.zeros(self.qshape, dtype=torch.int, device=device) if self.qshape else None
            self.sk = torch.zeros(self.sshape, dtype=torch.half, device=device) if self.sshape else None
''' + body[end:]
    old = '        arena = getattr(self, "_active_host_arena", None)'
    new = '''        owner = getattr(self, "_active_managed", None)
        self._active_managed = None
        if owner is not None:
            owner.close()
        arena = getattr(self, "_active_host_arena", None)'''
    if body.count(old) != 1:
        raise ValueError('Unexpected mapped source free path')
    return prefix + body.replace(old, new)


def dynamic_prepare(base, candidate, pid):
    prepare(base, candidate, pid, True, 1187840)
    plan = json.loads((candidate/'launch-plan.json').read_text())
    source = Path(__file__).parent
    util = candidate/'engine/exllamav3/util'
    shutil.copy2(source/'managed_kv.py', util/'managed_kv.py')
    shutil.copy2(source/'managed_kv_allocator.cpp', util/'managed_kv_allocator.cpp')
    subprocess.run(['c++','-shared','-fPIC','-O2','-I/usr/local/cuda-13.3/include',
                    str(util/'managed_kv_allocator.cpp'),'-L/usr/local/cuda-13.3/lib64',
                    '-lcudart','-o',str(util/'managed_kv_allocator.so')],check=True)
    path = candidate/'engine/exllamav3/cache/mla.py'
    mapped_sha = sha(path)
    path.write_text(managed_source(path.read_text()))
    plan['engine_changes']['exllamav3/cache/mla.py']['after'] = sha(path)
    changes = plan['engine_changes']
    for name in ('managed_kv.py','managed_kv_allocator.cpp','managed_kv_allocator.so'):
        changes['exllamav3/util/'+name] = {'before':None,'after':sha(util/name)}
    for p in (candidate/'service').rglob('*.py'):
        text = p.read_text().replace(mapped_sha, sha(path))
        if p.name == 'glm_api.py':
            text = text.replace("'backing': 'mapped_host_latent'", "'backing': 'cuda_managed_latent'")
            text = text.replace("'pressure_driven': False", "'pressure_driven': True")
        p.write_text(text)
    plan['paging'] = {'mechanism':'cuda_fault_migrate','policy':'driver_managed_no_preferred_location',
                      'host_first_touch':True,'latent_bytes':7108034560,'custom_lru':False}
    plan['qualification'] = 'Experimental demand paging; GPU correctness, migration and throughput unqualified'
    (candidate/'launch-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    files = sorted(p for p in (candidate/'engine').rglob('*') if p.is_file())
    (candidate/'engine.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(candidate/'engine'))+'\n' for p in files))
    manifest = {str(p.relative_to(candidate/'service')):sha(p) for p in (candidate/'service').rglob('*.py')}
    (candidate/'service-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True)
    a=p.parse_args();dynamic_prepare(a.base.resolve(),a.candidate.resolve(),a.pid)
