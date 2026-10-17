"""Prepare an isolated active-KV capacity experiment. Never changes hosting.

The mapped variant keeps Q8 latent payload/scales in registered system RAM;
RoPE and sparse index planes remain on GPU. This is static active backing,
not a pressure-driven hot-page cache. Requires the existing pinned_cuda_view
native API and retains the source arena for every alias's lifetime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare_server import configuration_environment
from prepare_row16 import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapped_source(text):
    start = text.index('class CacheLayer_MLA_quant(')
    prefix, body = text[:start], text[start:]
    old = '''        self.qk = torch.zeros(self.qshape, dtype = torch.int, device = device) if self.qshape else None
        self.sk = torch.zeros(self.sshape, dtype = torch.half, device = device) if self.sshape else None'''
    new = '''        # Experimental active backing. The GPU aliases do not own their storage.
        # Keep the exact registered arena alive until all cache views are released.
        self._active_host_arena = None
        if (torch.device("cuda", device) if isinstance(device, int) else torch.device(device)).type != "cuda":
            # TP first loads/export modules on CPU. Preserve that metadata and
            # export pass; only real CUDA worker allocations become aliases.
            self.qk = torch.zeros(self.qshape, dtype = torch.int, device = device) if self.qshape else None
            self.sk = torch.zeros(self.sshape, dtype = torch.half, device = device) if self.sshape else None
        elif self.qshape:
            from ..util.pinned_arena import PinnedArena
            from ..ext import exllamav3_ext as ext
            qbytes = int(np.prod(self.qshape)) * torch.int.itemsize
            sbytes = int(np.prod(self.sshape)) * torch.half.itemsize
            arena = PinnedArena(qbytes + sbytes, "experimental active MLA latent", mapped = True, required = True)
            self._active_host_arena = arena
            arena.tensor.zero_()
            dev = device if isinstance(device, int) else torch.device(device).index
            if dev is None:
                dev = torch.cuda.current_device()
            self.qk = ext.pinned_cuda_view(arena.tensor[:qbytes].view(torch.int).view(self.qshape), dev)
            self.sk = ext.pinned_cuda_view(arena.tensor[qbytes:qbytes + sbytes].view(torch.half).view(self.sshape), dev)
            print(f"ACTIVE_KV_HOST layer={getattr(getattr(self, 'attention', None), 'layer_idx', '?')} device={dev} bytes={arena.size}", flush = True)
        else:
            self.qk = self.sk = None'''
    if body.count(old) != 1:
        raise ValueError('Unexpected quantized MLA allocation source')
    body = body.replace(old, new)
    old = '        self._scratch = {}\n\n\n    def get_qc(self):'
    new = '''        self._scratch = {}
        arena = getattr(self, "_active_host_arena", None)
        self._active_host_arena = None
        if arena is not None:
            arena.close()


    def get_qc(self):'''
    if body.count(old) != 1:
        raise ValueError('Unexpected quantized MLA free source')
    return prefix + body.replace(old, new)


def prepare(base, candidate, pid, mapped, pool):
    if type(mapped) is not bool or type(pool) is not int or not 655360 <= pool <= 1187840 or pool % 256:
        raise ValueError('Require an explicit mapped mode and a page-aligned pool within 655360..1187840')
    if type(pid) is not int or pid <= 1:
        raise ValueError('Require a real production PID')
    if candidate.exists() or candidate == base or base in candidate.parents:
        raise ValueError('Require a new, separate candidate directory')
    verify(base / 'engine', base / 'engine.sha256')
    manifest = json.loads((base / 'service-sha256.json').read_text())
    for rel, digest in manifest.items():
        if sha(base / 'service' / rel) != digest:
            raise ValueError('Service source mismatch: ' + rel)
    argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if str(base / 'service/glm/glm_api.py') not in argv:
        raise ValueError('Production PID identity differs')
    required={'-cq':'8','--concurrent-max-sequences':'4','--context-cap':'392960'}
    if '-tp' not in argv or any(k not in argv or argv[argv.index(k)+1]!=v for k,v in required.items()):
        raise ValueError('This prototype requires the four-slot TP Q8 profile and its qualified per-request cap')
    config=json.loads((Path(argv[argv.index('-m')+1])/'config.json').read_text())
    config=config.get('text_config',config)
    if (config.get('kv_lora_rank')!=512 or config.get('qk_rope_head_dim')!=0
            or config.get('layer_types',[]).count('deepseek_sparse_attention')!=11
            or config.get('index_kpool')!=4 or config.get('index_head_dim')!=128
            or config.get('index_kpool_compress') is not True):
        raise ValueError('This prototype accounts for eleven NoPE MLA layers with a 512-wide latent')
    raw = (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
    env = configuration_environment(raw, base, candidate)
    if env.get('EXL3_MLA_KIDX_RING')!='32' or env.get('GLM53_TP_ACTIVE_HOST','0') not in ('','0'):
        raise ValueError('Require the 32-row index ring and no legacy active-host migration')
    candidate.mkdir()
    for name in ('engine', 'service', 'native', 'triton-cache'):
        shutil.copytree(base / name, candidate / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    changes = {}
    if mapped:
        path = candidate / 'engine/exllamav3/cache/mla.py'
        before = sha(path); path.write_text(mapped_source(path.read_text()))
        changes['exllamav3/cache/mla.py'] = {'before': before, 'after': sha(path)}
        # Placement accounting must count GPU storage, while storage_size and
        # get_tensors continue to describe all logical bytes for cache transfers.
        path = candidate / 'engine/exllamav3/modules/mla_attn.py'
        old = '            storage += cl.storage_size()'
        new = '''            storage += cl.storage_size()
            if hasattr(cl, "qshape") and cl.qshape:
                storage -= int(math.prod(cl.qshape)) * 4 + int(math.prod(cl.sshape)) * 2'''
        text = path.read_text()
        if text.count(old) != 1:
            raise ValueError('Unexpected MLA placement source')
        before = sha(path); path.write_text(text.replace('import torch\n', 'import torch\nimport math\n', 1).replace(old, new))
        changes['exllamav3/modules/mla_attn.py'] = {'before': before, 'after': sha(path)}
    for path in (candidate / 'service').rglob('*.py'):
        text = path.read_text()
        for change in changes.values():
            text = text.replace(change['before'], change['after'])
        if path.name == 'glm_api.py':
            marker = '            "cache_tokens": runtime["args"].cache_size,'
            if text.count(marker) != 1:
                raise ValueError('Unexpected health source')
            text = text.replace(marker, marker + '\n            "experimental_active_kv": ' + repr({
                'backing': 'mapped_host_latent' if mapped else 'gpu',
                'pressure_driven': False, 'host_latent_bytes_per_token': 544 * 11 if mapped else 0,
                'pool_tokens': pool, 'production_qualified': False}) + ',')
        if path.name == 'glm_b2_options.py':
            # This ceiling deliberately preserves the original qualified memory
            # footprint. Only this copied experiment may attempt a larger pool;
            # per-request staging and speculative-ring checks stay in place.
            old = 'size>ring_pool_limit(ring)'
            if text.count(old) != 1:
                raise ValueError('Unexpected pool qualification guard')
            text = text.replace(old, 'size>max(ring_pool_limit(ring), '+str(pool)+')')
        path.write_text(text)
    argv = [v.replace(str(base), str(candidate)) for v in argv]
    argv[argv.index('--port') + 1] = '8013'
    argv[argv.index('-cs') + 1] = str(pool)
    if '--disk_cache_dir' in argv:
        argv[argv.index('--disk_cache_dir') + 1] = str(candidate / 'disk-cache')
    files = sorted(p for p in (candidate / 'engine').rglob('*') if p.is_file())
    (candidate / 'engine.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(candidate / 'engine'))+'\n' for p in files))
    manifest = {str(p.relative_to(candidate / 'service')): sha(p) for p in (candidate / 'service').rglob('*.py')}
    (candidate / 'service-sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    native, = (candidate / 'native').glob('exllamav3_ext*.so')
    if env.get('GLM53_NATIVE_EXT_SHA256')!=sha(native):
        raise ValueError('Copied native binary differs from the production process pin')
    plan = {'status':'prepared_not_launched', 'production_pid':pid, 'argv':argv,
            'env':env, 'cwd':str(candidate), 'native_sha256':sha(native),
            'engine_changes':changes, 'mapped':mapped, 'pool_tokens':pool,
            'qualification':'Experimental only; all GPU qualification pending'}
    (candidate / 'launch-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (candidate / 'launch-plan.json').chmod(0o600)
    print(json.dumps({k:plan[k] for k in ('status','mapped','pool_tokens','engine_changes')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--mapped',action='store_true')
    p.add_argument('--pool',type=int,default=1064960)
    a=p.parse_args()
    if not 655360 <= a.pool <= 1187840 or a.pool % 256:
        p.error('Require a page-aligned experimental pool within 655360..1187840')
    prepare(a.base.resolve(),a.candidate.resolve(),a.pid,a.mapped,a.pool)
