"""Prepare a separately attested opt-in bulk-NCCL broadcast variant.

No arithmetic, native binary, launcher or source pins are changed in place.
The variant needs a qualified row32 binary and independent GPU qualification.
"""
import argparse
import json
from pathlib import Path
import shutil

from prepare_row16 import sha, verify

REL = 'exllamav3/model/model_tp_backend.py'
OLD = '''    def broadcast(self, tensor: torch.Tensor, src_device: int):
        self.fallback.broadcast(tensor, src_device)
        # src_rank = self.active_devices.index(src_device)
        # dist.broadcast(tensor, src = src_rank)
'''
NEW = '''    def broadcast(self, tensor: torch.Tensor, src_device: int):
        # Keep the native low-latency path for payloads up to its 2 KiB boundary.
        # Large payloads can opt into the existing NCCL GPU process group.
        if _nccl_broadcast == "bulk" and tensor.numel() * tensor.element_size() > 2048:
            src_rank = self.active_devices.index(src_device)
            dist.broadcast(tensor, src = src_rank, async_op = False)
        else:
            self.fallback.broadcast(tensor, src_device)
'''
FLAG_ANCHOR = '_nccl_fp32 = os.environ.get("EXL3_TP_NCCL_FP32", "0") != "0"'
FLAGS = '''
_nccl_broadcast = os.environ.get("EXL3_TP_NCCL_BROADCAST", "native")
if _nccl_broadcast not in ("native", "bulk"):
    raise ValueError("EXL3_TP_NCCL_BROADCAST must be native or bulk")'''


def patched(source):
    if source.count(OLD)!=1 or source.count(FLAG_ANCHOR)!=1:
        raise ValueError('NCCL backend anchors changed')
    return source.replace(OLD,NEW).replace(FLAG_ANCHOR,FLAG_ANCHOR+FLAGS)


def prepare(engine,manifest,output):
    engine=engine.resolve(strict=True);output=output.resolve()
    if output.exists() or output==engine or engine in output.parents:
        raise ValueError('Destination must be new and outside the qualified engine')
    expected=verify(engine,manifest)
    source=patched((engine/REL).read_text())
    output.mkdir(parents=True)
    target=output/'engine'
    shutil.copytree(engine,target,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    (target/REL).write_text(source)
    hashes={name:sha(target/name) for name in expected}
    (output/'engine.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(hashes.items())))
    result={'status':'unqualified_candidate','source_files':len(hashes),'native_rebuild_required':False,
        'base_manifest_sha256':sha(manifest),'changed_sources':{REL:{'before':expected[REL],'after':hashes[REL]}},
        'default_broadcast':'native','opt_in':{'EXL3_TP_NCCL_BROADCAST':'bulk'},'bulk_threshold_bytes':2048,
        'gpu_tests_run':False,'serving_changed':False}
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.engine,a.manifest,a.output)))
