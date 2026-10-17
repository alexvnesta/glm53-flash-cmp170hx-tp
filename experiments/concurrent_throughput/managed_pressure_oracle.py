"""Exclusive diagnostic: prove automatic eviction preserves dirty managed data.

Not a throughput benchmark. Temporarily occupies otherwise free VRAM in this
process only, leaves 768 MiB free, and revisits 1.5 GiB of demand-paged data.
Never run alongside a model server or any other GPU client.
"""
import argparse
import gc
import json
from pathlib import Path
import torch
from exllamav3.util.managed_kv import ManagedKV


@torch.inference_mode()
def run(output):
    rows=[]
    for dev in (0,1):
        torch.cuda.set_device(dev)
        torch.cuda.empty_cache()
        free,total=torch.cuda.mem_get_info(dev)
        leave=768*1024**2
        if free < 4*1024**3:
            raise RuntimeError('Insufficient idle VRAM for isolated eviction oracle')
        reserve=torch.empty((free-leave,),dtype=torch.uint8,device=dev)
        owner=ManagedKV((1536*1024**2//4,),(1,),dev)
        data=owner.q
        data.fill_(7)
        chunks=data.chunk(4)
        for i,part in enumerate(chunks):
            part.add_(i+1)
        checked=[]
        # Revisit in reverse then forward order to exercise evicted dirty pages.
        for i in list(reversed(range(4)))+list(range(4)):
            bounds=chunks[i].aminmax()
            lo,hi=bounds.min.item(),bounds.max.item()
            assert lo==8+i and hi==8+i,(dev,i,lo,hi)
            checked.append({'chunk':i,'min':lo,'max':hi,'elements':chunks[i].numel()})
        torch.cuda.synchronize(dev)
        now=torch.cuda.mem_get_info(dev)[0]
        rows.append({'device':dev,'correct':True,'free_before':free,'free_after':now,
                     'ordinary_reservation_bytes':reserve.numel(),'managed_bytes':owner.bytes,
                     'desired_free_headroom_bytes':leave,'validated_chunk_bounds':checked,
                     'prefetch_or_explicit_eviction':False})
        del part,chunks,data,reserve
        owner.close();del owner
        gc.collect();torch.cuda.empty_cache()
    report={'passed':True,'cases':rows,'timing_is_not_a_model_benchmark':True}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
