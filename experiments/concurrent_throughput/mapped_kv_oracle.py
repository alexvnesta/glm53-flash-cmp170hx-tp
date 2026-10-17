"""GPU oracle for the experimental mapped MLA allocator, separate from serving.

Compare identical Q8 append and sparse attention against GPU-resident copies,
including nonidentity page tables, 1/8/16-row decode and 512-row prefill.
Must run only in an exclusive GPU window.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from exllamav3.cache.mla import CacheLayer_MLA_quant
from exllamav3.modules.attention_fn.mla_triton import mla_kv_quant_append
from exllamav3.modules.attention_fn.dsa_triton import dsa_attn


@torch.inference_mode()
def run(output):
    rows=[]
    for device in (0,1):
        torch.cuda.set_device(device);torch.manual_seed(1729)
        dev=torch.device('cuda',device)
        attn=SimpleNamespace(kv_lora_rank=512,qk_rope_head_dim=0,
                             idx_plane_dim=512,index_kpool=4,index_head_dim=128,layer_idx=99)
        metadata=CacheLayer_MLA_quant(None,attn,0,256,k_bits=8,kidx_ring=32)
        metadata.alloc(torch.device('cpu'))
        assert metadata.qk.device.type=='cpu' and metadata._active_host_arena is None
        metadata.free()
        rows.append({'device':device,'case':'cpu-export-pass','correct':True})
        layer=CacheLayer_MLA_quant(None,attn,0,8192,k_bits=8,kidx_ring=32)
        layer.alloc(dev)
        assert layer._active_host_arena.pinned
        host=layer._active_host_arena
        # Verify that the advertised CUDA pointers address the registered host
        # arena, rather than allocating a second GPU payload.
        from exllamav3.ext import exllamav3_ext as ext
        assert layer.qk.data_ptr()==ext.cuda_host_get_device_pointer(host.ptr)
        qk=layer.qk.clone();sk=layer.sk.clone();rope=layer.v.clone()
        bt=torch.randperm(32,device=dev,dtype=torch.int32).view(1,32)
        x=torch.randn((1,8192,512),device=dev,dtype=torch.half)
        zero=torch.empty((1,8192,0),device=dev,dtype=torch.half)
        seq=torch.zeros(1,device=dev,dtype=torch.int32)
        mla_kv_quant_append(x,zero,layer.qk,layer.sk,layer.v,bt,seq,8)
        mla_kv_quant_append(x,zero,qk,sk,rope,bt,seq,8)
        torch.cuda.synchronize(device)
        assert torch.equal(layer.qk.cpu(),qk.cpu()) and torch.equal(layer.sk.cpu(),sk.cpu())
        rows.append({'device':device,'case':'8192-row-append','bit_identical':True})
        for n in (1,8,16,512):
            q=torch.randn((64,n,512),device=dev,dtype=torch.half)
            pe=torch.empty((n,64,0),device=dev,dtype=torch.half)
            # Independent, sparse selection with invalid padding, including the
            # first/last physical pages through the shuffled logical table.
            indices=torch.randint(8192,(n,2048),device=dev,dtype=torch.int32)
            indices[:,-8:]=-1
            kw=dict(indices=indices,pool_len=8192,q_pos0=8192,
                    q_pe=pe,out_latent=True,n_splits=8 if n<=16 else 1)
            expected=dsa_attn(q,qk,rope,bt,qc=(sk,8),**kw).clone()
            actual=dsa_attn(q,layer.qk,layer.v,bt,qc=(layer.sk,8),**kw).clone()
            torch.cuda.synchronize(device)
            assert torch.equal(actual,expected),(device,n,(actual-expected).abs().max().item())
            # Event timings are kernel microbenchmarks, not model tokens/sec.
            timing={}
            for label,pq,ps,pr in [('gpu',qk,sk,rope),('host',layer.qk,layer.sk,layer.v)]:
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(3):dsa_attn(q,pq,pr,bt,qc=(ps,8),**kw)
                b.record();b.synchronize();timing[label+'_milliseconds']=a.elapsed_time(b)/3
            rows.append({'device':device,'case':'sparse-'+str(n),'bit_identical':True,**timing})
        # CPU -> CUDA-alias -> CPU transfer semantics used by the parked-page
        # tier must hold too. Never assume the fake device tag implies VRAM.
        saved=layer.qk[3].clone().cpu();layer.qk[3].zero_();layer.qk[3].copy_(saved)
        torch.cuda.synchronize(device);assert torch.equal(layer.qk[3].cpu(),saved)
        rows.append({'device':device,'case':'host-page-roundtrip','bit_identical':True})
        layer.free()
    result={'passed':True,'cases':rows,'count':len(rows)}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
