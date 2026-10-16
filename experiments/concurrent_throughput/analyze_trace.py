"""Summarize a Nsight SQLite export without equating summed GPU time to wall time.

Example: python analyze_trace.py --sqlite steady.sqlite --output analysis.json
Kernels are assigned only when fully contained in an annotated generator range.
"""
import argparse
import collections
import json
import sqlite3
from pathlib import Path


def analyze(path):
    db=sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True)
    ranges=collections.defaultdict(list)
    for start,end,name in db.execute('select start,end,coalesce(text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id order by start'):
        if end and name:ranges[name].append((start,end))
    kernels=db.execute('select start,end,deviceId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by start').fetchall()
    result={'units':'Seconds. Kernel durations sum across GPUs and may overlap; these are not recoverable wall-time savings.','scopes':{},'devices':[]}
    for short in ('iterate_gen','iterate_draftmodel_dflash_gen','prefill'):
        matched=[name for name in ranges if name.endswith('.'+short)]
        if len(matched)!=1:raise ValueError('Expected one annotated generator function: '+short)
        rs=ranges[matched[0]];sums=collections.defaultdict(lambda:[0,0.]);i=0
        generic=set()
        for start,end,device,name in kernels:
            while i<len(rs) and rs[i][1]<start:i+=1
            if i==len(rs):break
            if start>=rs[i][0] and end<=rs[i][1]:
                cat='generic_moe' if 'exl3_moe_kernel' in name else 'coop_moe' if 'exl3_moe_coop' in name else 'nccl' if 'ncclDevKernel' in name else 'native_collectives' if 'pg_' in name else 'other'
                sums[cat][0]+=1;sums[cat][1]+=(end-start)/1e9
                if cat=='generic_moe':generic.add(i)
        result['scopes'][short]={'inclusive_range_s':sum(b-a for a,b in rs)/1e9,'ranges':len(rs),
            'gpu_sums':dict(sums),'generic_ranges':len(generic)}
    for device in sorted({k[2] for k in kernels}):
        rows=[k for k in kernels if k[2]==device];end=-1;busy=0
        for a,b,*_ in rows:busy+=max(0,b-max(a,end));end=max(b,end)
        result['devices'].append({'device':device,'span_s':(max(k[1] for k in rows)-min(k[0] for k in rows))/1e9,'union_busy_s':busy/1e9})
    result['collective_inclusive_ranges']={name:{'calls':len(rows),'seconds':sum(b-a for a,b in rows)/1e9}
        for name,rows in ranges.items() if '.model_tp_backend.' in name}
    db.close();return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sqlite',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.sqlite);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
