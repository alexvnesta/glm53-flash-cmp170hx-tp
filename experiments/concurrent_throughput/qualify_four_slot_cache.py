"""Four-slot long-context and cross-process cache recovery qualification.

Synthetic marker/array contract, separate from the frozen throughput benchmark.
Private backends only, in an exclusive window. Never changes hosting.
"""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time

import httpx
import measure
from sustained import check_traffic


def validate_budget(lengths, max_tokens, pool, context):
    if len(lengths)!=4 or any(type(n) is not int or n<=0 for n in lengths):
        raise ValueError('Require four positive input lengths')
    if any(n+max_tokens>context for n in lengths):
        raise ValueError('Input plus output exceeds the per-request context cap')
    reserved=sum(math.ceil((n+max_tokens)/256)*256 for n in lengths)+256
    if reserved>pool:
        raise ValueError('Four active requests exceed the shared GPU page budget')
    return reserved


async def body_for(client,target,label,index):
    expected={'a':f'MAPLE_{101+index}','b':f'OCEAN_{201+index}',
              'c':f'PEBBLE_{301+index}','values':list(range(96))}
    count=target-130
    for _ in range(10):
        text=(f'Four-slot context fixture {label}, independent session {index}. Record A is '+expected['a']+'.\n'
              +' ledger'*(count//2)+'\nRecord B is '+expected['b']+'.\n'
              +' ledger'*(count-count//2)+'\nRecord C is '+expected['c']+'. '
              +'Return only a JSON object with keys a, b, c containing the values of records A, B, C, '
              +'and key values containing all integers 0 through 95 inclusive in ascending order. No markdown.')
        body={'model':'GLM-5.3-Flash','reasoning_effort':'low','temperature':0,
              'max_tokens':2048,'messages':[{'role':'user','content':text}],
              'response_format':{'type':'json_object'}}
        response=await client.post('/tokenize',json=body);response.raise_for_status()
        actual=response.json()['token_length']
        if actual==target:
            return {'id':label+'-'+str(index),'input_tokens':target,'expected':expected,
                    'body':body,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
        count+=target-actual
    raise RuntimeError('Exact serving-tokenizer length construction failed')


async def stream(client,case,owned):
    start=time.perf_counter();content=[];reasoning=[];row={'id':case['id'],
        'input_tokens':case['input_tokens'],'prompt_sha256':case['prompt_sha256'],'ttft_s':None}
    body={**case['body'],'stream':True,'stream_options':{'include_usage':True}}
    usage=timings=finish=None
    async with client.stream('POST','/v1/chat/completions',json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: {'):continue
            event=json.loads(line[6:])
            if event.get('error'):raise RuntimeError(str(event['error']))
            if event.get('id'):owned.add(event['id']);row['request_id']=event['id']
            usage=event.get('usage') or usage;timings=event.get('timings') or timings
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{});text=delta.get('content') or '';thought=delta.get('reasoning_content') or ''
                if (text or thought) and row['ttft_s'] is None:row['ttft_s']=time.perf_counter()-start
                content.append(text);reasoning.append(thought);finish=choice.get('finish_reason') or finish
    answer=''.join(content)
    try:correct=finish=='stop' and json.loads(answer)==case['expected']
    except ValueError:correct=False
    row.update(answer=answer,correct=correct,finish_reason=finish,usage=usage,timings=timings,
               finish_s=time.perf_counter()-start,reasoning_characters=sum(map(len,reasoning)))
    if not usage or usage['prompt_tokens']!=case['input_tokens']:
        raise measure.ProbeResponseFailure('Long-context final usage missing or different token length',row)
    if not correct:raise measure.ProbeResponseFailure('Long-context known-answer failure',row)
    return row


async def wave(client,cases):
    before=await measure.health(client);measure.require_idle(before,4)
    identity=measure.identity(before);baseline={r['id'] for r in before['request_metrics']['recent']}
    owned=set();samples=[];start=time.perf_counter()
    async def watch():
        while True:
            h=await measure.health(client)
            check_traffic(h,identity,baseline,owned,4,before['request_queue']['admitted'])
            samples.append({'seconds':time.perf_counter()-start,'active':h['active_sequences'],
                'pending':h['pending_sequences'],'ring':h['concurrent_ring'],
                'phase_progress':[{k:r.get(k) for k in ('id','phase','prefilled_tokens','generated_tokens','progress_age_ms')}
                                  for r in h['request_metrics']['active']]})
            await asyncio.sleep(1)
    workers=[asyncio.create_task(stream(client,c,owned)) for c in cases]
    async def collect():return await asyncio.gather(*workers)
    work=asyncio.create_task(collect())
    monitor=asyncio.create_task(watch())
    try:
        done,_=await asyncio.wait((work,monitor),timeout=1800,return_when=asyncio.FIRST_COMPLETED)
        if monitor in done:await monitor
        if work not in done:raise TimeoutError('Four long-context requests exceeded 1800 seconds')
        rows=await work;wall=time.perf_counter()-start
    finally:
        for task in workers+[work,monitor]:task.cancel()
        await asyncio.gather(*workers,work,monitor,return_exceptions=True)
    for _ in range(100):
        after=await measure.health(client)
        if not after['busy'] and not after['request_metrics']['active']:break
        await asyncio.sleep(.2)
    measure.require_idle(after,4)
    assert measure.identity(after)==identity
    assert after['request_queue']['admitted']==before['request_queue']['admitted']+4
    assert after['concurrent_ring']['peak_ring_leases']==4
    assert len(owned)==4
    recent=[r for r in after['request_metrics']['recent'] if r['id'] in owned]
    assert len(recent)==4 and all(r['outcome']=='completed' for r in recent)
    return {'passed':True,'wall_s':wall,'requests':rows,'server_metrics':recent,'samples':samples,
            'before':before,'after':after}


async def flush(client):
    stable=0;previous=None
    for _ in range(300):
        h=await measure.health(client);measure.require_idle(h,4)
        d=h['persistent_cache']
        assert not d['writes_disabled'] and not d['metrics']['write_errors']
        now=(d['pages'],d['checkpoints'],d['pending_pages'],d['pending_checkpoints'])
        stable=stable+1 if now==previous and not d['pending_pages'] and not d['pending_checkpoints'] else 0
        if stable>=3:return h
        previous=now;await asyncio.sleep(1)
    raise RuntimeError('Persistent writes did not drain within 300 seconds')


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'mode':args.mode,'waves':[]}
    path=args.output/'summary.json'
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    async with httpx.AsyncClient(base_url=args.base_url,timeout=httpx.Timeout(1800,connect=3),
                                 limits=httpx.Limits(max_connections=12)) as client:
        try:
            h=await measure.health(client);measure.require_idle(h,4)
            report['identity']=measure.identity(h);report['initial_health']=h
            if args.mode=='seed':
                sets=[('balanced-128k',[131072]*4),('mixed-near-cap',[389120,65536,65536,65536])]
                for label,lengths in sets:
                    validate_budget(lengths,2048,h['cache_tokens'],h['context_length'])
                    cases=[await body_for(client,n,label,i) for i,n in enumerate(lengths)]
                    (args.output/(label+'-cases.json')).write_text(json.dumps(cases)+'\n')
                    result=await wave(client,cases);report['waves'].append({'label':label,**result})
                    save();await flush(client)
                report['final_health']=await flush(client)
            else:
                seed=json.loads((args.seed/'summary.json').read_text());assert seed['passed']
                assert measure.identity(h)==seed['identity']
                assert h['persistent_cache']['fingerprint']==seed['final_health']['persistent_cache']['fingerprint']
                assert h['request_queue']['admitted']==0,'Require a fresh backend process'
                cases=json.loads((args.seed/'mixed-near-cap-cases.json').read_text())
                result=await wave(client,cases);report['waves'].append({'label':'mixed-after-restart',**result});save()
                for row in result['requests']:
                    cached=(row.get('timings') or {}).get('cache_n',0)
                    assert cached>=row['input_tokens']-8192,('Deep restart resume missing',row['id'],cached)
                after=await flush(client);report['final_health']=after
                assert after['persistent_cache']['metrics']['page_reads']>0
                assert after['persistent_cache']['metrics']['state_reads']>=4
            report['passed']=True
        except BaseException as error:
            report['failure']=measure.failure_details(error);raise
        finally:save()
    print(json.dumps({'passed':True,'mode':args.mode,
                      'input_tokens':[[r['input_tokens'] for r in w['requests']] for w in report['waves']],
                      'wall_seconds':[w['wall_s'] for w in report['waves']]}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True)
    p.add_argument('--mode',choices=('seed','restore'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=Path)
    args=p.parse_args()
    if args.mode=='restore' and args.seed is None:p.error('Restore requires the seed receipt directory')
    asyncio.run(run(args))
