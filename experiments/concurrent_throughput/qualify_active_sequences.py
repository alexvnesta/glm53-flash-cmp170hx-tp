"""Four-slot isolation, mixed grammar, cancellation/survivor/reuse checks.

Run against a private backend in an exclusive window. Uses the original strict
task oracles; no answer repair. This does not change benchmark fixtures.
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx
import measure


async def run(args):
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'waves':[],'lifecycle':None}
    def save(): args.output.write_text(json.dumps(report,indent=2)+'\n')
    async with httpx.AsyncClient(base_url=args.base_url,timeout=120) as client:
        async def idle():
            for _ in range(100):
                h=await measure.health(client)
                if not h['busy'] and not h['request_metrics']['active']:
                    measure.require_idle(h,4)
                    if 'identity' in report: assert measure.identity(h)==report['identity']
                    return h
                await asyncio.sleep(.2)
            raise RuntimeError('Four-slot jobs failed to retire')
        try:
            report['identity']=measure.identity(await idle())
            for effort in ('low','high'):
                cases=measure.cases(96,'structured_json')+measure.cases(96,'structured')
                cases=[{**c,'id':f'mixed-{i}'} for i,c in enumerate(cases)]
                report['waves'].append(await measure.wave(client,await idle(),cases,effort,True));save()
            code=measure.cases(32,'mixed_json')[0]
            for _ in range(3):
                cases=[{**code,'id':f'code-{i}'} for i in range(4)]
                report['waves'].append(await measure.wave(client,await idle(),cases,'high',True));save()
            states=[{} for _ in range(4)];visible=[asyncio.Event() for _ in states]
            cases=[measure.cases(256,'structured_json')[i%2] for i in range(4)]
            async def holder(i):
                state=states[i];state['content']=''
                body={'model':'GLM-5.3-Flash','reasoning_effort':'high','temperature':0,
                    'messages':[{'role':'user','content':cases[i]['prompt']}],
                    'max_tokens':4096,'response_format':{'type':'json_object'},
                    'stream':True,'stream_options':{'include_usage':True}}
                async with client.stream('POST','/v1/chat/completions',json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data: {'): continue
                        event=json.loads(line[6:])
                        if event.get('error'): raise RuntimeError(str(event['error']))
                        if event.get('id'):state['id']=event['id']
                        if event.get('usage'):state['usage']=event['usage']
                        for c in event.get('choices',[]):
                            text=c.get('delta',{}).get('content') or ''
                            state['content']+=text
                            if text:visible[i].set()
                            if c.get('finish_reason'):state['finish_reason']=c['finish_reason']
            before=await idle();tasks=[asyncio.create_task(holder(i)) for i in range(4)]
            try:
                await asyncio.wait_for(asyncio.gather(*(v.wait() for v in visible)),60)
                during=await measure.health(client)
                assert during['concurrent_ring']['peak_ring_leases']==4
                assert during['concurrent_ring']['peak_draft_batch']==4
                assert during['concurrent_ring']['peak_target_batch']==4
                for i in (0,2):tasks[i].cancel()
                await asyncio.gather(tasks[0],tasks[2],return_exceptions=True)
                replacements=[{**c,'id':f'reuse-{i}'} for i,c in enumerate(measure.cases(32,'structured_json'))]
                # The original two holders remain active. Track replacements
                # directly so a separate wave cannot mistake survivors for
                # unowned traffic; account for all six admissions afterward.
                await asyncio.sleep(.3)
                replacement_ids=set()
                replacement=await asyncio.gather(*(measure.stream(client,c,'low',replacement_ids) for c in replacements))
                await asyncio.wait_for(asyncio.gather(tasks[1],tasks[3]),120)
            finally:
                for t in tasks:t.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                report['lifecycle_partial']=states;save()
            for i in (1,3):
                assert states[i]['finish_reason']=='stop' and states[i].get('usage')
                assert json.loads(states[i]['content'])==cases[i]['expected']
            after=await idle()
            known={r['id'] for r in before['request_metrics']['recent']}|replacement_ids|{s['id'] for s in states}
            assert {r['id'] for r in after['request_metrics']['recent']}<=known
            assert after['request_queue']['admitted']==before['request_queue']['admitted']+6
            retired=[r for r in after['request_metrics']['recent'] if r['id'] in {states[i]['id'] for i in (0,2)}]
            assert len(retired)==2 and all(r['outcome'] in ('cancelled','disconnected') for r in retired)
            assert after['concurrent_ring']['reused_slots']>before['concurrent_ring']['reused_slots']
            report['lifecycle']={'passed':True,'retired':retired,'replacement':replacement,
                                  'before':before['concurrent_ring'],'after':after['concurrent_ring']}
            report['final_health']=after;report['passed']=True
        except BaseException as e:
            report['failure']=measure.failure_details(e);raise
        finally:save()
    print(json.dumps({'passed':True,'four_way_waves':len(report['waves']),'lifecycle':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(run(p.parse_args()))
