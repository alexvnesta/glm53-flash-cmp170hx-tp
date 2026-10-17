import asyncio, hashlib, json, time
from pathlib import Path
import httpx

LAB=Path('/data2/llm-models/glm53-stream-usage-20261017')

async def one(client, effort, number, continuous=True):
    expected={'value':number,'ok':True}
    payload={'model':'GLM-5.3-Flash-'+effort,'messages':[{'role':'user','content':'Return exactly this JSON object: '+json.dumps(expected)}],
             'response_format':{'type':'json_object'},'max_tokens':512,'temperature':0,'stream':True,
             'stream_options':{'include_usage':True,'continuous_usage_stats':continuous}}
    counts=[];content='';reasoning='';timings={};ident=None;begin=time.monotonic();end=False
    async with client.stream('POST','/v1/chat/completions',json=payload,timeout=120) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: '):continue
            if line[6:]=='[DONE]':end=True;break
            e=json.loads(line[6:]);assert 'error' not in e,e
            ident=e['id']
            if 'usage' in e and e['usage'] is not None:counts.append(e['usage']['completion_tokens'])
            timings.update(e.get('timings',{}))
            for c in e.get('choices',[]):
                content+=c.get('delta',{}).get('content','');reasoning+=c.get('delta',{}).get('reasoning_content','')
    assert end and json.loads(content)==expected,(effort,content)
    assert counts==sorted(counts) and counts[-1]==timings['predicted_n'],counts
    if continuous:assert len(set(counts))>=3,counts
    else:assert len(counts)==1
    return {'effort':effort,'id':ident,'continuous':continuous,'usage':counts,'completion_tokens':counts[-1],
            'wall_s':time.monotonic()-begin,'content':content,'reasoning_sha256':hashlib.sha256(reasoning.encode()).hexdigest()}

async def main():
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8014') as client:
        records=[]
        for effort in ('low','high','max'):
            for continuous in (False,True):records.append(await one(client,effort,52,continuous))
        batches=await asyncio.gather(*(one(client,'max',n) for n in (101,202,303,404)))
        records.extend(batches)
        # Close an opt-in stream before EOS, then require all admission/native
        # slots to return to idle. No external client's request is cancelled.
        async with client.stream('POST','/v1/chat/completions',json={
            'model':'GLM-5.3-Flash-low','messages':[{'role':'user','content':'Count upward from 1, one integer per line.'}],
            'ignore_eos':True,'max_tokens':8192,'stream':True,'temperature':0,
            'stream_options':{'include_usage':True,'continuous_usage_stats':True}},timeout=120) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith('data: {'):
                    event=json.loads(line[6:])
                    if (event.get('usage') or {}).get('completion_tokens',0)>=1024:break
        for _ in range(100):
            h=(await client.get('/health')).json()
            if not h['busy'] and not h['request_metrics']['active'] and not h['active_sequences'] and not h['pending_sequences']:break
            await asyncio.sleep(.2)
        else:raise RuntimeError('Cancellation failed to drain native jobs')
        for index,effort in enumerate(('low','high','max')):
            a,b=records[2*index:2*index+2]
            assert a['content']==b['content'] and a['completion_tokens']==b['completion_tokens']
            assert a['reasoning_sha256']==b['reasoning_sha256'],effort
        result={'passed':True,'requests':records,'cancelled_stream_drained':True,'profile_digest':h['profile_digest']}
        (LAB/'live-stream-qualification-v2.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'passed':True,'completed_requests':len(records),'four_way_max_correct':len(batches),'cancellation_drained':True}),flush=True)

asyncio.run(main())
