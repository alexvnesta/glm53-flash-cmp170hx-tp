"""Four independent 256K contexts; synthetic correctness/capacity probe.

Separate from the frozen short-task cohort. Uses serving token counts, includes
reasoning in completion counts, and reports each agent over the same wall time.
Never starts/stops hosting. An exclusive private backend is required.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import httpx
import measure
from qualify_four_slot_cache import body_for, validate_budget, wave, flush


def decode_overlap(result):
    """Observed token-counter rate while all four requests are decoding.

    Never sum separately timed per-request rates: they may include different
    prefill stalls or run after other agents finish. Report a common interval
    only when at least two samples show all four making decode progress.
    """
    ids=[r['request_id'] for r in result['requests']]
    candidates=[]
    for sample in result['samples']:
        active={r['id']:r for r in sample['phase_progress']}
        if sample['active']==4 and set(active)==set(ids) and all(
                active[i].get('phase')=='streaming' and active[i].get('generated_tokens',0)>0 for i in ids):
            candidates.append((sample['seconds'],[active[i]['generated_tokens'] for i in ids]))
    if len(candidates)<2:return None
    left,right=candidates[0],candidates[-1];dt=right[0]-left[0]
    if dt<=0:return None
    delta=[b-a for a,b in zip(left[1],right[1])]
    if min(delta)<0:raise RuntimeError('Per-agent generated token counter regressed')
    return {'seconds':dt,'aggregate_tokens_per_second':sum(delta)/dt,
            'per_agent_tokens_per_second':[n/dt for n in delta],
            'tokens_each':delta,'sample_count':len(candidates),
            'definition':'Observed generated-token counter increments over one common four-decoding interval'}


def server_decode_rates(result):
    metrics={r['id']:r for r in result['server_metrics']}
    return [metrics[r['request_id']]['decode_tokens_per_second'] for r in result['requests']]


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'waves':[], 'workload':'independent256K-marker-array96',
            'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save(): (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    async with httpx.AsyncClient(base_url=args.base_url,timeout=httpx.Timeout(1800,connect=3),
                                limits=httpx.Limits(max_connections=12)) as client:
        try:
            h=await measure.health(client);measure.require_idle(h,4)
            report['identity']=measure.identity(h)
            validate_budget([262144]*4,32768,h['cache_tokens'],h['context_length'])
            for effort in args.effort:
                path=args.cases/f'cases-{effort}.json'
                if path.exists(): cases=json.loads(path.read_text())
                else:
                    args.cases.mkdir(parents=True,exist_ok=True)
                    cases=[]
                    for i in range(4):
                        case=await body_for(client,262144,'active256K-'+effort,i)
                        case['body']['reasoning_effort']=effort
                        case['body']['max_tokens']=32768
                        # Reasoning markers can change the serving token count.
                        for _ in range(6):
                            r=await client.post('/tokenize',json=case['body']);r.raise_for_status()
                            delta=262144-r.json()['token_length']
                            if not delta: break
                            text=case['body']['messages'][0]['content']
                            if delta>0:text=text.replace('\nRecord C',' ledger'*delta+'\nRecord C')
                            else:
                                for _ in range(-delta):text=text.replace(' ledger','',1)
                            case['body']['messages'][0]['content']=text
                        else:raise RuntimeError('Unable to construct exact high-effort length')
                        case['prompt_sha256']=hashlib.sha256(case['body']['messages'][0]['content'].encode()).hexdigest()
                        cases.append(case)
                    path.write_text(json.dumps(cases)+'\n')
                for case in cases:
                    assert case['body']['reasoning_effort']==effort
                    r=await client.post('/tokenize',json=case['body']);r.raise_for_status()
                    assert r.json()['token_length']==262144
                start=time.monotonic(); result=await wave(client,cases)
                counts=[r['usage']['completion_tokens'] for r in result['requests']]
                common=result['wall_s']
                result['summary']={'effort':effort,'input_tokens_each':262144,
                    'completion_tokens_each':counts,'wall_s':common,
                    'aggregate_tokens_per_second':sum(counts)/common,
                    'per_agent_tokens_per_second':[n/common for n in counts],
                    'server_decode_tokens_per_second':server_decode_rates(result),
                    'four_active_observed':any(s['active']==4 for s in result['samples']),
                    'four_decoding_overlap':decode_overlap(result),
                    'rate_definition':'Completion tokens including reasoning / common pack wall time; includes prefill'}
                assert result['summary']['four_active_observed']
                report['waves'].append(result);save()
                print(json.dumps(result['summary']),flush=True)
                await flush(client)
            report['passed']=True
        except BaseException as error:
            report['failure']=measure.failure_details(error);raise
        finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--effort',nargs='+',choices=('low','high'),default=['low','high'])
    asyncio.run(run(p.parse_args()))
