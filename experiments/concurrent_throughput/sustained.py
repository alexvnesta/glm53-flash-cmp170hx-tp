"""Fixed-work multi-agent load with immediate backfill and observed token counters.

The server retains two active slots. Four agents create an intentional queue.
Rates use completion tokens including reasoning, never SSE chunk counts.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import httpx
import measure


def fixtures(count):
    arrays = measure.cases(96)
    code = measure.cases(32, 'mixed_json')[0]
    literal = {'values': list(range(700, 764))}
    copy = {'id': 'copy-object', 'expected': literal, 'response_format': {'type': 'json_object'},
            'prompt': 'Return exactly this JSON object, without markdown: ' + json.dumps(literal)}
    templates = [arrays[0], code, arrays[1], copy]
    return [{**templates[i % 4], 'id': f'{i:03d}-'+templates[i % 4]['id']} for i in range(count)]


def check_traffic(h, identity, baseline, owned, started, admitted):
    if h['status'] != 'ok' or measure.identity(h) != identity:
        raise RuntimeError('Service identity or health changed; load yields')
    if h['request_queue']['admitted']-admitted > started:
        raise RuntimeError('Other traffic arrived; load yields')
    records = h['request_metrics']
    unknown = {r['id'] for r in records['active']+records['recent'] if r['id'] not in baseline|owned}
    if len(unknown) > started-len(owned):
        raise RuntimeError('Unowned traffic arrived; load yields')


def summarize(rows, samples, agents, wall, warmup=5):
    tokens = sum(r['usage']['completion_tokens'] for r in rows)
    by_agent = []
    for a in range(agents):
        own = [r for r in rows if r['agent'] == a]
        by_agent.append({'agent': a, 'requests': len(own),
            'tokens': sum(r['usage']['completion_tokens'] for r in own),
            'delivered_tok_s': sum(r['usage']['completion_tokens'] for r in own)/wall,
            'median_request_tok_s': statistics.median(r['usage']['completion_tokens']/r['finish_s'] for r in own) if own else None,
            'median_finish_s': statistics.median(r['finish_s'] for r in own) if own else None})
    steady = [s for s in samples if warmup <= s['seconds'] <= wall-warmup]
    interval = None
    if len(steady) > 1:
        left, right = steady[0], steady[-1]
        dt = right['seconds']-left['seconds']
        delta = [right['agent_tokens'][a]-left['agent_tokens'][a] for a in range(agents)]
        if min(delta) < 0: raise RuntimeError('Observed token counter moved backwards')
        interval = {'seconds': dt, 'aggregate_tok_s': sum(delta)/dt,
                    'per_agent_tok_s': [n/dt for n in delta],
                    'fraction_two_active': sum(s['active_sequences']==2 for s in steady)/len(steady),
                    'mean_active_sequences': statistics.mean(s['active_sequences'] for s in steady),
                    'mean_waiting': statistics.mean(s['waiting'] for s in steady)}
    return {'wall_s': wall, 'tokens': tokens, 'aggregate_tok_s': tokens/wall,
            'all_correct': all(r['correct'] for r in rows), 'failed_tasks': sum(not r['correct'] for r in rows),
            'agents': by_agent, 'steady': interval}


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'settings': vars(args)|{'output': str(args.output)}, 'runs': []}
    pair = fixtures(args.requests)
    report['fixture_sha256'] = hashlib.sha256(json.dumps(pair,sort_keys=True).encode()).hexdigest()
    path = args.output/'results.json'
    def save(): path.write_text(json.dumps(report,indent=2)+'\n')
    async with httpx.AsyncClient(base_url=args.base_url, timeout=httpx.Timeout(90,connect=3),
                                limits=httpx.Limits(max_connections=12)) as client:
        try:
            initial=await measure.health(client);measure.require_idle(initial)
            report['identity']=measure.identity(initial)
            await asyncio.sleep(1)
            measure.require_idle(await measure.health(client))
            for effort in args.effort:
                for agents in args.agents:
                    before=await measure.health(client);measure.require_idle(before)
                    assert measure.identity(before)==report['identity']
                    row={'agents':agents,'effort':effort,'requests':[],'samples':[]}
                    report['runs'].append(row);save()
                    started_n=0; owned=set(); assignment={}; completed_metrics={}
                    baseline={r['id'] for r in before['request_metrics']['recent']}
                    baseline_admitted=before['request_queue']['admitted']
                    queue=asyncio.Queue()
                    for case in pair:queue.put_nowait(case)
                    began=time.perf_counter()

                    async def worker(agent):
                        nonlocal started_n
                        while not queue.empty():
                            case=queue.get_nowait();started_n+=1
                            # Register the agent at the initial role event, including
                            # before its first model token becomes visible.
                            class IDs(set):
                                def add(self, ident):
                                    owned.add(ident);assignment[ident]=agent
                            try:
                                result=await measure.stream(client,case,effort,IDs())
                            except measure.ProbeResponseFailure as error:
                                result=error.response
                                if result.get('correct') is not False or not result.get('usage'):
                                    raise
                            result['agent']=agent
                            row['requests'].append(result)
                            queue.task_done()

                    async def monitor():
                        while True:
                            h=await measure.health(client)
                            check_traffic(h,report['identity'],baseline,owned,started_n,baseline_admitted)
                            records=h['request_metrics']
                            for record in records['recent']:
                                if record['id'] in owned:completed_metrics[record['id']]=record
                            counts=Counter()
                            for ident,record in completed_metrics.items():
                                counts[assignment[ident]]+=record['completion_tokens']
                            for record in records['active']:
                                if record['id'] in owned and record['id'] not in completed_metrics:
                                    counts[assignment[record['id']]]+=record['generated_tokens']
                            row['samples'].append({'seconds':time.perf_counter()-began,
                                'agent_tokens':[counts[a] for a in range(agents)],
                                'active_sequences':h['active_sequences'],'waiting':h['request_queue']['waiting'],
                                'phases':dict(Counter(r['phase'] for r in records['active'])),
                                'hybrid_rounds':h.get('hybrid_drafting',{}).get('hybrid_rounds'),
                                'ring':h['concurrent_ring']})
                            if len(row['samples'])%20==0:save()
                            await asyncio.sleep(.2)

                    tasks=[asyncio.create_task(worker(a)) for a in range(agents)]
                    async def all_workers(): await asyncio.gather(*tasks)
                    work=asyncio.create_task(all_workers());watch=asyncio.create_task(monitor())
                    try:
                        done,_=await asyncio.wait((work,watch),timeout=300,return_when=asyncio.FIRST_COMPLETED)
                        if watch in done:await watch
                        if work not in done:raise TimeoutError('Fixed-work load exceeded 300 seconds')
                        await work
                        wall=time.perf_counter()-began
                    finally:
                        for task in tasks+[work,watch]:task.cancel()
                        await asyncio.gather(*tasks,work,watch,return_exceptions=True)
                    for _ in range(50):
                        after=await measure.health(client)
                        if not after['busy'] and not after['request_metrics']['active']:break
                        await asyncio.sleep(.2)
                    measure.require_idle(after)
                    assert after['request_queue']['admitted']==baseline_admitted+len(pair),'Other traffic arrived'
                    assert len(owned)==len(pair)==len(row['requests'])
                    for record in after['request_metrics']['recent']:
                        if record['id'] in owned:completed_metrics[record['id']]=record
                    assert set(completed_metrics)==owned,'Per-request metrics missing'
                    assert all(r['outcome']=='completed' for r in completed_metrics.values())
                    row['server_metrics']=list(completed_metrics.values())
                    row['summary']=summarize(row['requests'],row['samples'],agents,wall)
                    row['ring_before']=before['concurrent_ring'];row['ring_after']=after['concurrent_ring']
                    save()
                    print(json.dumps({'effort':effort,'submitting_agents':agents,**row['summary']}),flush=True)
            report['completed']=True
        except BaseException as error:
            report['failure']=measure.failure_details(error)
            raise
        finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8012')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--agents',nargs='+',type=int,choices=(1,2,4),default=[1,2,4])
    p.add_argument('--effort',nargs='+',choices=('low','high'),default=['high'])
    p.add_argument('--requests',type=int,default=24)
    args=p.parse_args()
    if not 8<=args.requests<=64 or args.requests%4:p.error('Use 8..64 requests in multiples of four')
    asyncio.run(run(args))

if __name__=='__main__':main()
