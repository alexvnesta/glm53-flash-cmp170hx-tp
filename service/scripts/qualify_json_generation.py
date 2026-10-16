"""Bounded JSON semantics, cancellation, isolation and concurrent soak.

No service configuration is changed. Run only in a coordinated test window;
the existing measurement client's ownership and identity guards cover soak waves.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import httpx


async def run(args, probe):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "functional": [], "soak": [], "lifecycle": None}
    path = args.output / "summary.json"

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    async with httpx.AsyncClient(base_url=args.base_url, timeout=120) as client:
        async def idle():
            for _ in range(100):
                h = await probe.health(client)
                if not h['busy'] and not h['active_sequences'] and not h['pending_sequences'] and not h['request_metrics']['active']:
                    probe.require_idle(h)
                    assert probe.identity(h) == report['identity']
                    return h
                await asyncio.sleep(.2)
            raise RuntimeError("JSON jobs did not release their slots")

        async def checked(body, expected, label):
            started = time.perf_counter()
            response = await client.post('/v1/chat/completions', json=body)
            response.raise_for_status()
            data = response.json()
            assert data['choices'][0]['finish_reason'] == 'stop', data
            content = data['choices'][0]['message']['content']
            assert json.loads(content) == expected, data
            assert data['usage']['completion_tokens'] > 0
            return {'label': label, 'seconds': time.perf_counter()-started, 'response': data}

        def request(prompt, effort='high', **kw):
            return {'model': 'GLM-5.3-Flash', 'messages': [{'role': 'user', 'content': prompt}],
                    'reasoning_effort': effort, 'temperature': 0, 'max_tokens': 2048,
                    'response_format': {'type': 'json_object'}, **kw}

        try:
            h = await probe.health(client); probe.require_idle(h)
            assert h.get('json_output', {}).get('enabled') is True
            report['identity'] = probe.identity(h)
            if not args.soak_only:
                values = [{'answer': 323}, {'nested': [None, True, False, -12.5], 'label': 'left'},
                          {'text': 'line1\nline2', 'quote': 'She said "hi"', 'unicode': 'β雪'}]
                for effort in ('low', 'high', 'max'):
                    for idx, expected in enumerate(values):
                        prompt = 'Return exactly this JSON object, without markdown: ' + json.dumps(expected, ensure_ascii=False)
                        row = await checked(request(prompt, effort), expected, f'{effort}-{idx}')
                        report['functional'].append(row); save()
                        await idle()

                # One active grammar and one unconstrained request: compare both
                # outputs to their own fixture, never to the other job's answer.
                pair = probe.cases(16, 'structured_json')
                pair[1] = {**probe.cases(16, 'structured')[1], 'id': 'plain-array-control'}
                before = await idle()
                report['mixed_filter_wave'] = await probe.wave(client, before, pair, 'high', True)
                assert all(r['correct'] for r in report['mixed_filter_wave']['requests'])
                await idle()

                states = [{}, {}]
                visible = [asyncio.Event(), asyncio.Event()]

                async def holder(index, start):
                    body = request(f'Return only a JSON object with key values containing all integers from {start} '
                                   f'through {start+255}, inclusive, in order.', stream=True,
                                   max_tokens=4096,
                                   stream_options={'include_usage': True})
                    state = states[index]; state.update(content='', finish_reason=None)
                    async with client.stream('POST', '/v1/chat/completions', json=body) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith('data: {'): continue
                            event = json.loads(line[6:])
                            if event.get('error'): raise RuntimeError(str(event['error']))
                            if event.get('id'): state['id'] = event['id']
                            if event.get('usage'): state['usage'] = event['usage']
                            for choice in event.get('choices', []):
                                text = choice.get('delta', {}).get('content') or ''
                                if text: state['content'] += text; visible[index].set()
                                if choice.get('finish_reason'): state['finish_reason'] = choice['finish_reason']

                before = await idle()
                tasks = [asyncio.create_task(holder(0, 1)), asyncio.create_task(holder(1, 1001))]
                try:
                    await asyncio.wait_for(asyncio.gather(*(e.wait() for e in visible)), 45)
                    tasks[0].cancel()
                    await asyncio.gather(tasks[0], return_exceptions=True)
                    replacement = await checked(request('Return only {"tag":"replacement"}.', 'low'),
                                                {'tag': 'replacement'}, 'replacement')
                    await asyncio.wait_for(tasks[1], 120)
                finally:
                    for task in tasks: task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    report['lifecycle_partial'] = states
                    save()
                assert states[1]['finish_reason'] == 'stop' and states[1].get('usage'), states[1]
                assert json.loads(states[1]['content']) == {'values': list(range(1001, 1257))}
                after = await idle()
                retired = [r for r in after['request_metrics']['recent'] if r['id'] == states[0]['id']]
                assert len(retired) == 1 and retired[0]['outcome'] in ('cancelled', 'disconnected'), retired
                assert after['concurrent_ring']['active_ring_leases'] == 0
                report['lifecycle'] = {'passed': True, 'retired': retired[0], 'survivor': states[1],
                                       'replacement': replacement, 'native_leases_after': 0}
                save()

            started = time.monotonic()
            while time.monotonic()-started < args.soak_seconds:
                index = len(report['soak'])
                effort = ('low', 'high', 'max')[index % 3]
                pair = probe.cases(32, 'mixed_json' if index % 2 else 'structured_json')
                before = await idle()
                row = await probe.wave(client, before, pair, effort, True)
                assert all(r['correct'] for r in row['requests'])
                assert row['ring_after']['batched_verifier_rounds'] > row['ring_before']['batched_verifier_rounds']
                report['soak'].append(row); save()
            report['soak_seconds'] = time.monotonic()-started
            report['soak_requests'] = 2 * len(report['soak'])
            report['final_health'] = await idle()
            report['passed'] = True
        except BaseException as error:
            report['failure'] = probe.failure_details(error)
            raise
        finally:
            save()
    print(json.dumps({'passed': True, 'functional_requests': len(report['functional']),
                      'soak_requests': report['soak_requests'], 'soak_seconds': report['soak_seconds'],
                      'lifecycle': bool(report['lifecycle'])}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--soak-seconds', type=int, default=300)
    p.add_argument('--soak-only', action='store_true')
    args = p.parse_args()
    if not 0 <= args.soak_seconds <= 600:
        p.error('Bounded soak must be 0..600 seconds')
    sys.path.insert(0, str(args.probe_dir))
    import measure
    asyncio.run(run(args, measure))


if __name__ == '__main__':
    main()
