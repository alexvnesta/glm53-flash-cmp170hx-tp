"""Verify one synthetic request during an already-running backend reload.

Never starts, stops or restarts services. Refuses a healthy backend because that
would not prove the reload wait. Run only during an authorized operator window.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
import httpx


async def run(a):
    report = {'passed': False}
    async with httpx.AsyncClient(base_url=a.base_url, timeout=300) as client:
        before = await client.get('/health')
        if before.status_code != 503:
            raise RuntimeError('Backend already ready; no reload-wait proof collected')
        seen = []; began = time.monotonic()
        async def watch():
            while True:
                r = await client.get('/health')
                h = r.json()
                seen.append({'seconds': time.monotonic()-began, 'status': r.status_code,
                             'switcher': h.get('switcher')})
                await asyncio.sleep(1)
        task = asyncio.create_task(watch())
        content = []; usage = None; finish = None; comments = 0
        try:
            body = {'model': 'GLM-5.3-Flash', 'reasoning_effort': 'low',
                    'temperature': 0, 'max_tokens': 128,
                    'response_format': {'type': 'json_object'},
                    'messages': [{'role': 'user', 'content':
                        'Return exactly {"answer":323} without markdown.'}],
                    'stream': True, 'stream_options': {'include_usage': True}}
            async with client.stream('POST', '/v1/chat/completions', json=body) as r:
                r.raise_for_status(); report['headers_s'] = time.monotonic()-began
                async for line in r.aiter_lines():
                    if line.startswith(':'): comments += 1
                    if not line.startswith('data: {'): continue
                    event = json.loads(line[6:]); assert not event.get('error'), event
                    usage = event.get('usage') or usage
                    for choice in event.get('choices', []):
                        content.append(choice.get('delta', {}).get('content') or '')
                        finish = choice.get('finish_reason') or finish
            assert finish == 'stop' and json.loads(''.join(content)) == {'answer': 323}
            assert usage and usage['completion_tokens'] > 0
            assert any(s['status'] == 503 and (s['switcher'] or {}).get('inflight', 0) >= 1 for s in seen)
            assert report['headers_s'] < 2
            report.update(passed=True, wall_s=time.monotonic()-began,
                          keepalive_comments=comments, usage=usage, health_samples=seen,
                          answer={'answer': 323})
        finally:
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'health_samples'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(run(p.parse_args()))
