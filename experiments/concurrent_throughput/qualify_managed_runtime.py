"""Bounded managed-KV soak and four-context restart recovery.

Uses the frozen sustained fixtures and the existing long-context oracles. This
client never changes hosting; an exclusive private backend is required.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace

import httpx
import measure
from probe_active_kv import decode_overlap
from qualify_four_slot_cache import flush, wave
import sustained


def require_managed(h):
    measure.require_idle(h, 4)
    assert h['cache_tokens'] == 1187840
    assert h['experimental_active_kv']['backing'] == 'cuda_managed_latent'
    assert h['experimental_active_kv']['pressure_driven'] is True
    assert not h['persistent_cache']['writes_disabled']
    m = h['persistent_cache']['metrics']
    assert not any(m.get(k, 0) for k in ('page_read_failures', 'write_errors'))
    assert not h['prompt_cache']['shared_pending_tickets']
    assert not h['prompt_cache']['active_job_records']
    assert h['prompt_cache']['paired_host_bytes'] <= h['prompt_cache']['paired_budget_bytes']


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'mode': args.mode, 'waves': []}
    def save():
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    async with httpx.AsyncClient(base_url=args.base_url,
            timeout=httpx.Timeout(1800, connect=3),
            limits=httpx.Limits(max_connections=12)) as client:
        try:
            before = await measure.health(client)
            require_managed(before)
            report['initial_health'] = before
            began = time.monotonic()
            if args.mode == 'soak':
                while time.monotonic()-began < args.seconds:
                    out = args.output/f'wave-{len(report["waves"])}'
                    await sustained.run(SimpleNamespace(output=out, requests=24,
                        active_sequences=4, base_url=args.base_url, agents=[4],
                        effort=['low', 'high']))
                    receipt = json.loads((out/'results.json').read_text())
                    assert receipt['completed']
                    assert all(r['summary']['all_correct'] for r in receipt['runs'])
                    h = await flush(client)
                    require_managed(h)
                    report['waves'].append({'receipt': out.name,
                        'summaries': [{**r['summary'], 'effort': r['effort']}
                                      for r in receipt['runs']], 'health': h})
                    report['seconds'] = time.monotonic()-began
                    save()
            else:
                seed = json.loads((args.seed/'summary.json').read_text())
                assert seed['passed']
                assert measure.identity(before) == seed['identity']
                if not args.allow_prior_requests:
                    assert before['request_queue']['admitted'] == 0, 'Require fresh process'
                assert before['persistent_cache']['fingerprint'] == seed['waves'][-1]['after']['persistent_cache']['fingerprint']
                cases = json.loads((args.cases/'cases-high.json').read_text())
                assert len(cases) == 4
                result = await wave(client, cases)
                for row in result['requests']:
                    assert row['input_tokens'] == 262144
                    assert row['timings']['cache_n'] >= 262144-8192
                counts = [r['usage']['completion_tokens'] for r in result['requests']]
                result['summary'] = {'effort': 'high', 'wall_s': result['wall_s'],
                    'completion_tokens_each': counts,
                    'aggregate_tokens_per_second': sum(counts)/result['wall_s'],
                    'per_agent_tokens_per_second': [n/result['wall_s'] for n in counts],
                    'four_decoding_overlap': decode_overlap(result)}
                report['waves'].append(result)
                h = await flush(client)
                require_managed(h)
                m = h['persistent_cache']['metrics']
                initial = before['persistent_cache']['metrics']
                assert m['page_reads'] > initial['page_reads']
                assert m['state_reads']-initial['state_reads'] >= 4
            report['final_health'] = await flush(client)
            require_managed(report['final_health'])
            assert measure.identity(report['final_health']) == measure.identity(before)
            report['seconds'] = time.monotonic()-began
            report['passed'] = True
        except BaseException as error:
            report['failure'] = measure.failure_details(error)
            raise
        finally:
            save()
    print(json.dumps({'passed': True, 'mode': args.mode,
                      'seconds': report['seconds'], 'waves': len(report['waves'])}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('soak', 'restore'), required=True)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--seed', type=Path)
    p.add_argument('--cases', type=Path)
    p.add_argument('--allow-prior-requests', action='store_true',
        help='Allow other clients before this probe; operator must attest a new process')
    a = p.parse_args()
    if a.seconds < 1: p.error('seconds must be positive')
    if a.mode == 'restore' and (a.seed is None or a.cases is None):
        p.error('restore requires seed and cases')
    asyncio.run(run(a))
