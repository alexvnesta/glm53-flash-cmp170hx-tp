"""Validate the retained 2026-10-17 community runs and package their unchanged metrics.

Usage: python experiments/community_benchmark/package.py
Reads results/2026-10-17/runs/ (copied unchanged from the server) and the serving
/health captured after the deploy (controller/serving-health-after-deploy.json), and
writes the per-effort envelopes, summary.json and docs/COMMUNITY_BENCHMARK_2026-10-17.md.
Every accepted cell must reconcile its per-agent tokens with the upstream total, and every
accepted group must have stayed under the CPU-pressure gate. The validation counts in the
doc are transcribed from the qualification logs, not re-run here.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / 'experiments/community_benchmark/results/2026-10-17'
RUNS = DEST / 'runs'
HEALTH = DEST / 'controller/serving-health-after-deploy.json'
DOC = REPO / 'docs/COMMUNITY_BENCHMARK_2026-10-17.md'
EFFORTS = ('low', 'high', 'max')
CONTEXTS = (0, 32768, 131072)
PROFILE = '7c6da98c10332410ccf838449ab87fe86c6907c351bdd72958685e034f593031'
TOOL_COMMIT = '497350a81a77d7d11b4e418b2a26c30c3065949a'


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def audits_for(folder, finite=False):
    records = [json.loads(line) for line in (folder / 'agent-audit.jsonl').read_text().splitlines()]
    if finite:
        return [r for r in records if r['benchmark_mode'] == 'request-count' and r['request_count_target'] == 4]
    return [r for r in records if r['benchmark_mode'] == 'duration' and r['requested_duration'] == 30]


def collect():
    groups, excluded = {}, []
    for receipt_path in sorted(RUNS.glob('*/receipt.json')):
        receipt = load(receipt_path)
        if receipt['label'] == 'smoke':
            continue
        if not receipt['accepted']:
            excluded.append({'label': receipt['label'], 'attempt': receipt['attempt'], 'error': receipt['error']})
            continue
        assert receipt['profile_digest'] == PROFILE
        assert receipt['label'] not in groups
        folder = receipt_path.parent
        report = load(folder / 'benchmark_results.json')
        audits = audits_for(folder, receipt['finite'])
        assert len(audits) == len(report['results']) == 3
        for cell in report['results']:
            audit, = [a for a in audits if a['concurrency'] == cell['concurrency']]
            assert sum(audit['completion_tokens']) == cell['client_output_tokens']
            assert abs(sum(audit['tokens_per_second']) - cell['aggregate_tps']) < 1e-6
            assert cell['num_errors'] == 0 and cell['aggregate_tps'] > 0 and not cell['warmup_timed_out']
            if receipt['finite']:
                assert cell['completed_request_count'] == 4 and cell['client_output_tokens'] == 2048
        # run_guarded.py yields a group on CPU PSI avg10 >= 2; assert it rather than trust it.
        assert max(s['psi_cpu_avg10'] for s in receipt['health_samples']) < 2, receipt['label']
        groups[receipt['label']] = {'receipt': receipt, 'folder': folder, 'report': report, 'audits': audits}
    assert len(groups) == 12, f'Benchmark campaign is incomplete: {sorted(groups)}'
    return groups, excluded


def envelope(effort, groups):
    primary = [groups[f'{effort}-ctx{ctx}'] for ctx in CONTEXTS]
    finite = groups[f'{effort}-finite-ctx0']
    source_runs = []
    for item in [*primary, finite]:
        raw_path = item['folder'] / 'benchmark_results.json'
        receipt, samples = item['receipt'], item['receipt']['health_samples']
        source_runs.append({
            'label': receipt['label'], 'raw_file': raw_path.relative_to(DEST).as_posix(),
            'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            'started': receipt['started'], 'finished': receipt['finished'], 'command': receipt['command'],
            # Acceptance implies zero: run_guarded.py rejects a group whose window has any
            # POST in the switcher journal. The receipt keeps no separate count.
            'gates': {'public_router_posts': 0,
                      'cpu_psi_avg10_max': max(s['psi_cpu_avg10'] for s in samples),
                      'observed_efforts': sorted({e for s in samples for e in s['efforts']}),
                      'peak_active_sequences': max(s['active_sequences'] for s in samples)}})
    version = primary[0]['report']['metadata']['version']
    return {
        'tool': 'llm-inference-bench', 'tool_version': version, 'tool_commit': TOOL_COMMIT,
        'engine': 'exllamav3', 'engine_version': '0f685aa + row32 four-slot profile',
        'model': 'GLM-5.3-Flash', 'model_revision': '332ab457b709b7ba30dd9a448be5de03b80a7ac9',
        'gpu_model': 'NVIDIA CMP 170HX', 'gpu_count': 2, 'driver': '610.57.04', 'slot_ceiling': 4,
        'tier': 'spot-check', 'reasoning_effort': effort, 'profile_digest': PROFILE,
        'deployment': {'layout_kind': 'tp', 'layout_detail': 'TP2 on 2x CMP170HX 64GiB, 4 active sequences',
                       'offload': 'none', 'offload_detail': 'Active weights/KV on GPU; inactive RAM/disk retention'},
        'run_conditions': {
            'shared_server': True, 'duration_per_cell': 30, 'repeats_per_cell': 1,
            'note': 'Dedicated quiet intervals on the shared service; yield to public router traffic; '
                    'CPU PSI avg10 <2%. Effort via model alias. Q8 GPU KV; Weightless GLP-44 alpha2. '
                    'One run per cell. No cold-prefill claim.'},
        'metadata': {
            'version': version, 'engine': 'openai_proxy', 'model': 'GLM-5.3-Flash-' + effort,
            'context_lengths': list(CONTEXTS), 'concurrency_levels': [1, 2, 4], 'decode_mode': 'duration',
            'primary_decode_layer': 'sustained_decode', 'duration_per_test': 30, 'max_tokens': 2048,
            'temperature': 0, 'ignore_eos': True, 'prefill_mode': 'skipped', 'skip_prefill': True,
            'input_token_targeting': 'upstream estimate fallback: /tokenize returns token_length, not count',
            'packaged_from_multiple_original_runs': True},
        'results': [cell for item in primary for cell in item['report']['results']],
        'burst_results': finite['report']['results'],
        'agent_audits': [audit for item in primary for audit in item['audits']],
        'burst_agent_audits': finite['audits'], 'source_runs': source_runs}


def rows_for(groups):
    sustained, finite_rows = [], []
    for effort in EFFORTS:
        for ctx in CONTEXTS:
            item = groups[f'{effort}-ctx{ctx}']
            for cell in item['report']['results']:
                audit, = [a for a in item['audits'] if a['concurrency'] == cell['concurrency']]
                sustained.append({'effort': effort, **cell,
                                  'per_agent_tokens_per_second': audit['tokens_per_second'],
                                  'per_agent_completion_tokens': audit['completion_tokens']})
        item = groups[f'{effort}-finite-ctx0']
        for cell in item['report']['results']:
            audit, = [a for a in item['audits'] if a['concurrency'] == cell['concurrency']]
            finite_rows.append({'effort': effort, **cell, 'per_agent_tokens_per_second': audit['tokens_per_second']})
    return sustained, finite_rows


def rate(rows, effort, ctx, count):
    row, = [r for r in rows if r['effort'] == effort and r['context_tokens'] == ctx and r['concurrency'] == count]
    return row


def serving(health):
    """The header facts, read from the /health the deploy captured, never typed."""
    assert health['status'] == 'ok' and health['profile_digest'] == PROFILE
    steering = health['weightless']
    assert steering['enabled'] and steering['mode'] == 'project'
    return (f"GLM-5.3-Flash EXL3 3.05bpw, Weightless GLP-44 alpha {steering['alpha']:g} on {steering['layers']} "
            f"layers, TP2, {health['max_active_sequences']} native slots, {health['recurrent_state_dtype'].upper()} "
            f"recurrent state, {health['cache_format']} GPU KV pool {health['cache_tokens']:,}, per-request cap "
            f"{health['context_cap']:,}. Service profile digest `{health['profile_digest']}`; native SHA-256 "
            f"`{health['native_extension']['sha256']}`. Active RAM KV paging is disabled.")


def long_context_note(sustained):
    """How the single-agent long-context cells compare with the short prompt, computed."""
    ratios = [rate(sustained, e, ctx, 1)['aggregate_tps'] / rate(sustained, e, 0, 1)['aggregate_tps']
              for e in EFFORTS for ctx in CONTEXTS[1:]]
    worst = (1 - min(ratios)) * 100
    return (f'- At one agent, every nominal 32K/128K cell measured within {worst:.1f}% of the short prompt or '
            f'faster ({sum(r > 1 for r in ratios)} of {len(ratios)} faster). The tool ')


def document(sustained, finite_rows, window, health, psi_max):
    four = {e: rate(sustained, e, 0, 4)['aggregate_tps'] for e in EFFORTS}
    text = [
        '# GLM community benchmark on two CMP 170HX cards', '',
        f'October 17, 2026, {window} UTC (accepted groups). ' + serving(health), '',
        'The [llm-inference-bench v0.8.0](https://github.com/local-inference-lab/llm-inference-bench/tree/'
        + TOOL_COMMIT + ') source, unmodified and run under `observe.py` (which wraps two of its functions and '
        'proxies its shared counters to record each agent), supplies its original prose-generation workload. '
        'Each sustained cell measures 30 seconds '
        'after readiness warmup, temperature zero, max 2,048 completion tokens per request, with immediate backfill '
        'and ignore-EOS enabled. Effort is selected with the API model aliases. Context 0 means the short prompt, not '
        'literally zero input tokens. The requested 32K/128K sizes are nominal: exact targeting fell back to the '
        "tool's 8K usage-based padding calibration because this API returns `token_length`, while the tool expects "
        '`count`. The tables retain each server-reported actual input length. This input-size fallback does not '
        'change exact completion-token counting.', '',
        'Aggregate tok/s is the upstream continuous native completion-token delta divided by its steady-state '
        'interval. It includes reasoning and special/stop tokens counted by the engine. Per-agent contributions use '
        'that same interval and reconcile exactly with the aggregate token count. The observer reconstructs the '
        'unrounded interval from tokens/aggregate tok/s because the JSON seconds field is rounded to six decimals. '
        'It does not change upstream prompts, settings, scheduling or the aggregate measurement.', '',
        '## How to read these numbers', '',
        '- These are inference-throughput measurements, not task-solving or intelligence scores. Prefill rates and '
        'cold-context completion times were not measured.',
        '- The workload differs from the structured-task tables elsewhere in this repository, and the two are not '
        f'comparable. Four agents here deliver {four["low"]:.1f} / {four["high"]:.1f} tok/s at low / high effort on '
        'the short prompt; the [four-active trial](FOUR_ACTIVE_2026-10-16.md) measured 155.5 / 143.7 on its own '
        'tasks with the same slot count. Because the workloads differ, this campaign cannot say whether serving '
        'throughput changed between the two.',
        '- Each cell is one 30-second run with no repeats. Differences between efforts at the same context and '
        'agent count are not separated from run-to-run variation. With ignore-EOS, a request does not stop when '
        'its answer ends, so effort changes what the model writes, not how many tokens it generates.',
        long_context_note(sustained) +
        "received no draft-acceptance statistics from this server (its speculative-decoding fields read zero), so "
        'this campaign cannot say why.',
        '- The server stayed available to other clients. Every accepted group saw no public router POST (the '
        f'controller rejects a group that does) and CPU PSI avg10 at most {psi_max:.2f}%. Attempts interrupted by a public client or failed by the observer remain excluded, '
        'and their run folders are retained.', '']
    for ctx in CONTEXTS:
        text += ['## ' + ('Short prompt' if ctx == 0 else f'{ctx // 1024}K nominal input context'), '',
                 '| Effort | Agents | Actual input tokens | Aggregate tok/s | Actual per-agent tok/s |',
                 '|---|---:|---:|---:|---|']
        for effort in EFFORTS:
            for count in (1, 2, 4):
                row = rate(sustained, effort, ctx, count)
                text.append(f"| {effort} | {count} | {row['input_seq_len_avg']:,.0f} | {row['aggregate_tps']:.2f} | "
                            + ' / '.join(f'{x:.2f}' for x in row['per_agent_tokens_per_second']) + ' |')
        text.append('')
    text += ['## Finite completion time', '',
             'Each cell completes four requests of exactly 512 generated tokens, including reasoning, for 2,048 '
             'completion tokens total. The same requests run sequentially or concurrently. These are completion '
             'times for a fixed generation budget, not time to solve a scored task. Rates include request '
             'startup/prefill.', '',
             '| Effort | Agents | Four-request completion time (s) | Aggregate tok/s | Actual per-agent tok/s |',
             '|---|---:|---:|---:|---|']
    for effort in EFFORTS:
        for count in (1, 2, 4):
            row, = [r for r in finite_rows if r['effort'] == effort and r['concurrency'] == count]
            text.append(f"| {effort} | {count} | {row['measurement_seconds']:.2f} | {row['aggregate_tps']:.2f} | "
                        + ' / '.join(f'{x:.2f}' for x in row['per_agent_tokens_per_second']) + ' |')
    text += ['', '## Reporting and cancellation qualification', '',
             'Continuous usage is opt-in and counts native emitted token IDs before API channel splitting; the final '
             'authoritative engine total reconciles withheld stop tokens. Final-only clients retain their original '
             'usage events. The live checks completed ten requests across low/high/max, including four simultaneous '
             'max-effort JSON answers, and drained a stream cancelled after 1,024 generated tokens. Native '
             'cancellation now performs any last-job KV/disk flush on the existing GPU step worker under its '
             'ownership guard. Async queue notifications stay on the HTTP loop.', '',
             'The first duration smoke (`runs/smoke-attempt1`) ended in an HTTP 503 from that pre-existing '
             'cancellation ownership fault (server log: `controller/smoke-backend-failure.log`) and is excluded. A '
             'deployment path-comparison mistake caused one unnecessary rollback before the reporting fix was '
             'activated. Early observer trials exposed the upstream event wrapper and a sub-microsecond '
             'rounded-window boundary; those trials are excluded. No failed row was repaired into an accepted score.',
             '',
             'Validation, as recorded in the qualification logs and not re-run by this script: 47 HTTP tests, 68 server concurrency CPU tests (one skipped; CUDA uninitialized), two '
             'observer tests, source/lifecycle checks, full 972-file profile reproduction, live cancellation/JSON '
             'checks and the completed community campaign. Native kernels were not rebuilt or changed by the '
             'reporting/cancellation fixes.', '',
             'Reproduce the serving profile with [the packaged four-slot source profile]'
             '(../profiles/community-four-active/README.md). [The retained runs]'
             '(../experiments/community_benchmark/results/2026-10-17/) hold every attempt as the server wrote it '
             '(tool report, receipt with health samples, per-agent audit, event stream and log), the campaign '
             'controller, and the per-effort envelopes and summary that '
             '[package.py](../experiments/community_benchmark/package.py) builds from them. Every rate comes from '
             'API completion-token usage, with no chunk-count fallback.', '']
    return '\n'.join(text)


def main():
    groups, excluded = collect()
    for effort in EFFORTS:
        write(DEST / f'{effort}.json', envelope(effort, groups))
    sustained, finite_rows = rows_for(groups)
    write(DEST / 'summary.json', {'profile_digest': PROFILE, 'sustained': sustained, 'finite': finite_rows,
                                  'excluded_attempts': excluded})
    receipts = [g['receipt'] for g in groups.values()]
    window = (min(r['started'] for r in receipts)[11:16] + '–' + max(r['finished'] for r in receipts)[11:16])
    psi_max = max(s['psi_cpu_avg10'] for r in receipts for s in r['health_samples'])
    DOC.write_text(document(sustained, finite_rows, window, json.loads(HEALTH.read_text()), psi_max))
    print(f'Packaged {len(sustained)} sustained cells and {len(finite_rows)} finite cells; '
          f'{len(excluded)} excluded attempts retained')


if __name__ == '__main__':
    main()
