# GLM community benchmark on two CMP 170HX cards

October 17, 2026, 12:31–13:19 UTC (accepted groups). GLM-5.3-Flash EXL3 3.05bpw, Weightless GLP-44 alpha2, TP2, four native slots, FP16 recurrent state, Q8 GPU KV pool 655,360, per-request cap 392,960. Service profile digest `7c6da98c10332410ccf838449ab87fe86c6907c351bdd72958685e034f593031`; native SHA-256 `f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`. Active RAM KV paging is disabled.

The unmodified [llm-inference-bench v0.8.0](https://github.com/local-inference-lab/llm-inference-bench/tree/497350a81a77d7d11b4e418b2a26c30c3065949a) supplies its original prose-generation workload. Each sustained cell measures 30 seconds after readiness warmup, temperature zero, max 2,048 completion tokens per request, with immediate backfill and ignore-EOS enabled. Effort is selected with the API model aliases. Context 0 means the short prompt, not literally zero input tokens. The requested 32K/128K sizes are nominal: exact targeting fell back to the tool's 8K usage-based padding calibration because this API returns `token_length`, while the tool expects `count`. The tables retain each server-reported actual input length. This input-size fallback does not change exact completion-token counting.

Aggregate tok/s is the upstream continuous native completion-token delta divided by its steady-state interval. It includes reasoning and special/stop tokens counted by the engine. Per-agent contributions use that same interval and reconcile exactly with the aggregate token count. The observer reconstructs the unrounded interval from tokens/aggregate tok/s because the JSON seconds field is rounded to six decimals. It does not change upstream prompts, settings, scheduling or the aggregate measurement.

## How to read these numbers

- These are inference-throughput measurements, not task-solving or intelligence scores. Prefill rates and cold-context completion times were not measured.
- The workload differs from the structured-task tables elsewhere in this repository, and the two are not comparable. Four agents here deliver 82.7 / 96.3 tok/s at low / high effort on the short prompt; the [four-active trial](FOUR_ACTIVE_2026-10-16.md) measured 155.5 / 143.7 on its own tasks with the same slot count. Because the workloads differ, this campaign cannot say whether serving throughput changed between the two.
- Each cell is one 30-second run with no repeats. Differences between efforts at the same context and agent count are not separated from run-to-run variation. With ignore-EOS, a request does not stop when its answer ends, so effort changes what the model writes, not how many tokens it generates.
- At one agent, the nominal 32K/128K cells measured the same as or faster than the short prompt. The tool received no draft-acceptance statistics from this server (its speculative-decoding fields read zero), so this campaign cannot say why.
- The server stayed available to other clients. Accepted groups observed no public router POSTs and CPU PSI avg10 below 2%. Attempts interrupted by a public client or failed by the observer remain excluded, and their run folders are retained.

## Short prompt

| Effort | Agents | Actual input tokens | Aggregate tok/s | Actual per-agent tok/s |
|---|---:|---:|---:|---|
| low | 1 | 78 | 41.72 | 41.72 |
| low | 2 | 78 | 62.83 | 31.42 / 31.42 |
| low | 4 | 78 | 82.70 | 20.68 / 20.68 / 20.68 / 20.68 |
| high | 1 | 78 | 43.16 | 43.16 |
| high | 2 | 78 | 59.70 | 29.87 / 29.83 |
| high | 4 | 78 | 96.27 | 24.07 / 24.07 / 24.07 / 24.07 |
| max | 1 | 78 | 41.47 | 41.47 |
| max | 2 | 78 | 65.36 | 32.68 / 32.68 |
| max | 4 | 78 | 101.22 | 25.30 / 25.30 / 25.30 / 25.30 |

## 32K nominal input context

| Effort | Agents | Actual input tokens | Aggregate tok/s | Actual per-agent tok/s |
|---|---:|---:|---:|---|
| low | 1 | 32,319 | 47.60 | 47.60 |
| low | 2 | 32,319 | 44.73 | 22.36 / 22.36 |
| low | 4 | 32,319 | 68.18 | 17.05 / 17.05 / 17.05 / 17.05 |
| high | 1 | 32,314 | 42.68 | 42.68 |
| high | 2 | 32,314 | 44.07 | 22.03 / 22.03 |
| high | 4 | 32,314 | 72.21 | 18.05 / 18.05 / 18.05 / 18.05 |
| max | 1 | 32,319 | 46.92 | 46.92 |
| max | 2 | 32,319 | 46.35 | 23.18 / 23.18 |
| max | 4 | 32,319 | 64.14 | 16.06 / 16.06 / 16.06 / 15.96 |

## 128K nominal input context

| Effort | Agents | Actual input tokens | Aggregate tok/s | Actual per-agent tok/s |
|---|---:|---:|---:|---|
| low | 1 | 128,886 | 43.37 | 43.37 |
| low | 2 | 128,886 | 43.38 | 21.69 / 21.69 |
| low | 4 | 128,886 | 56.09 | 14.06 / 14.06 / 14.06 / 13.90 |
| high | 1 | 128,886 | 45.87 | 45.87 |
| high | 2 | 128,886 | 47.32 | 23.66 / 23.66 |
| high | 4 | 128,886 | 65.45 | 16.41 / 16.41 / 16.41 / 16.21 |
| max | 1 | 128,878 | 49.82 | 49.82 |
| max | 2 | 128,878 | 51.77 | 25.89 / 25.89 |
| max | 4 | 128,878 | 67.97 | 17.03 / 17.03 / 17.03 / 16.87 |

## Finite completion time

Each cell completes four requests of exactly 512 generated tokens, including reasoning, for 2,048 completion tokens total. The same requests run sequentially or concurrently. These are completion times for a fixed generation budget, not time to solve a scored task. Rates include request startup/prefill.

| Effort | Agents | Four-request completion time (s) | Aggregate tok/s | Actual per-agent tok/s |
|---|---:|---:|---:|---|
| low | 1 | 48.01 | 42.66 | 42.66 |
| low | 2 | 36.73 | 55.76 | 27.88 / 27.88 |
| low | 4 | 26.39 | 77.60 | 19.40 / 19.40 / 19.40 / 19.40 |
| high | 1 | 46.25 | 44.28 | 44.28 |
| high | 2 | 34.21 | 59.87 | 29.94 / 29.94 |
| high | 4 | 21.35 | 95.91 | 23.98 / 23.98 / 23.98 / 23.98 |
| max | 1 | 39.72 | 51.56 | 51.56 |
| max | 2 | 32.45 | 63.11 | 31.55 / 31.55 |
| max | 4 | 21.36 | 95.87 | 23.97 / 23.97 / 23.97 / 23.97 |

## Reporting and cancellation qualification

Continuous usage is opt-in and counts native emitted token IDs before API channel splitting; the final authoritative engine total reconciles withheld stop tokens. Final-only clients retain their original usage events. The live checks completed ten requests across low/high/max, including four simultaneous max-effort JSON answers, and drained a stream cancelled after 1,024 generated tokens. Native cancellation now performs any last-job KV/disk flush on the existing GPU step worker under its ownership guard. Async queue notifications stay on the HTTP loop.

The first duration smoke exposed that pre-existing cancellation ownership fault and is excluded. A deployment path-comparison mistake caused one unnecessary rollback before the reporting fix was activated. Early observer trials exposed the upstream event wrapper and a sub-microsecond rounded-window boundary; those trials are excluded. No failed row was repaired into an accepted score.

Validation: 47 HTTP tests, 68 server concurrency CPU tests (one skipped; CUDA uninitialized), two observer tests, source/lifecycle checks, full 972-file profile reproduction, live cancellation/JSON checks and the completed community campaign. Native kernels were not rebuilt or changed by the reporting/cancellation fixes.

Reproduce the serving profile with [the packaged four-slot source profile](../profiles/community-four-active/README.md). [The retained runs](../experiments/community_benchmark/results/2026-10-17/) hold every attempt as the server wrote it (tool report, receipt with health samples, per-agent audit, event stream and log), the campaign controller, and the per-effort envelopes and summary that [package.py](../experiments/community_benchmark/package.py) builds from them. Every rate comes from API completion-token usage, with no chunk-count fallback.
