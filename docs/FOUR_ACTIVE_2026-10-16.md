# Two versus four active sequences on 2× CMP 170HX

**Result:** four active sequences improve aggregate four-agent delivery by
**15.3% at low effort** and **14.9% at high effort**, with all **384 scored
tasks and 32 warmup requests correct**. The same 24 tasks finish about
**14.6–14.8% sooner**. Four-way native drafting/verification and cancellation
checks pass. At the end of this speed screen, the qualified two-slot
production profile was restored. The subsequent
[runtime qualification](FOUR_SLOT_RUNTIME_2026-10-16.md) adds four-slot
long-context, soak and restart-recovery evidence.

This capacity-only trial keeps the qualified row32 engine and native extension.
Both arms use the same reviewed service extension: paired-cache admission,
shared-prefix tickets and tier ownership follow the declared two/four-slot
capacity. The original two-slot defaults and capacity-drift terminal checks
remain. The initial unmodified service rejected four slots before allocation;
that failed startup and the earlier CPU-confounded controls remain separate.

It changes `--concurrent-max-sequences` and its matching allocation flag `-ambs`
from 2 to 4. Four submitting agents on the control share two active slots;
the candidate can draft and verify four independent sequences together.

## Matched workload and configuration

The frozen workload remains 24 tasks with immediate backfill: two exact integer
arrays, a Python run-compression implementation checked by the original behavior
oracle, and a literal JSON copy. Each profile uses 1, 2 and 4 submitting agents
at **low** and **high** effort. The four-agent cells have a second 24-task run.
Both profiles warm up with four submitting agents, at each effort. Warmups are
reported separately. Prompts, `temperature=0`, `max_tokens=768`,
response formats and validators are unchanged. Failed answers retain their
original score; no answer repair or output normalization is permitted.

Fixture SHA-256:
`38b5c9340800821e1ba5cf10f997c3831f5cc2858f23c5008b14ad916fb05ee7`.

Both profiles retain GLM-5.3-Flash EXL3 3.05 bpw, TP2, Weightless alpha 2 on
layers 1–44, Q8 cache pool 655,360 tokens, context cap 392,960, K7 DFlash2,
hybrid drafting 15, fair rounds 4/chunk 512, prefill chunk 2,048, NCCL reductions
and native broadcasts. Private compilation and disk-cache directories isolate
the arms. Neither profile inherits a previous arm's disk cache.

The same autosplit policy and GPU budgets derive slightly different weight
partitions when the allocation batch grows to four. The vocabulary-head
boundary moves, for example. This is a
comparison of the resulting capacity profiles, including that derived placement
change; it does not isolate a kernel-only batching effect. Draft ring state
also grows with capacity. The unchanged context cap is a per-request ceiling;
the finite shared pool does not promise four simultaneous maximum-length chats.

Native SHA-256:
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`.
The complete 972-file engine manifest is identical between arms. No kernel
arithmetic, launch geometry or native capacity is changed for this experiment.

The fused small-row path supports 32 target verification rows. Four regular
K7 sequences require 32 rows. A hybrid-15 round with four sequences can require
64 rows and use the existing generic path. This trial measures that existing
behavior; it does not claim a qualified 64-row fused kernel.

## Metrics and interpretation

- **Aggregate tok/s:** total API completion tokens, including reasoning, divided
  by the whole 24-task elapsed time, including prefill, queueing and validation.
- **Per-agent tok/s:** that agent's actual completion tokens divided by the same
  workload time. Individual rates sum to the aggregate. Assignment and output
  lengths can differ between agents and runs.
- **Completion time:** elapsed seconds to finish the same 24 tasks. Output-token
  counts are retained because shorter answers can finish sooner without a
  higher token-generation rate.
- **Steady rates:** generated-token counter differences excluding the first and
  last five seconds, separate from the complete-work rates above.

Repeated cells pool total tokens over total wall time; completion time is the
mean seconds per 24 tasks. These short synthetic tasks do not establish general
coding quality, long-context capacity or performance of real agent traces.

Before candidate results, the operational speed screen requires at least 2%
more four-agent aggregate tok/s at both efforts, no greater than 2% regression
in one/two-agent cells, no task failures, and observed four-sequence native
drafting/verification. This is an engineering screen, not a statistical
significance claim. A passing speed screen alone does not complete production
soak, long-context, restart or cache-recovery qualification.

## Safety and reproducibility

The benchmark defaults to the qualified two-slot contract. Four slots require
the explicit `sustained.py --active-sequences 4` option; a capacity mismatch,
busy service, foreign traffic or identity drift aborts the client. The preparer
verifies the source/native identities, clones the profile, and changes both
capacity flags together without starting it.

Each scored cell waits for CPU PSI `some avg10 < 2%` at three five-second
samples, then records PSI once per second during the run. A sample at or above
2% flags that entire cell's timing for exclusion and a bounded repeat. Its
answers, scores, timing and pressure samples remain in the receipts. This rule
was established before the revised four-slot comparison; it is not selected
from the observed speed result.

The GPU window checks idle requests/queue/leases and sole production-unit GPU
ownership before stopping the model. It has a bounded runtime, a `finally`
restoration, and an independent systemd post-stop restoration command. Other
clients and CPU jobs are left running. An initial CPU-pressure preflight failure
occurred before model shutdown; that attempt is retained separately.

The four-slot lifecycle probe separately exercises filtered/unfiltered requests,
four concurrent high-effort coding jobs, cancellation of two streams, two
correct survivors and replacement-slot reuse. It preserves the strict existing
JSON/code oracles. GPU batches and slot retirement are checked directly.

## Results

| Effort | Submitting agents | Two-slot aggregate tok/s | Four-slot aggregate tok/s | Change |
|---|---:|---:|---:|---:|
| Low | 1 | 120.1 | 122.8 | +2.2% |
| Low | 2 | 136.9 | 137.5 | +0.4% |
| Low | 4 | 134.8 | 155.5 | +15.3% |
| High | 1 | 112.4 | 113.0 | +0.5% |
| High | 2 | 124.6 | 125.4 | +0.7% |
| High | 4 | 125.0 | 143.7 | +14.9% |

| Effort | Agents | Two-slot actual per-agent tok/s | Four-slot actual per-agent tok/s |
|---|---:|---|---|
| Low | 1 | 120.1 | 122.8 |
| Low | 2 | 70.2 / 66.7 | 70.5 / 67.0 |
| Low | 4 | 20.8 / 38.5 / 36.1 / 39.4 | 41.3 / 37.6 / 36.2 / 40.5 |
| High | 1 | 112.4 | 113.0 |
| High | 2 | 65.4 / 59.2 | 65.8 / 59.6 |
| High | 4 | 21.2 / 24.2 / 42.1 / 37.5 | 45.9 / 22.8 / 37.4 / 37.5 |

These are individual delivered-token rates, not an equal division of the
aggregate. Immediate backfill can assign different tasks to each agent across
profiles; do not interpret one agent's rate change as a matched-task latency
comparison. Rounded entries can differ by 0.1 tok/s from the rounded sum.

| Effort | Agents | Two-slot seconds / 24 tasks | Four-slot seconds / 24 tasks | Two-slot tokens | Four-slot tokens |
|---|---:|---:|---:|---:|---:|
| Low | 1 | 50.96 | 49.84 | 6120 | 6120 |
| Low | 2 | 44.70 | 44.51 | 6121 | 6121 |
| Low | 4 | 45.42 | 38.77 | 6123 | 6028 |
| High | 1 | 51.66 | 51.39 | 5808 | 5808 |
| High | 2 | 48.18 | 47.85 | 6002 | 6002 |
| High | 4 | 48.62 | 41.42 | 6078 | 5950 |

Four-slot four-agent answers emit **1.6% fewer tokens at low effort** and
**2.1% fewer at high effort**. Completion-time savings include these length
differences; the separate tok/s gains use the actual emitted token counts.
No whitespace normalization, answer extraction or prompt change is used.

## Repeatability and native batching

| Effort | Profile | Repeat 1 aggregate tok/s | Repeat 2 aggregate tok/s |
|---|---|---:|---:|
| Low | slots2 | 137.87 | 131.86 |
| Low | slots4 | 155.65 | 155.28 |
| High | slots2 | 124.67 | 125.36 |
| High | slots4 | 143.27 | 144.06 |

The low-effort control varies by about 4.5% between repeats, while both
four-slot repeats exceed both controls. Two repeats support the observed
profile comparison; they are not a broad statistical characterization.
All primary timing cells pass the predeclared CPU-pressure rule; none is
excluded. Earlier original-service controls remain in a separate receipt
section with the pre-candidate confound decision.

The two-slot four-agent runs average **1.90–1.93 active sequences** during
the steady interval. Four slots average **3.80–3.83**, with all four active
in **82–85%** of steady samples. Native peak ring leases, draft batch and
target verification batch all reach **4**. This is observed four-sequence
batching, rather than four HTTP submitters sharing a two-sequence engine.
Every primary cell restores **1,536 cached prompt tokens**, in both profiles.

Draft ring state grows from **104,857,600 bytes (100 MiB)** to
**209,715,200 bytes (200 MiB)**. Source/native/service-overlay identities are
verified equal between profiles: all **972 engine files** and the same native
binary. Autosplit moves one expert between ranks in each of layers 35 and 42,
and the vocabulary-head split changes from 75,136/79,744 to 75,008/79,872
channels. Those derived changes remain part of the measured capacity profile.

## Runtime checks and disposition

The four-slot functional probe passes two mixed filtered/unfiltered waves
(eight requests), three four-way high-effort coding waves (12 requests), and
a cancellation/reuse scenario: two streams cancel, two survivors return the
exact expected JSON, and two replacements complete. All slots and admission
records retire. This is 24 successful functional requests plus two intentional
cancellations, separate from the performance cohort.

CPU checks execute **68 native scheduler/cache contracts** with one native-only
SAM skip against the exact row32 Python sources; CUDA stays uninitialized.
New cases cover independent four-job target/draft state, lossless outputs,
slot reuse, fifth-job rejection, capacity-drift terminal behavior and four
shared-prefix tickets transitioning to owned jobs. The bounded probe/preparer
suite passes 44 tests on Linux (43 plus one platform skip on macOS).
HTTP checks pass 44 cases and response-lifetime AST checks pass four.
The regular pinned 970-file engine source also reproduces exactly.

slots2 scored-interval CPU-PSI avg10: mean 0.148%, maximum 1.34% in ten-second host samples; per-cell one-second samples are retained separately.

slots4 scored-interval CPU-PSI avg10: mean 0.015%, maximum 0.14% in ten-second host samples; per-cell one-second samples are retained separately.

Independent production inspection at `2026-10-16T21:51:33.044700+00:00` verifies the
full source/native identity, sole production-unit GPU ownership, healthy idle
state, empty requests/queue/leases, Weightless alpha 2 on both devices,
`NRestarts=0` and inactive Qwen. All experiment model, controller
and watcher units are inactive. Production retains two slots, native SHA
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046` and profile digest
`2bfd7ca630f488034cfc1be093317719aeb41e227a8e8766a3e2c856e637c91c`.

**The speed screen passes; four slots are an opt-in experiment.** This screen
does not include long-context soak, persistent-cache restart/recovery or
general real-agent qualification. The later
[runtime follow-up](FOUR_SLOT_RUNTIME_2026-10-16.md) supplies synthetic capacity,
soak and recovery checks. The general launcher defaults to two sequences.

[Complete synthetic receipts](../experiments/concurrent_throughput/four_active_2026-10-16.json)
retain all primary requests and scores, warmups, actual per-agent rates,
completion times, token totals, prefix reuse, native batching, per-cell CPU
pressure, earlier confounded controls, source/native/overlay hashes and the
restored-production inspection. Private host logs and original receipts remain
in the dated lab.
