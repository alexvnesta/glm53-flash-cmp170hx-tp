# Active system-RAM KV investigation on two CMP170HX cards

This experiment asks whether four independent 262,144-token inputs can stay
active together. The input total is 1,048,576 tokens. The normal GPU target pool
is 655,360 tokens; the existing inactive-page RAM tier does not increase that
active capacity.

The private prototype uses 1,187,840 logical token slots. This covers four
262,144-token inputs, a 32,768-token output reservation for each request, and
8,192 additional token slots. The per-request context cap stays 392,960.

## Execution path

The Q8 latent payload and its group scales live in a fixed registered system-RAM
arena for each attention layer. The existing native `pinned_cuda_view` API gives
the CUDA kernels an alias of that storage. The sparse index planes remain in
VRAM. Attention reads active latent entries over PCIe while the four native
target/draft state slots continue to hold their respective owners.

This is **static active host backing**, not a pressure-driven hot-page manager.
It preserves the Q8 format, index selection, attention math, recurrent precision,
speculative settings and scheduler. It is different from parking a completed
request or queueing requests until another agent finishes.

The older `GLM53_TP_ACTIVE_HOST` migration flag remains disabled. Native
concurrency still refuses that older path. The copied prototype instead reports
`experimental_active_kv.backing = mapped_host_latent` in `/health`; its physical
backing is fixed when the target cache is allocated.

The arena must outlive every CUDA alias. The prototype preserves the loader's
CPU export pass and changes only real CUDA worker allocations. Teardown drops
the cache views before closing the arena and synchronizes the worker. TP
placement accounts for GPU-resident bytes while logical cache storage accounting
continues to include all planes for transfers and retention.

The source geometry is 7,392 target bytes/token over eleven MLA layers, including
5,984 bytes/token for the Q8 latent and scales. At the prototype's pool size that
is 7,108,034,560 bytes (6.62 GiB) of active latent backing in RAM. Index storage,
model weights, recurrent/draft state, staging and workspaces still consume VRAM.
The existing 8 GiB inactive-KV and 5 GiB paired-checkpoint budgets are separate.

## Short-task results

Fresh baseline and prototype runs, each with 24 unchanged tasks per effort and
four submitting/active slots, after separate warmup:

| Setup | Effort | Aggregate tok/s | Actual agent 1 / 2 / 3 / 4 tok/s | 24-task wall time | Completion tokens |
|---|---|---:|---|---:|---:|
| Current GPU pool | Low | 154.98 | 41.14 / 37.43 / 36.07 / 40.34 | 38.89 s | 6,028 |
| Active RAM latent | Low | 128.37 | 30.20 / 30.89 / 27.72 / 39.56 | 43.25 s | 5,552 |
| Current GPU pool | High | 143.12 | 45.75 / 22.73 / 37.26 / 37.38 | 41.57 s | 5,950 |
| Active RAM latent | High | 120.73 | 33.32 / 20.66 / 28.29 / 38.47 | 48.89 s | 5,903 |

All 96 scored answers passed their independent oracles; all primary CPU timing
samples remained below 2% CPU PSI avg10. The practical throughput cost was 17.2%
at low effort and 15.6% at high effort. Output lengths differ, particularly in
low effort, so wall-time changes alone do not measure token-generation speed.
This is one fresh run per configuration/effort, not a statistical confidence
interval or proof that these rates hold across all real workloads.

The direct GPU-only control with a 1,064,960-token pool exhausted GPU 0 during
warmup and never became a serving backend. Its final rank-strategy validation
error was secondary to the logged CUDA OOM. It produced no scored requests or
valid throughput result. The current production ceiling was restored after
this failed attempt.

## Four independent 256K inputs: capacity and speed

**Both packs passed all four marker/array oracles.** All four requests were
observed decoding together, with their full independent contexts held by the
same native four-slot generator. Every request reported exactly 262,144 input
tokens and **zero cached input tokens** in this cold test.

| Effort | Aggregate four-agent decode tok/s | Actual agent 1 / 2 / 3 / 4 decode tok/s | Common decode interval | Cold pack completion | First-token range |
|---|---:|---|---:|---:|---|
| Low | 93.53 | 24.32 / 23.33 / 23.13 / 22.74 | 10.07 s, 11 samples | 732.10 s (12m 12s) | 718.33–720.53 s |
| High | 104.76 | 26.08 / 26.64 / 25.52 / 26.52 | 16.10 s, 17 samples | 738.23 s (12m 18s) | 718.86–721.05 s |

These decode rates use observed generated-token counter increments during one
common interval in which **all four** requests had started generation and were
still active. They exclude cold prefill and the completion tail. They are short
observed intervals, not a long-duration generation soak. Each pack processed
1,048,576 input tokens before completing the four answers.

Low effort produced 1,016 completion tokens; high produced 1,750, including its
additional reasoning. The higher high-effort decode rate is an observation for
that different token mix, not evidence that increasing effort always speeds up
generation. Including the entire cold-prefill wall time, aggregate completion
delivery was **1.39 tok/s low and 2.37 tok/s high**. Per-agent whole-pack delivery
was 0.347 / 0.350 / 0.347 / 0.344 tok/s and
0.592 / 0.595 / 0.592 / 0.592 tok/s, respectively.

The short-task 16–17% setup cost cannot be applied as a measured slowdown factor
for 256K contexts: the GPU-only larger-pool control did not finish startup. This
study does not rule out another GPU-only memory/placement/warmup configuration
fitting those inputs.

The 32K output **reservation** per agent was tested; four complete 32K generated
answers were not. All 14 direct allocator/attention/export checks passed, as did
24 lifecycle completions, two intended cancellations, 16 prototype warmups,
48 prototype short tasks and the eight long-context requests. The prototype API
finished with 96 successful completions and the two expected cancellations.

## Qualification and measurement contracts

The short comparison uses the unchanged 24-task mixed coding/structured cohort,
four agents, temperature zero, 768 output tokens and separate low/high effort.
Completion token counts include reasoning. Aggregate and actual per-agent rates
use the same full cohort wall time, including admission, prefill and the tail.
Per-agent rates need not be equal because task lengths and backfill differ.

The separate capacity probe uses four independently hashed exact 262,144-token
inputs, each with three marker records and a known JSON array of integers
0..95. It reserves 32K output tokens but normally completes with a much shorter
answer. Correct marker retrieval is a capacity/custody check, not a general
long-context coding or reasoning benchmark. Cold-prefill time must be shown
separately from decoding; short-task tok/s must not be presented as 256K speed.

The direct GPU oracle compares Q8 append and sparse attention results exactly
against GPU-resident copies, covers shuffled page tables, 1/8/16-row decode,
512-row prefill and page transfer, and checks the TP loader's CPU export path.
Model-level lifecycle checks cover simultaneous requests, cancellation and reuse.

Changing the active cache footprint also changes the TP allocator's placement.
The practical setup comparison therefore includes that placement change; it does
not isolate a single kernel's overhead. A three-repeat attention microbenchmark
is supplementary latency evidence and is not model tokens/second.

The two cold long-context packs run in low-then-high order. The high pack also
evicts the preceding histories through the ordinary inactive RAM/disk tiers.
Their elapsed-time difference therefore includes that cache work and is not an
isolated causal estimate of reasoning effort.

## Upstream distinction

Upstream's [host-cache discussion](https://github.com/turboderp-org/exllamav3/issues/269)
describes both idle-page caching and the harder active-cache request. The
maintainer's Tier 1 implementation comment does not establish active
oversubscription. The checked-in production path still restores inactive pages
during allocation and keeps active pages in VRAM. This prototype uses the existing
mapped-memory native API to investigate the missing active path separately.

Upstream was rechecked during this investigation: master
[`058b8630fd`](https://github.com/turboderp-org/exllamav3/commit/058b8630fd081f3ff46fce6246955b157c3a2e3b)
and dev
[`ba2feb286e`](https://github.com/turboderp-org/exllamav3/commit/ba2feb286ee7854eff33c71b95d6e278fe7c4e91).
Their changes after the recipe's source pin do not replace the MLA cache
allocator or `CPUPageCache` with an active oversubscription implementation.
The FP16 recurrent-state and allocation-rollback fixes in that history are
already backported in this recipe. A full upstream upgrade was not mixed into
this experiment.

`prepare_active_kv.py` only prepares a new, isolated directory and launch plan.
It never stops a service, changes a systemd unit, or deploys the result. The
production pool guard is unchanged; the larger limit applies only to copied
experiment sources. Do not raise the normal service ceiling without GPU
qualification and a bounded restoration window.

## Disposition and next work

The current production service retains its qualified GPU-resident active cache.
This allocator is an isolated capacity experiment. It has not received production
soak, process-restart, vision or general long-context coding qualification.

The most useful next investigation is retaining a bounded GPU hot set and
staging selected Q8 latent rows once per batched attention step, rather than
repeated direct host reads across attention head tiles. Partial host backing and
per-request warmup bounds are also untested options. These are possible ways to
recover overhead, not measured improvements from this study.

The [complete synthetic evidence](../experiments/concurrent_throughput/active_kv_2026-10-17.json)
contains per-request usage, timings, answers, concurrent progress samples, source
and receipt hashes, the 133-row placement comparison (43 changed rows), numerical
oracles and restoration identity. The direct production GPU pool is unchanged.
