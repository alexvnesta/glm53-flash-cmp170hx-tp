# MoE and GPU communication on two CMP170HX cards

Result: **retain the qualified row32 production profile**. The tested global
MoE overrides do not provide a suitable improvement, and the bulk-NCCL
broadcast policy fails the throughput gate. It remains an opt-in experiment.

The control is the qualified 32-row ExLlamaV3 deployment from the
[sustained-traffic report](SUSTAINED_CONCURRENCY_2026-10-16.md): GLM-5.3-Flash
EXL3, Weightless alpha 2, TP2 NCCL, DFlash2 K7, hybrid 15 and two active slots.
This follow-up screens existing cooperative-MoE geometry and tests a separate
opt-in large-broadcast implementation. It leaves the frozen task prompts,
768-token budgets, temperature and correctness oracles unchanged.

## Fresh profile

Nsight Systems 2026.1 captured 12 seconds of four-agent high-effort traffic on
an isolated exact copy of the current deployment. Loading finished 118 seconds
after launch; capture began at 150 seconds. All 48 diagnostic requests passed.
The instrumented run's throughput is excluded from the unprofiled comparison.

Across 96 verification ranges, the annotated generator spent 9.443 seconds
inside target verification, 1.079 inside drafting, and 0.946 inside prefill.
No verification range used the generic MoE fallback. Generic MoE remains in
prefill, which is expected for larger batches.

| Verification kernels | Summed seconds across both GPUs | Share |
|---|---:|---:|
| Cooperative MoE | 8.010 | 52.7% |
| NCCL collectives | 2.171 | 14.3% |
| Native process-group kernels | 1.902 | 12.5% |
| Other kernels | 3.127 | 20.6% |

These sums can overlap and exceed wall time. A communication kernel can wait
for the other GPU, so its duration is not a directly recoverable speedup.
Per-device kernel-interval unions were 9.007/11.985 seconds and 8.696/11.985
seconds. Inclusive annotated collective-call durations are retained separately
from GPU kernel durations. The analysis tool opens the SQLite export read-only.

## Cooperative-MoE screening

The pinned engine already provides `EXL3_MOE_COOP_WIDE` and
`EXL3_MOE_COOP_KSPLIT`. Before introducing new arithmetic, we replayed CUDA
kernel graphs at GLM widths 4096/2048 on each card, with 8, 16 and 32 rows.
Routed cases use random 3-bit projections, 32 local experts, eight selections,
foreign-expert masking and 256 scratch slots. Shared cases use random 5-bit
projections, one expert and 32 scratch slots. These synthetic kernel timings
are screening evidence, not server tokens per second or an exact reconstruction
of production routing. Five sets of 100 replays are retained for every case.

Narrow tiles slowed routed cases by 19–53% and shared 16/32-row cases by
16–45%, despite improving the shared eight-row case by about 27%. A global
narrow-tile override is therefore unsuitable. Forcing K-split 1 was inconsistent
or equivalent. K-split 2/4 improved some shared cases while slowing routed
16-row cases by about 11%; their apparent eight-row routed differences varied
between cards. The 32-row routed capacity caps splitting at one. No global
MoE override is selected from these results.

The default, narrow and split-1 configurations each passed the 34-case upstream
dense/reconstruct oracle at its unchanged 3e-3 relative-to-maximum tolerance.
The split-2/4 timing screen checks finite and repeated-identical outputs but is
not a dense-reference qualification. It does not authorize deploying those
settings. A useful future kernel hypothesis is a separate shared-expert policy;
this screen has not established its server-level benefit.

## Communication candidate

The [upstream NCCL backend](https://github.com/turboderp-org/exllamav3/blob/0f685aa9fda3a98a864149b367b7ef570257572e/exllamav3/model/model_tp_backend.py)
still delegates broadcasts to the native host-shared-memory backend. Its
small-message path switches at 2 KiB. The isolated candidate adds
`EXL3_TP_NCCL_BROADCAST=bulk`: payloads above that boundary use the existing
NCCL process group, with physical source devices mapped to process-group ranks.
Smaller messages retain the native path. All-reduce, its BF16 wire rounding,
gathers, model weights and native binary are unchanged.

The source preparation tool verifies the complete input manifest, refuses
in-place edits, and writes a separate engine and manifest. The default remains
`native`. The candidate's engine bytes and opt-in flag participate in the
service profile digest. No NCCL protocol, thread or channel override is added.
The [NVIDIA guidance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
cautions against carrying forced protocol tuning into normal operation.

A two-rank oracle exercises both source directions, five dtypes, sizes around
the byte threshold and larger 128 KiB/1 MiB payloads. It compares contents as
raw bytes and interleaves broadcasts with reductions. The initial oracle
attempt mistakenly included an odd-byte input unsupported by the retained
native path; the corrected test uses even sizes there. The failed attempt is
retained separately and contributes no performance data.

## Measurement contract

Each unprofiled cell uses the same 24 short structured/code tasks and immediate
backfill as the previous report. Aggregate tok/s is API completion tokens,
including reasoning, divided by the complete workload wall time. Each agent's
own tokens use that same denominator and sum to the aggregate. Four submitting
agents share two GPU slots; they keep a queue rather than four active sequences.
Actual sampled generated-token counters also give a steady-window rate.
Unequal per-agent rates can reflect different assigned tasks and output lengths;
they do not alone establish a fairness problem. Per-request latency is retained.

Both profiles receive identical eight-request low/high warmup runs before
measurement. Warmup results remain separate. High-effort two/four-agent cells
have a second run. Compare prefix-cache counts, task scores, elapsed time and
output length alongside token rate. No failed answer is repaired or removed.
This workload does not establish general coding quality or long-context gains.

Promotion uses the previous optimization gate: at least 2% higher pooled
high-effort throughput for both two and four submitting agents, no low-effort
concurrent regression, and no serial regression beyond 2% jitter. A candidate
must also retain correct scored tasks, mixed JSON isolation, cancellation and
slot reuse, then pass a guarded production canary, runtime soak and persisted
cache restart. Completing an experiment alone is not a deployment decision.


## Matched serving results

All **384 scored comparison requests passed**, 192 per profile. The 32 warmup
requests are separate and also passed. Every measured cell in both profiles
restored exactly **1,536 prompt tokens** and produced the same total completion
tokens as its matching cell. The high-effort repeated rates below pool total
tokens over total wall time. Other cells have one run. These measurements ran
on the same two cards at PCIe Gen2 x16, with no protocol overrides or changes to clock/power settings.

| Effort | Agents | Native aggregate tok/s | Bulk NCCL aggregate tok/s | Change | Native seconds/24 tasks | Bulk seconds/24 tasks |
|---|---:|---:|---:|---:|---:|---:|
| Low | 1 | 122.2 | 114.4 | -6.4% | 50.08 | 53.49 |
| Low | 2 | 137.5 | 119.8 | -12.9% | 44.51 | 51.09 |
| Low | 4 | 137.7 | 121.8 | -11.5% | 44.46 | 50.26 |
| High | 1 | 112.6 | 109.9 | -2.4% | 51.58 | 52.84 |
| High | 2 | 125.0 | 123.3 | -1.4% | 48.03 | 48.69 |
| High | 4 | 124.8 | 124.9 | +0.1% | 48.71 | 48.66 |

Actual per-agent delivery uses each agent's own tokens over the shared wall
time. It is not the aggregate divided equally by the number of agents.

| Effort | Agents | Native per-agent tok/s | Bulk NCCL per-agent tok/s |
|---|---:|---|---|
| Low | 1 | 122.2 | 114.4 |
| Low | 2 | 70.5 / 67.0 | 61.5 / 58.3 |
| Low | 4 | 21.2 / 39.3 / 36.9 / 40.3 | 18.8 / 34.8 / 32.7 / 35.6 |
| High | 1 | 112.6 | 109.9 |
| High | 2 | 65.6 / 59.4 | 64.7 / 58.6 |
| High | 4 | 21.2 / 24.1 / 42.0 / 37.4 | 21.2 / 24.2 / 42.1 / 37.4 |

The high-effort candidate repeated at 120.55/126.11 tok/s with two agents and
124.20/125.62 with four. The control repeated at 124.64/125.31 and
125.24/124.34 respectively. The candidate's pooled four-agent +0.1% is within
the run variation, while its two-agent pooled result is -1.4%. Low-effort
concurrency regressed by 11.5–12.9%. The low-effort steady-window rates also
regressed (138.97 to 120.50 tok/s with two agents; 137.28 to 120.80 with four),
so the observed low-effort difference is not confined to the dropped five-second
start/end windows. With only one low-effort run per profile, do not infer a
precise universal regression percentage.

For the retained native profile, two submitting agents deliver 12.5% more
aggregate tokens per second at low effort and 11.0% more at high effort than
one. Four submitters add a queue and essentially no further aggregate gain.
This is measured concurrency benefit, not near-linear scaling.

The result rejects this particular broadcast policy on this workload. It does
not prove that every NCCL broadcast policy is slower, that long-prefill-only
routing cannot help, or that communication accounts for 27% recoverable wall
time. Collective overhead and arrival/synchronization effects can outweigh
avoided host staging; this comparison does not isolate their individual cost.

## Receipts and reproduction

[Complete synthetic receipts](../experiments/concurrent_throughput/moe_comm_2026-10-16.json)
retain all scored answers, usage, failures, request latencies, sampled token
counters, native batching deltas, cache counts, warmups, source hashes and both
profile identities. The [kernel summary](../experiments/concurrent_throughput/moe_comm_kernels_2026-10-16.csv)
is the original Nsight output. The raw SQLite/trace and guarded operator logs
remain in the private host receipts.

Tools in `experiments/concurrent_throughput/`:

- `moe_micro.py`: exclusive kernel-graph timing screen against an attested row32 profile.
- `prepare_nccl_broadcast.py`: source-only opt-in variant preparation; never edits production.
- `broadcast_oracle.py`: exclusive two-rank byte-content and reduction-ordering checks.
- `analyze_trace.py`: read-only annotated-range and GPU interval analysis.
- `sustained.py`, `measure.py`, `code_oracle.py`: unchanged frozen serving fixture and oracle.

Preparing a candidate requires a separately qualified row32 engine/manifest.
The source-only communication patch can reuse its exact native binary and
build receipt. Use independent compilation/disk caches and a private loopback
port; never compete with the production model for the two cards. Preparation
alone does not authorize promotion, and the tested bulk policy is rejected.

The next kernel hypothesis worth isolating is the shared-expert launch policy:
the eight-row shared kernel benefits from different geometry while routed
experts regress under the same global override. Its server-level benefit is
unmeasured and its contribution is much smaller than the main routed-expert
kernels. The fresh profile supports targeted work on those kernels and
communication scheduling, rather than deploying global tuning flags based on
microbenchmark wins.

## Qualification outcome

The candidate passed the 120 rank-level broadcast content checks, nine exact
JSON objects across low/high/max effort, mixed-filter isolation, cancellation
with a surviving request and slot reuse, and three high-effort coding pairs.
The two profiles passed 384/384 scored comparison tasks and 32/32 warmup tasks.
The diagnostic profile's 48 tasks also passed. The Linux CPU probe suite passed
28 tests; the pinned HTTP suite passed 44 and the complete-response AST suite
passed four. These bounded checks do not establish broad coding quality.

The candidate failed the performance gate and was not promoted. It receives
no new production soak or persisted-cache restart qualification. The already
qualified row32 deployment remains the production and rollback target for
this experiment. Its native binary remains
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`;
the control profile digest is
`2bfd7ca630f488034cfc1be093317719aeb41e227a8e8766a3e2c856e637c91c`.
The rejected source/flag variant has digest
`a0ef28ef0e1af7eae9fa6a92e5e238c4dcf743717adb4fd8cf10340b5983460d`.


Production restoration completed at **17:15 UTC**. A separate read-only
inspection at 17:15 UTC verified the complete production engine manifest,
native SHA, profile digest, healthy idle state, no active requests/queue/native
leases, sole production-unit GPU ownership, zero service crash restarts, and
Qwen inactive. No production configuration was promoted from this experiment.
