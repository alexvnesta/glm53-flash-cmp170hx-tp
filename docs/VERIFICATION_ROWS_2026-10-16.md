# Four-agent verification and the 64-row candidate

<!-- BEGIN_RESULT -->
**Result:** retain the 32-row kernel. Expanding capacity to 64 rows changes four-agent delivery by **-0.8% at low effort** and **+1.6% at high effort**, missing the prespecified gate. All 384 scored tasks and 32 warmups pass. Four-active-sequence runtime qualification is retained separately.
<!-- END_RESULT -->

## Measured target

A fresh 12-second Nsight Systems 2026.1.3 capture records sustained traffic on
the qualified four-active-sequence row32 profile. The capture sits inside a
48-request high-effort run, after separate low/high warmups. It uses the same
engine, native binary and service configuration as the unprofiled control.
Diagnostic timing is excluded from the throughput comparison.

The row-count annotation wraps the actual `Generator.iterate_gen` call and
records runnable sequence count multiplied by the actual verification width.
It accounts for all 55 captured target-verification rounds:

| Target rows | Sequences × verification width | Rounds | Routed MoE path |
|---:|---|---:|---|
| 24 | 3 × 8 | 7 | Cooperative |
| 32 | 4 × 8 | 23 | Cooperative |
| 32 | 2 × 16 | 1 | Cooperative |
| 48 | 3 × 16 | 2 | Generic |
| 64 | 4 × 16 | 22 | Generic |

Widths 8 and 16 correspond to K7 DFlash2 and hybrid-15 verification, including
the next target token. Thus 24/55 verification rounds exceed the compiled
32-row cooperative capacity.

| Verification GPU category | Summed GPU seconds | Share |
|---|---:|---:|
| Generic MoE | 4.848 | 30.7% |
| Cooperative MoE | 3.864 | 24.5% |
| NCCL | 1.707 | 10.8% |
| Native collectives | 2.068 | 13.1% |
| Other | 3.314 | 21.0% |

Durations sum across both GPUs and can overlap. They are not recoverable wall
time or a predicted speedup. Native collective kernels can include peer waiting.

## Candidate

The isolated variant changes three engine files: the native MLP capacity rises
from 32 to 64, cooperative routed-slot capacity rises from 256 to 512, and both
expert-sort keys use nine slot-index bits instead of eight. Increasing the
bound alone would alias sort keys above slot 255. A static assertion requires
at least one sorting thread per supported slot.

This is a global small-row capacity change, so it also changes eligible short
prefill and shared/dense-MLP dispatch. It is not restricted to four-agent
verification calls. Model output lengths can differ even when isolated
common-shape kernel outputs agree; the comparison retains those lengths.

The MMA tile, staged decoder, prefetch, accumulator folding and gate/up/down
arithmetic remain unchanged. Compiled hot 3-bit gate/up and down kernels retain
64 registers and the same static shared-memory/stack requirements. The rotation
kernel retains 36 registers; its static shared memory rises from 1,104 to 2,128
bytes. These resource checks do not establish a serving speed gain.

The CPU-only build rebuilds all 15 affected translation units: three C++ units,
the cooperative launcher, and all eleven cooperative CUDA instantiation units.
It retains 176 other objects only after verifying their attested bytes. CUDA
stays uninitialized during preparation/import and CPU contract checks.

The candidate native SHA is
`9776be77b0b22227be5f0efa655275409e6a610a963cf38dd89d88c8c09f4c55`;
the control is
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`.

## Comparison method

Both profiles have four active slots, the same weights and Weightless alpha 2,
K7/hybrid-15, Q8 target cache, FP16 recurrent state, fair4/512 and the same cache
and context budgets. Prompts, correctness oracles, temperature zero, 768-token
output budget and low/high efforts remain frozen. The 24-task fixture SHA is
`38b5c9340800821e1ba5cf10f997c3831f5cc2858f23c5008b14ad916fb05ee7`.

Each profile runs one- and two-agent cells once, and four-agent cells twice,
at both efforts. Actual per-agent delivered tokens are divided by the common
cell wall time. Aggregate rates pool token totals and wall time across repeats;
time per 24-task pack is the mean repeat wall time. Reasoning tokens count;
SSE chunks do not. Output lengths are reported separately.

CPU PSI `some avg10` is sampled every second. Cells reaching 2% are retained
as excluded timing attempts and can be retried at most twice. A prespecified
selection gate requires at least 2% more high-effort four-agent throughput,
no low-effort four-agent regression, and other cells within 2% of control.
Every scored answer must pass. Numerical, functional, runtime and production
canary checks remain separate gates.

<!-- BEGIN_MEASURED -->
| Effort | Agents | Control tok/s | Candidate tok/s | Change | Control seconds/24 | Candidate seconds/24 |
|---|---:|---:|---:|---:|---:|---:|
| low | 1 | 123.16 | 122.47 | -0.56% | 49.69 | 45.32 |
| low | 2 | 137.54 | 136.72 | -0.60% | 44.50 | 40.59 |
| low | 4 | 156.26 | 155.01 | -0.80% | 38.58 | 35.83 |
| high | 1 | 113.73 | 111.25 | -2.19% | 51.07 | 52.69 |
| high | 2 | 125.21 | 128.48 | +2.61% | 47.93 | 50.48 |
| high | 4 | 144.06 | 146.39 | +1.62% | 41.30 | 40.49 |

Actual per-agent delivered rates (tok/s, in agent-index order):

| Effort | Agents | Control | Candidate |
|---|---:|---|---|
| low | 1 | 123.16 | 122.47 |
| low | 2 | 70.56 / 66.98 | 71.05 / 65.68 |
| low | 4 | 41.48 / 37.74 / 36.37 / 40.67 | 36.45 / 43.12 / 33.46 / 41.97 |
| high | 1 | 113.73 | 111.25 |
| high | 2 | 65.69 / 59.52 | 67.73 / 60.75 |
| high | 4 | 46.05 / 22.88 / 37.50 / 37.63 | 39.62 / 30.45 / 34.18 / 42.14 |

All **384 scored tasks and 32 warmups pass**. All primary timing attempts satisfy the CPU-PSI guard; none are excluded. Every primary cell restores 1,536 cached prompt tokens in both profiles. The 133 target placement-table rows, including expert and vocabulary-head allocations, match exactly between profiles.

Four-agent low: output totals change from **6028 to 5554 tokens per 24 tasks** (-7.86%). Time changes by -7.12%, while delivered tok/s changes by -0.80%. The shorter output must be considered when interpreting completion time.

Four-agent high: output totals change from **5950 to 5927 tokens per 24 tasks** (-0.39%). Time changes by -1.97%, while delivered tok/s changes by +1.62%. The shorter output must be considered when interpreting completion time.

The 64-row candidate passes 52 direct GPU numerical cases and 30 full-width dense-reference/hash cases on the two cards; all 22 common control/candidate outputs are bit-identical. Four-way coding, cancellation and reuse also pass. The variant fails the speed gate: four-agent high effort gains less than 2%, low effort regresses, and serial high effort falls more than 2%. A long-context/soak/production qualification of this rejected native is not claimed.

The supplemental two-GPU 512-slot all-local numerical check also passes. Production is accepted on the retained row32 native with **four active sequences** after the separate [long-context/soak/recovery qualification](FOUR_SLOT_RUNTIME_2026-10-16.md), router-facing cancellation/reuse and 48-task canary, plus another process restart and paired-cache recovery. The general launcher remains at two slots.

[Complete synthetic comparison receipts](../experiments/concurrent_throughput/verification_rows_2026-10-16.json) retain every scored request, token total, per-agent rate, timing guard, warmup, source/build hash, numerical check and trace category.

<!-- END_MEASURED -->

## Reproduction

`experiments/concurrent_throughput/prepare_row64.py` requires an attested row32
source tree and creates a separate candidate. `build_row64.py` also requires
the original row32 object's build-custody receipt and native object store; a
four-slot clone containing only the shared library is insufficient. Neither
tool deploys a service.

`profile_rows_bootstrap.py` adds the row-count NVTX ranges around an API run
under Nsight. Use the attested native/engine import environment. A valid capture
must fall inside sustained traffic after model load and warmup.
`analyze_trace.py` reads the SQLite export without modifying it and retains
the row groups separately from whole generator scopes.

The [four-slot runtime report](FOUR_SLOT_RUNTIME_2026-10-16.md) and
[RAM capacity notes](RAM_KV_CAPACITY.md) distinguish active GPU context from
retained RAM/disk histories.
