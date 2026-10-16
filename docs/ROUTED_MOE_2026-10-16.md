# Targeted routed-MoE experiments on two CMP170HX cards

Outcome: **retain the qualified row32 production build**. The staged decoder
did not produce a consistent serving gain; deeper prefetch failed its kernel
screen. Both variants remain isolated experiments.

The control is the qualified row32 GLM-5.3-Flash EXL3 deployment with Weightless
alpha 2, DFlash2 K7/hybrid 15 and two active slots. The frozen 24-request
structured/code workload, prompts, effort settings, temperature, 768-token
budgets and correctness oracles are unchanged from the
[previous comparison](MOE_COMMUNICATION_2026-10-16.md).

## Largest measured cost

The existing 12-second trace of the exact current production build identifies
the wide, mul1-codebook 3-bit routed **gate/up kernel** as the largest individual
kernel: 4.721 seconds of summed GPU duration across both cards, 26.7% of all
kernel time. Its down kernel accounts for 2.579 seconds, 14.6%. Together these
are about 41% of summed kernel duration; all cooperative MoE kernels account
for about 53% inside verification ranges. No verification range uses the
generic MoE fallback. These sums are not wall time or directly recoverable
speedup. Full trace context and the original kernel CSV remain in the previous
report and its receipts.

This identifies the largest timed kernel, rather than proving whether its
limiting hardware resource is memory bandwidth, instruction issue or latency.
Nsight Compute 2026.2.1 was found outside `PATH`. The driver restricts counters
to administrators, so hardware capture uses an existing administrator launch
during a separate exclusive screen, with clock/cache controls disabled and
no persistent permission changes. These counters describe a synthetic full-width
routed launch, rather than the full production workload. The experiments test
specific hypotheses about its weight-loading and decoding loop.

## Two targeted candidates

The pinned [upstream kernel](https://github.com/turboderp-org/exllamav3/blob/0f685aa9fda3a98a864149b367b7ef570257572e/exllamav3/exllamav3_ext/quant/exl3_moe_coop_kernel.cuh)
has register and shared-memory staged decoders. The 3-bit staged path is
selected on SM86; these SM80 cards use register shuffles. Its prefetch ring
normally has four entries. Current upstream `dev` was inspected and retains
those policies.

Both candidates apply only to the **SM80, 3-bit, mul1 gate/up path**:

- **Staged A** uses the existing staged weight decoder to test the shuffle/
  shared-memory tradeoff on this architecture.
- **Prefetch8 A** retains the register decoder and increases its prefetch ring
  from four to eight entries to test greater latency hiding.

The accumulation fold cadence stays four. Routing, reductions, activations,
other bitrates/codebooks, down kernels, shared experts, TP communication and
serving settings retain the existing implementation. The batch limit stays 32.
Neither candidate adds a global environment override.

Each variant preserves the original header and adds an attested derived header
included only by the K3 translation unit. The source preparation tool validates
all anchors before creating a new destination. It produces a 973-file manifest
from the qualified 972-file engine. The derived header differs only at the
gate/up template selector and the selected decoder/prefetch policy. Reproducing
the preparation yields the exact measured derived-header hashes.

Each CPU-only incremental build hid GPUs, rebuilt one CUDA object and reused
190 byte-attested objects. Builds took about 23 seconds each. The prefetch build
completed before scored measurement began; no compiler ran during scored
traffic. The native import check confirmed row32 and no CUDA initialization.
A service-copy helper was updated to handle an added source without an old
pin hash; the initial preparatory failure happened before any GPU window.

## Numerical and timing screen

The standard 34-case upstream oracle runs on each binary. An additional 22
full-width routed cases cover both cards and rows 1/2/4/7/8/9/16/17/24/31/32,
with 4096/2048 widths, 3-bit mul1 projections, eight expert selections,
foreign-expert masking, empty rows, FP32 gate/up scratch and clipping. The
upstream dense/reconstruct reference keeps its original 3e-3 relative-to-maximum
tolerance and repeated-output bit checks. An AST return exposes the already
checked output; SHA256 of its raw FP32 bytes is compared between binaries.

Staged A passed the standard oracle and all 22 full-width cases, whose outputs
matched the control bit for bit. This is bounded numerical evidence, not a
claim of equivalence across every input and model checkpoint.

The initial CUDA-graph timing screen uses the unchanged `moe_micro.py`: five
sets of 100 replays per shape, both GPUs, GLM widths, routed 3-bit and shared
5-bit cases at 8/16/32 rows. Staged A made routed 16/32-row timings about
0.8–1.8% slower. Its eight-row result varied between cards, including a noisy
control on GPU0; no consistent win is inferred. Shared cases differed by less
than 0.3%, consistent with leaving their implementation unchanged. Kernel
milliseconds are never converted into claimed server tokens per second.

Prefetch8 receives a separate exclusive screen after production restoration,
with a fresh control before and after. Serving qualification proceeds only if
all four routed 16/32-row device cases show at least 3% higher micro throughput.
This screen gate avoids treating a noisy eight-row timing as a serving win.

## Serving measurement contract

The full Staged A comparison uses one, two and four submitting agents at low
and high effort, with immediate backfill. Each profile receives identical
8-request low/high warmups; warmup data stays separate. High-effort two/four
agent cells repeat. Four agents share two active GPU request slots.

Aggregate tok/s is actual API completion tokens, including reasoning, divided
by the complete workload wall time. Each agent's own tokens use that same wall
time, so per-agent rates sum to the aggregate. Sampled generated-token counters
also give a steady-window rate, dropping five seconds from each end. Queueing,
prefill and client validation remain included in full-work rates. Task scores,
output lengths, elapsed time and prefix-cache counts remain visible. Failed
answers stay failed and are never repaired or removed.

Promotion requires at least 2% higher pooled high-effort throughput for both
two and four agents, no low-effort concurrent regression and at most 2% serial
jitter, followed by lifecycle checks and a guarded production soak/cache-restart
canary. A microbenchmark gain or successful native build alone is insufficient.


## Matched serving results: staged gate/up decoder

All **384 scored requests passed**, 192 per profile. All 32 warmup requests
passed separately. Every measured cell restored exactly **1,536 cached prompt
tokens**. Output lengths and task assignments can vary despite temperature zero:
low-effort four-agent output totaled 6,123 control versus 6,121 candidate tokens;
high-effort four-agent repeats averaged 6,040 versus 6,078. The complete output
and scoring records remain intact. Rates pool actual tokens over total elapsed
time for repeated high-effort cells. Low-effort and serial cells have one run.

| Effort | Agents | Control aggregate tok/s | Staged A aggregate tok/s | Change | Control seconds/24 tasks | Staged seconds/24 tasks |
|---|---:|---:|---:|---:|---:|---:|
| Low | 1 | 122.0 | 121.6 | -0.4% | 50.16 | 50.35 |
| Low | 2 | 137.2 | 136.7 | -0.3% | 44.63 | 44.77 |
| Low | 4 | 136.4 | 135.5 | -0.6% | 44.91 | 45.18 |
| High | 1 | 112.9 | 112.0 | -0.8% | 51.46 | 51.85 |
| High | 2 | 124.2 | 124.8 | +0.5% | 48.32 | 48.09 |
| High | 4 | 124.8 | 124.1 | -0.5% | 48.40 | 48.96 |

| Effort | Agents | Control per-agent tok/s | Staged A per-agent tok/s |
|---|---:|---|---|
| Low | 1 | 122.0 | 121.6 |
| Low | 2 | 70.4 / 66.8 | 70.1 / 66.6 |
| Low | 4 | 21.0 / 38.9 / 36.6 / 39.9 | 20.9 / 35.9 / 36.3 / 42.4 |
| High | 1 | 112.9 | 112.0 |
| High | 2 | 65.2 / 59.0 | 65.5 / 59.3 |
| High | 4 | 25.2 / 29.6 / 30.9 / 39.1 | 21.1 / 24.0 / 41.8 / 37.2 |

The candidate's high-effort two-agent runs delivered 124.12/125.53 tok/s;
control delivered 123.79/124.63. Four-agent candidate runs delivered
124.26/124.03 versus 124.58/125.02 for control. Pooled changes of +0.5% and
-0.5% do not meet the 2% gate. Low-effort concurrency was -0.3%/-0.6%.
Unequal per-agent rates also reflect different assigned tasks and output
lengths; they do not alone establish a fairness issue.

## Prefetch screen and hardware evidence

Prefetch8 passed the standard 34-case oracle and all 22 full-width cases,
matching the control's output bytes. The fresh control-before/candidate/
control-after timing screen rejected it:

| GPU | Routed rows | Control mean of bracketing medians, ms | Prefetch8 median, ms | Duration increase |
|---|---:|---:|---:|---:|
| 0 | 16 | 0.9247 | 1.1327 | +22.5% |
| 0 | 32 | 1.0885 | 1.3447 | +23.5% |
| 1 | 16 | 0.9258 | 1.1235 | +21.3% |
| 1 | 32 | 1.0741 | 1.3330 | +24.1% |

Prefetch8 did not proceed to a serving benchmark. Its kernel timings are not
converted into estimated token rates.

Nsight Compute captured one warmed GPU0 launch per binary, each replayed 15
passes. The synthetic case uses GLM widths, row32, 146 local / 288 global
experts, eight choices and 127 selected local slots. It uses random EXL3
projections, rather than production weights or recorded production routing.
Cache flushing and clock forcing were disabled; the profiler warns about
possible variability. Captures were separate from all scored serving traffic.

| Metric | Control | Staged A | Prefetch8 A |
|---|---:|---:|---:|
| Registers/thread | 64 | 64 | 90 |
| Register residency limit, blocks/SM | 2 | 2 | 1 |
| Achieved active-warps occupancy | 49.6% | 49.6% | 25.0% |
| SM throughput / issue-active, elapsed | 78.9% | 77.3% | 53.3% |
| DRAM throughput, percent of peak | 18.7% | 19.9% | 14.7% |
| Eligible warps/scheduler/cycle | 2.67 | 2.60 | 0.93 |

All three captures report zero local/shared spilling requests. The deeper
prefetch ring crosses a register-residency threshold: two blocks of
512 threads × 90 registers cannot fit in the SM register budget, while two
64-register blocks can. Occupancy and eligible-warps measurements support this
explanation for the regression. Extra prefetch did not fix a spill problem.

For the control capture, `sm__throughput` is dominated by the issue-active
metric, rather than indicating 79% tensor-core FLOP utilization. Combined with
the modest DRAM throughput, it points toward reducing instruction work while
preserving occupancy. Memory-latency stalls remain present; this does not
prove that every production route is exclusively compute-bound. It also does
not justify a deeper prefetch queue or predict a server-level gain from
synthetic counters alone.

The [NVIDIA CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
documents push/pop NVTX range selection and replay. `routed_profile.py` marks
only the intended warmed launch. Its direct MoE kernels have no cross-process
collective dependency. Driver permission settings remain unchanged.

## Qualification, custody and restoration

Each of the three binaries passed 34 standard and 22 full-width numerical
cases. Staged A additionally passed nine exact JSON-object checks across
effort levels, mixed-filter isolation, cancellation with a survivor and slot
reuse, and three high-effort coding pairs. The Linux CPU probe suite passed
31 tests, the pinned HTTP suite passed 44 and complete-response AST checks
passed four. These are bounded correctness checks, not general coding-quality
or checkpoint-equivalence proof.

Neither candidate passed its promotion gate. No new production soak or
persisted-cache restart qualification was run for them. Production was restored
from the first comparison, then again after the separate screen/counter window
at **18:26 UTC**. An independent read-only inspection verified the full original
engine manifest, native SHA and profile digest, healthy idle state, no active
requests/queue/native leases, sole production-unit GPU ownership, zero service
crash restarts, and Qwen inactive.

The retained native SHA is
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`;
the profile digest remains
`2bfd7ca630f488034cfc1be093317719aeb41e227a8e8766a3e2c856e637c91c`.
Candidate binary/source hashes, build custody, all scored answers, sampled
counters, per-agent rates, warmups, cache hits, raw hardware metrics and
private-receipt digests are retained in the
[complete synthetic receipts](../experiments/concurrent_throughput/routed_moe_2026-10-16.json).
Raw Nsight reports and operator logs remain in private host receipts.

Reproduction tools in `experiments/concurrent_throughput/` are
`prepare_routed_moe.py` (`--mode staged` or `--mode prefetch8`),
`build_routed_moe.py`, `routed_numerics.py`, and `routed_profile.py`. The preparer
requires the attested row32 engine; the builder runs with GPUs hidden and
retains all other object hashes. GPU tools refuse an occupied pair. Keep the
qualified binary, service override and cache directories for restoration;
preparation is never a deployment action.
