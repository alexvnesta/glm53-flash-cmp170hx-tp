# Routed gate/up warp-layout experiment on two CMP170HX cards

Outcome: **retain the qualified row32 production layout**. The alternative
passes the bounded numerical and functional checks but fails the throughput
gate. It remains an isolated experiment.

The experiment changes the warp layout of the largest individual kernel in the
existing production trace: the SM80 3-bit/mul1 wide routed gate/up kernel.
The original source, codebook decoder, FP16-to-FP32 fold cadence, routing,
down kernel, 32-row capacity, two active request slots and TP communication
policy are retained.

The baseline uses four warps along output columns and four along the input
reduction, with two 16-column tiles per warp. The candidate uses eight warps
along columns and two along the reduction, with one 16-column tile per warp.
Both use 512 threads, eight rows per expert run and 128 output columns per
block. The reduction has two partials instead of four. This changes floating
point grouping, so bit equality with the control is measured separately from
the unchanged upstream numerical tolerance.

This targets register use, reduction overhead and the instruction working set.
It does not assume that lower register use necessarily improves occupancy or
serving speed. NVIDIA documents the register, block-size and residency tradeoff
in its [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy).
The production trace and preceding negative experiments are preserved in the
[MoE profile](MOE_COMMUNICATION_2026-10-16.md),
[staging/prefetch](ROUTED_MOE_2026-10-16.md) and
[decoder](DECODE_INSTRUCTIONS_2026-10-16.md) reports.

## Comparison contract

The existing `sustained.py`, `measure.py` and `code_oracle.py` fixture remains
byte-for-byte unchanged. It submits the same 24 tasks at temperature zero,
768 completion tokens, separate low/high effort, and one/two/four submitting
agents. Four agents share two active request slots. Immediate backfill,
validation, prefill and queueing remain included in the workload time.

Aggregate tok/s counts actual API completion tokens, including reasoning,
divided by full workload wall time. Per-agent tok/s uses each agent's actual
tokens divided by that same wall time; those rates sum to the aggregate.
High-effort two/four-agent cells repeat twice per profile. Both profiles receive
identical eight-request low/high warmups before scoring. Output lengths, failed
scores, per-request results and prefix-cache reuse are retained.

A serving gain requires at least 2% pooled high-effort improvement at both two
and four agents, no low-effort concurrent regression and serial throughput
within 2%. Numerical, JSON and cancellation checks must pass. A qualifying
candidate then requires a production canary, soak and cache-restart recovery.

## Source and build custody

`prepare_routed_moe.py --mode retile` validates the qualified source manifest
before creating an isolated tree. Only the K3 translation unit includes a new
derived header. The altered layout is guarded to SM80, integer 3-bit, mul1,
wide gate/up calls. Other bitrates, codebooks, narrow calls and down calls
retain their existing layout. The original shared-memory reservation stays
conservative; no global environment override is added.

An initial draft kept the old per-warp shared-reduction column stride. The
standard GPU oracle rejected it before any serving measurement. That binary
and its failed receipt remain isolated; production was restored before the
corrected candidate's window. The corrected layout uses the selected per-warp
column count for both addressing and computation. No oracle or tolerance was
relaxed.

The corrected CPU-only build hides GPUs, rebuilds one CUDA object and reuses
190 byte-attested objects. The corrected native SHA is
`0e53d84c435de0f11cd560bb1229b06f7d9260c0123e7d5f63cb982204964de6`.
Its 973-file source manifest reproduces exactly; the derived-header SHA is
`7d955fef374ad4509fd63791fc310c5ca2654b312624cd5e97b0b6d342460d73`.
The production launcher is unchanged.

## Hardware screen

Both binaries pass 34 standard GPU cases and 22 additional full-width routed
cases spanning both cards and rows 1/2/4/7/8/9/16/17/24/31/32. The dense
reference retains its original 3e-3 relative-to-maximum tolerance and repeated
bit reproducibility checks. Six of the 22 output hashes match between binaries;
the other 16 differ with the changed reduction grouping while passing the same
reference check. This is bounded numerical evidence.

The isolated Nsight Compute capture uses 32 rows, 146 local experts from 288,
127 selected local slots and GLM widths, on GPU0. Clock and cache controls are
disabled. The hardware capture is separate from unprofiled serving traffic.

| Metric | Control | Retiled A |
|---|---:|---:|
| Registers/thread | 64 | 60 |
| Register-limited resident blocks | 2 | 2 |
| Active-warp occupancy | 49.55% | 49.66% |
| Instruction issue active | 77.99% | 73.45% |
| Eligible warps/active cycle | 2.61 | 1.93 |
| DRAM throughput/peak | 17.10% | 16.73% |
| Captured A-kernel duration | 1.7448 ms | 1.7862 ms |

The candidate cuts register use without crossing a residency threshold. Lower
eligible-warp and issue rates are consistent with losing instruction-level
parallelism when each warp has fewer independent output tiles and a longer
input loop. That is an interpretation of this isolated capture, not proof of
full-model stall attribution. The captured duration is 2.4% longer; kernel
milliseconds are not converted to estimated server token rates.

Current upstream dev was checked at
[`bd4da327`](https://github.com/turboderp-org/exllamav3/commit/bd4da327a820e3af602b9131c54a41a5736c33ae).
Its cooperative MoE header has exactly the qualified header's SHA256
`bdd49840d121bc9d298de296f4ed975dd25a7b5c4331363ace0f5fbbda3917ed`.
The experiment is therefore a local alternative to the current upstream layout.

## Shared-host timing observation

The original control high-effort repeats overlapped a separate CPU test job
that started at 19:41:11 UTC, after all six initial control cells had finished.
The repeated two-agent rate fell from 122.90 to 118.90 tok/s; four agents fell
from 122.00 to 109.39. CPU pressure and independent process snapshots supplied
evidence of host contention. The test job and other clients were left running.

These two repeat timings were flagged before candidate scoring began. Their
48 request records and all scores remain in the receipts. They do not enter
the primary pooled rates. The same two high-effort cells are repeated on the
restored production backend after identical warmup, with the exact control
profile digest, native SHA and benchmark fixture. This introduces no prompt,
effort, budget or validator change. The replay uses the backend's direct port
8014; the isolated arms use direct port 8013.

Candidate scoring also has read-only host telemetry. The first part overlaps
the drain of the external job; CPU pressure then declines. The results describe
this shared-host run. Small differences cannot be attributed entirely to the
kernel. Production promotion still requires the stated correctness and speed
gates, followed by a separate soak and recovery qualification.

## Serving results

All **384 requests in the valid comparison passed**, with another **48
CPU-confounded control requests retained and correctly scored**. All 48
warmup requests passed separately. Each measured cell restored exactly 1,536
cached prompt tokens. High-effort concurrent rates pool the two valid repeats
using total tokens divided by total time, rather than averaging rates.

| Effort | Agents | Control aggregate tok/s | Retiled A aggregate tok/s | Change | Control seconds/24 tasks | Retiled seconds/24 tasks |
|---|---:|---:|---:|---:|---:|---:|
| Low | 1 | 121.6 | 111.1 | -8.6% | 50.34 | 55.08 |
| Low | 2 | 133.4 | 127.5 | -4.4% | 45.87 | 48.00 |
| Low | 4 | 132.1 | 128.2 | -2.9% | 46.36 | 47.74 |
| High | 1 | 112.4 | 105.2 | -6.4% | 51.68 | 55.19 |
| High | 2 | 123.9 | 118.7 | -4.2% | 48.43 | 50.04 |
| High | 4 | 123.1 | 118.1 | -4.1% | 49.39 | 50.87 |

| Effort | Agents | Control actual per-agent tok/s | Retiled A actual per-agent tok/s |
|---|---:|---|---|
| Low | 1 | 121.6 | 111.1 |
| Low | 2 | 68.5 / 65.0 | 65.4 / 62.1 |
| Low | 4 | 20.4 / 37.7 / 35.4 / 38.6 | 19.8 / 36.6 / 34.4 / 37.5 |
| High | 1 | 112.4 | 105.2 |
| High | 2 | 65.0 / 58.9 | 62.4 / 56.3 |
| High | 4 | 20.9 / 23.8 / 41.5 / 36.9 | 32.2 / 32.9 / 23.0 / 29.9 |

Agents draw from a common queue. Different task assignments and answer
lengths affect individual rates; an uneven split alone does not show unfair
scheduling. High-effort output averaged 6,002 control versus 5,941 candidate
tokens for two agents, and 6,078 versus 6,006 for four agents. Full-work elapsed
time and output counts remain visible alongside tok/s. These are observed
end-to-end rates under the documented shared-host conditions, not a pure
kernel causal speed estimate.

The restored control repeats delivered **125.01 tok/s** with two agents and
**124.15 tok/s** with four, at low CPU pressure. Candidate repeats delivered
118.56/118.87 and 117.96/118.16 respectively. The flagged original control
repeats are preserved at 118.90 and 109.39; none of their request scores was
deleted or repaired.

## Whole-operation CUDA-graph screen

This unchanged screening tool times rotation, gate/up and down together, with
32 local experts selected from 64, rather than just kernel A. Each median uses
five sets of 100 replays. It is separate from the 146-expert counter capture
and is not converted into server token rates.

| GPU | Routed rows | Control median, ms | Retiled A median, ms |
|---|---:|---:|---:|
| 0 | 8 | 0.7362 | 0.7690 |
| 0 | 16 | 0.9347 | 1.0623 |
| 0 | 32 | 1.0926 | 1.2351 |
| 1 | 8 | 0.6318 | 0.7700 |
| 1 | 16 | 0.9228 | 1.0504 |
| 1 | 32 | 1.0747 | 1.2211 |

## Checks and production disposition

The corrected candidate passes nine exact JSON-object cases, a mixed-filter
wave, cancellation with a surviving request and replacement-slot reuse, and
three high-effort coding pairs. The unchanged coding oracle still checks
behavior and nonmutation. CPU validation passes 38 tests on Linux, with 37
passes and one platform skip on macOS; HTTP checks pass 44 tests and the
response-lifetime AST suite passes four. The pinned base source was reproduced
before local HTTP checks.

Independent final inspection at `2026-10-16T19:57:52.054005+00:00` verifies the
full production source manifest, native binary, sole-unit GPU ownership,
healthy idle state, empty requests/queue/leases, Weightless alpha 2 on both
devices, `NRestarts=0` and inactive Qwen. The retained native SHA is
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046` and profile digest is
`2bfd7ca630f488034cfc1be093317719aeb41e227a8e8766a3e2c856e637c91c`.

The candidate fails the established speed gate, so it is **not promoted**.
No new canary, soak or cache-restart qualification is claimed for it. The
production service, default launcher and communication policy retain the
qualified row32 build. Experiment processes are bounded and restoration
receipts are retained.

[Complete synthetic receipts](../experiments/concurrent_throughput/moe_retile_2026-10-16.json)
include per-request answers/scores, warmups, both valid and flagged timings,
actual per-agent delivery, prefix-cache counts, batching counters, numerical
hashes, native/source identities and hardware metrics. Raw controller logs,
Nsight reports and process/pressure snapshots remain in the dated private lab.
