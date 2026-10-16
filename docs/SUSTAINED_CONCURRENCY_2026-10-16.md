# Sustained ExLlamaV3 traffic on two CMP170HX cards

> Matched benchmarks are complete. The guarded production canary is in progress.

This follow-up measures immediate request backfill rather than isolated pairs.
The model remains GLM-5.3-Flash EXL3 with Weightless alpha 2, TP2, DFlash2 K7,
hybrid drafting up to 15, explicit JSON-object support, and two active slots.
Four submitting agents keep two more requests queued; this is not four active
GPU sequences. The original qualified row16 profile is the control.

## Work and rate definitions

Each cell submits the same 24 requests: six repetitions of four short tasks,
two exact integer arrays, a Python `compress_runs` implementation, and an exact
JSON object copy. Explicit JSON mode applies to the code and object requests.
The code uses the original six-case behavioral and nonmutation oracle. Arrays
and objects require exact equality. Temperature is zero and each request has
the same 768-completion-token budget. Low and high effort are separate cells.
The worker immediately takes the next task after validating its previous one.

The client is [sustained.py](../experiments/concurrent_throughput/sustained.py).
It records complete responses, usage, task scores, request timings, observed
active/waiting counts and native batch counters. Foreign traffic or identity
drift invalidates the run. Known-answer failures stay failed; no answer repair
or prompt changes are allowed during comparison.

- **Aggregate tok/s:** all API completion tokens divided by the complete
  workload's wall time, including prefill, queueing and client validation.
- **Per-agent tok/s:** that agent's completion tokens divided by the same wall
  time. These rates sum to the aggregate. Different job mixes can give agents
  different rates, so a rate difference alone is not a fairness diagnosis.
- **Steady tok/s:** differences in actual generated-token counters between
  health samples, dropping the first and last five seconds. These are token
  counters, not SSE events or text chunks.

Completion tokens include reasoning. Reasoning/code lengths can differ even
when every answer passes, so both fixed-work completion time and token rates
are retained. These fixtures test serving and a small correctness contract;
they do not represent a general coding-quality benchmark or long-context
agent workflow. Prompts are short; prefix caching remains enabled as in
production. This comparison does not attribute a cache-reuse benefit to row32.

## Matched results

All **384 comparison requests passed** their original task oracle. Each
profile completed 192 requests. High-effort two- and four-agent cells have two
24-request runs per profile; other cells have one. The control ran before and
after the candidate. Repeated rates pool actual tokens over total elapsed
time, rather than averaging per-request speeds. Both profiles used the same
direct backend endpoint and identical fixtures, flags and two active slots.
Every two- and four-agent run in both profiles restored the same **1,536
prompt tokens**: six 256-token hits on the literal-copy prompt. Low-effort
single-agent runs restored 1,280 tokens each. High-effort single-agent control
restored 1,536 versus the candidate's 1,280, an extra initial prefix miss on
the candidate. Retain that cell's observed timing; its -0.9% change is not a
clean measurement of an intrinsic serial regression. The primary concurrent
before/after comparison has matched cache-hit counts.

| Effort | Submitting agents | Row16 aggregate tok/s | Row32 aggregate tok/s | Row32 per-agent tok/s | Change |
|---|---:|---:|---:|---|---:|
| Low | 1 | 120.8 | 121.1 | 121.1 | +0.2% |
| Low | 2 | 134.0 | 136.5 | 70.0 / 66.5 | +1.8% |
| Low | 4 | 132.6 | 137.7 | 21.2 / 39.3 / 36.9 / 40.2 | +3.8% |
| High | 1 | 112.1 | 111.1 | 111.1 | -0.9% |
| High | 2 | 121.5 | 125.4 | 65.8 / 59.6 | +3.2% |
| High | 4 | 121.6 | 124.8 | 21.2 / 24.1 / 42.1 / 37.4 | +2.6% |

| Effort | Agents | Row16 seconds per 24 tasks | Row32 seconds per 24 tasks | Row32 steady tok/s |
|---|---:|---:|---:|---:|
| Low | 1 | 50.65 | 50.55 | 122.0 |
| Low | 2 | 45.67 | 44.85 | 137.7 |
| Low | 4 | 46.15 | 44.48 | 137.3 |
| High | 1 | 51.80 | 52.27 | 111.5 |
| High | 2 | 49.40 | 47.85 | 126.7 |
| High | 4 | 49.97 | 48.70 | 122.8 |

Within row32, two agents raise aggregate throughput **12.7% at low effort**
and **12.9% at high effort** relative to one. Four submitting agents add a
queue, with about two waiting during steady traffic; they do not add GPU
slots or double throughput. High-effort fixed-work completion improves by
8.4% with two agents relative to one, rather than 12.9%, because reasoning
and code output lengths differ between concurrency levels.

The high-effort repeats are consistent: row16 delivered 121.65/121.36 tok/s
with two agents and 121.71/121.56 with four; row32 delivered 125.08/125.78
and 124.79/124.80 respectively. Low-effort changes have only one matched run.
This is a modest improvement on this host, not evidence of near-linear
scaling or a measured improvement in general agent-task intelligence.

[Synthetic receipts](../experiments/concurrent_throughput/sustained_2026-10-16.json)
retain every scored answer, usage, per-agent rates for both profiles, sampled
token counters, occupancy, batch-counter deltas, fixture/source hashes and
private raw-receipt digests. The
[kernel summary](../experiments/concurrent_throughput/profile_kernels_2026-10-16.csv)
is the original profiler output. The earlier exploratory run through the
router and the instrumented trace's token rate are excluded from these tables.

## Profile and selected bottleneck

Nsight Systems 2026.1 captured 12 seconds of steady four-agent high-effort
traffic on an exact isolated copy of the row16 production engine. Loading and
warmup preceded capture. Profiling used CUDA, NVTX, OS runtime, Python GIL and
100 Hz Python sampling, with native CPU sampling disabled. No permissions or
kernel policies were changed. The full trace stays with the private operator
receipts; its throughput is excluded from the unprofiled comparison.

The capture contained 96 generator iterations:

| Scope | CPU range duration |
|---|---:|
| Target verification, including sampling | 9.644 s |
| DFlash draft generation | 1.071 s |
| Prompt prefill ranges | 0.880 s |
| Native ownership checks | 0.220 s |
| Receiving logits | 0.136 s |

These are annotated range durations, not additive CPU utilization percentages.
Inside target-verification ranges, MoE kernels accounted for 8.117 seconds of
summed GPU kernel duration across both cards, about 53% of the classified
kernel time. NCCL and native process-group kernels accounted for about 26%.
Per-device unions of all kernel intervals occupied 8.941/11.977 seconds and
8.625/11.985 seconds respectively; overlapping intervals are counted once.
Summed kernel time across cards is not wall time or a recoverable speedup.

Fifteen target-verification ranges invoked the generic MoE path. Together they
took 2.237 of the 9.644 seconds of target-verification range time. The source
explains a capacity mismatch: two hybrid-15 requests can form 32 verification
rows, exceeding the deployed fused path's 16-row capacity. The cooperative
MoE implementation already has a 256-slot bound, sufficient for GLM's
32 rows × 8 selected experts. The candidate expands the existing native
capacity to 32 rather than replacing its arithmetic or disabling hybrid drafts.

The [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
documents the trace and sampling options. Exact source dispatch is retained
in the pinned engine; the candidate changes only `MAX_BSZN` in `mlp.h` from
16 to 32. Its three dependent C++ objects are rebuilt and 188 attested objects
are reused. Python reads the extension's exported `decode_max_rows`.

The resulting 1.8–3.8% concurrent improvement addresses this capacity
mismatch. It does not remove the main cooperative MoE kernels' arithmetic
cost or tensor-parallel communication. Those remain the larger targets for
further kernel work; increasing the submitting-agent count alone does not
solve them.

## Qualification and identity

The row32 candidate passed **34 upstream numerical cases** across both cards,
including rows 1/2/4/7/8/9/16/17/24/31/32, GLM 4096/2048 dimensions, expert
masking and 3-/5-bit coverage. Reference reconstruction and the 3e-3
relative-to-maximum tolerance are unchanged. The 384-request comparison kept
every original answer and correctness score. Additional JSON qualification
passed nine exact objects across low/high/max effort, filtered/unfiltered
isolation, cancellation with a correct surviving request, replacement/slot
reuse, and six high-effort coding requests.

CPU checks passed 24 probe/preparation tests on Linux, 44 HTTP/lifecycle tests
with pinned dependencies, four response-AST tests, and pinned engine source
reproduction/verification. The Mac probe run skips the Linux resource-limit
oracle; that oracle ran on Linux. The unpinned Mac HTTP environment was not
used as a qualification environment.

Row16 native SHA-256:
`6f29d6fec981e09825fe1be463ca0f2effa4e32df24ce2108a7532e3bfb52649`.
Row32 native SHA-256:
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`.
Row32 profile digest:
`2bfd7ca630f488034cfc1be093317719aeb41e227a8e8766a3e2c856e637c91c`.

The general launcher and engine manifests remain unchanged; this is a
separately attested local variant. It retains two active slots, the Q8
655,360-token cache pool, 392,960-token context cap, DFlash2 K7/hybrid15,
Weightless alpha 2 on both devices, vision on GPU0 and the existing API.
These short synthetic measurements do not qualify a new near-limit context
pressure scenario or represent a long-running real multi-agent workflow.

## Reproduction

Use an already qualified row16 profile with its complete manifest, native
object-custody receipt and build command inventory:

```sh
python experiments/concurrent_throughput/prepare_row32.py \
  --engine /ABS/QUALIFIED_ROW16/engine \
  --manifest /ABS/QUALIFIED_ROW16/engine.sha256 --output /ABS/NEW/row32
python experiments/concurrent_throughput/build_row16.py \
  --base-profile /ABS/QUALIFIED_ROW16 --candidate /ABS/NEW/row32 \
  --python /ABS/VENV/bin/python --base-native-sha256 QUALIFIED_ROW16_NATIVE_SHA
```

The builder is host-specific and remains CPU-only. Neither command changes a
service. In an exclusive GPU window, run the unchanged numerical reference
with expanded buffers before loading the model:

```sh
python experiments/concurrent_throughput/gpu_oracles.py \
  --engine /ABS/NEW/row32/engine --native /ABS/NEW/row32/native \
  --decode-max-rows 32 --output /ABS/RECEIPTS/numerics.json --exclusive-window
```

Preparation refuses source drift. GPU qualification additionally refuses
existing model owners or an unexpected native identity. Copy the qualified service and construct an
isolated private launch plan with `prepare_server.py`; it does not start it.
After the operator starts the test profile on a separate loopback port:

```sh
python experiments/concurrent_throughput/sustained.py \
  --base-url http://127.0.0.1:8013 --agents 1 2 4 --effort low high \
  --requests 24 --output /ABS/NEW/RECEIPTS
```

Keep prompts, budget, scoring, engine flags, two-slot capacity and measurement
endpoint identical across profiles. Always stop the task-owned test service
before restoring production, and verify health, source/native identity, zero
leases and exclusive GPU ownership after restoration.
