# SM80 three-bit gate/up instruction experiments

**Outcome: retain the qualified row32 deployment.** The procedural byte-sum
change regressed serving throughput slightly. Both lookup variants failed their
isolated timing screens despite preserving occupancy. None is promoted.

The control is the qualified row32 GLM-5.3-Flash EXL3 deployment on two
CMP170HX cards: Weightless alpha 2, DFlash2 K7/hybrid 15, two active slots,
NCCL reductions and native broadcasts. The
[previous routed-MoE profile](ROUTED_MOE_2026-10-16.md) identified its 3-bit,
mul1, wide gate/up kernel as the largest individual timed kernel. Its
synthetic hardware capture showed high instruction issue activity while
retaining two resident blocks per SM. These experiments test decode changes
without increasing registers or changing accumulation.

## Same serving workload

The 24-task structured/code fixture, prompts, oracles, low/high effort,
temperature zero and 768-token budgets are unchanged. The fixture SHA is
`38b5c9340800821e1ba5cf10f997c3831f5cc2858f23c5008b14ad916fb05ee7`.
Each arm gets the same eight low- and eight high-effort warmup requests.
One/two/four submitting agents share **two active GPU slots**; four agents
include queueing. All rates below come from unprofiled serving runs.

Aggregate tok/s is actual API completion tokens, including reasoning, divided
by full workload wall time. Each agent's rate uses its own tokens divided by
that same wall time. It is not aggregate divided by agent count or an SSE
chunk count. Actual task assignments and output lengths can vary. Repeated
high-effort concurrent cells pool tokens over pooled time. Other cells have
one run. Steady counters, cache reuse, native batching deltas, every answer,
score and failure are retained in the receipts.

| Effort | Agents | Control tok/s | Byte-sum tok/s | Change | Control seconds/24 tasks | Byte-sum seconds/24 tasks |
|---|---:|---:|---:|---:|---:|---:|
| Low | 1 | 122.2 | 121.5 | -0.5% | 50.10 | 50.36 |
| Low | 2 | 135.5 | 134.7 | -0.6% | 45.16 | 45.43 |
| Low | 4 | 137.4 | 136.0 | -1.0% | 44.57 | 45.03 |
| High | 1 | 112.5 | 111.9 | -0.6% | 51.63 | 51.93 |
| High | 2 | 125.2 | 123.8 | -1.1% | 47.96 | 48.49 |
| High | 4 | 124.3 | 123.5 | -0.7% | 48.88 | 49.20 |

| Effort | Agents | Control actual per-agent tok/s | Byte-sum actual per-agent tok/s |
|---|---:|---|---|
| Low | 1 | 122.2 | 121.5 |
| Low | 2 | 69.5 / 66.0 | 69.1 / 65.6 |
| Low | 4 | 21.2 / 39.2 / 36.8 / 40.2 | 21.0 / 38.8 / 36.5 / 39.7 |
| High | 1 | 112.5 | 111.9 |
| High | 2 | 65.7 / 59.5 | 64.9 / 58.8 |
| High | 4 | 21.1 / 24.1 / 41.9 / 37.3 | 21.0 / 23.9 / 41.6 / 37.0 |

All **384 scored tasks and 32 warmup requests passed**. Every scored cell
reused 1,536 cached prefix tokens in each arm. Output-token totals also match
within each corresponding cell in this comparison. The pooled high-effort
changes are **-1.1% at two agents and -0.7% at four**. The byte-sum change
does not qualify for production.


The performance gate requires at least 2% higher pooled high-effort throughput
at both two and four agents, no low-effort concurrent regression and no
serial regression greater than 2%, with all scores correct. A kernel-only
speedup does not satisfy this gate. Further canary, soak and restart checks
are required before promotion.

## Candidates and numerical checks

Changes apply only to **SM80, 3-bit, mul1 gate/up**. The prefetch depth remains
four and the FP16-to-FP32 fold cadence remains four. Down projections, other
bitrates/codebooks, shared experts, routing, communication and serving
configuration retain the original implementation.

- **Lookup:** precompute the fixed 16-bit codebook's 65,536 half bit patterns
  into a 128 KiB read-only device table. This removes the repeated procedural
  multiply/byte-sum/FMA decode at the cost of scattered table reads. It uses
  global read-only loads rather than divergent constant-memory accesses.
- **Vabsdiff:** keep the procedural multiply and half2 FMA, replacing DP4A byte
  sums with the native unsigned `vabsdiff4...add` form. The pinned upstream
  code retains this older form in comments and reports DP4A as faster in its
  own Ampere tests. This is a host/workload check of that alternative, not a
  claim that the upstream policy is generally wrong.
- **Lookup L1:** follow the lookup's measured cache-wait regression with a
  cache-aware variant. The SM80 register decoder never reads its reserved
  8 KiB staging allocation. Omit it only for K3/cb2 gate/up, request a 33%
  shared-memory carveout and use evict-last global table loads. At GLM's
  width, two blocks still fit in a 64 KiB shared-memory partition. The CUDA
  carveout is a preference; actual allocation and occupancy must be verified.
  The host selector refuses other GPU architectures and configures both
  kernel geometries separately for each current device.

The [NVIDIA PTX reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
documents load eviction priorities; the
[CUDA runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html)
documents per-function carveout preferences. No driver, clock or global cache
policy is modified.

Each candidate preserves the original engine headers, adds two attested
headers and changes only the K3 translation unit. The lookup-L1 selector is
derived from the original instance macro, with down-kernel code preserved.
Preparation validates source identity and anchors before creating a new
output. Native builds hide GPUs, rebuild that one CUDA object and reuse 190
byte-attested objects. No compilation runs during scored serving traffic.

An exhaustive standalone CUDA oracle compares every codebook index on both
cards. Vabsdiff checks both halves and supplies nonzero unused high bits to
verify the required 16-bit masking. Its first unmasked source draft was caught
in review and preserved without GPU testing or serving. The measured candidate
includes the masks. Lookup table generation uses exact binary arithmetic
followed by one binary16 rounding; the GPU oracle verifies its result against
the original half2 FMA rather than relying on that arithmetic argument alone.

Each tested binary also runs the original 34-case GPU oracle and 22 additional
full-width cases across both cards and rows 1/2/4/7/8/9/16/17/24/31/32. The dense
reference retains its original 3e-3 relative-to-maximum tolerance. Raw FP32
output hashes compare the control and candidates without response repair.
These checks establish bounded numerical evidence, not general model-quality
or all-input equivalence.

All four binaries (control and three candidates) passed the 34 standard and
22 full-width cases. All 22 FP32 output hashes match across them. Both exhaustive
codebook checks report zero mismatches on each card. The cache-aware variant
uses the identical verified lookup-table source. Re-running the final public
preparers reproduces each measured 974-file engine manifest exactly.

## Kernel screen and hardware evidence

The unprofiled screen uses GLM widths with random EXL3 projections. The
hardware capture uses row32, 146 local/288 global experts and eight choices,
with 127 selected local slots on GPU0. Nsight Compute 2026.2.1 captures a single
warmed gate/up launch in 15 replay passes. Cache flushing and clock forcing
are disabled; replay/clock variation limits comparisons of profiler duration.
Counters diagnose the synthetic launch and do not establish a whole-model
stall distribution or a serving token rate.

The byte-sum alternative produced roughly 1–2% longer unprofiled routed
operation duration at rows 16/32 on both cards. Static SASS for the entire wide
gate/up function totals 4,037 instructions for control versus 4,029 for the
alternative: changing instruction type barely reduces code size. This is
static code, including branches and unrolling, not a dynamic instruction count.
The first lookup totals 4,101 instructions after load/address overhead.

| Capture | Registers/thread | Register block limit | Achieved occupancy | SM issue throughput | L1 sector hit rate |
|---|---:|---:|---:|---:|---:|
| Control (byte-sum window) | 64 | 2 | 49.56% | 78.07% | 30.62% |
| Byte sum | 64 | 2 | 49.57% | 74.65% | 30.60% |
| Lookup | 64 | 2 | 49.81% | 7.36% | 36.73% |
| Lookup L1 | 64 | 2 | 49.75% | 12.43% | 74.28% |

Each lookup capture also has its own control in the receipts; control
occupancy is approximately 49.56% and issue throughput ranges 78.1–78.8%.
SM throughput here is dominated by instruction issue activity, not tensor-core
FLOP utilization. Cache hit rates cover the captured kernel rather than the
lookup accesses alone.

The first lookup capture has a 97.7% L2 hit rate but a long-scoreboard
warps-per-active-issue ratio of 49.18 versus 1.97 for its control. Its low
DRAM throughput (1.9% of peak) does not indicate saturated DRAM bandwidth.
Together with the unprofiled slowdown, this supports memory-access waiting as
the new problem. Removing procedural arithmetic did not reduce the total
instruction/address work enough to offset scattered reads.


The lookup screen finished its numerical, microtiming and counter checks. Its
serving controller was stopped during the control model's startup, before any
scored serving traffic. No server tok/s is inferred for that rejected variant.
Its artifacts and the independent production-restoration receipt are kept.
Two controller path errors occurred before GPU maintenance and were corrected
before the measured comparison. The receipts retain these unscored events.

The cache-aware variant successfully changed the configured shared-memory
partition from **100 KiB to 64 KiB**, with dynamic shared memory decreasing
from **32 KiB to 24 KiB**. Total per-block allocation reported by the profiler
is 33,840 versus 25,648 bytes, including static and driver storage. Register
and shared-memory limits both allow two blocks. Achieved occupancy is
49.75% versus 49.56% for its fresh control.

Its L1 sector hit rate rose to 74.3%, and the long-scoreboard ratio fell to
25.70; its paired control is 2.08. The cache change helps the lookup relative
to the first attempt, but it remains much slower than the procedural decoder.
The fresh bracketed operation screen rejects it:

| GPU | Routed rows | Mean of bracketing control medians, ms | Lookup L1 median, ms | Duration multiple |
|---|---:|---:|---:|---:|
| 0 | 16 | 0.9354 | 3.8588 | 4.13× |
| 0 | 32 | 1.0910 | 4.3051 | 3.95× |
| 1 | 16 | 0.9251 | 3.8407 | 4.15× |
| 1 | 32 | 1.0779 | 4.2538 | 3.95× |

It passes the occupancy gate and fails the 3% timing-gain gate by a wide
margin. It does not proceed to a serving benchmark, and no server token rate
is estimated for either lookup. The source/numerical/counter evidence is kept;
production retains the procedural DP4A decoder.


## Custody, checks and production state

The Linux CPU suite passes 37 tests; macOS passes 36 with one platform skip.
The pinned HTTP suite passes 44 tests and the response-lifetime AST suite
passes four. Byte sum additionally passes nine exact JSON-object requests,
mixed-filter isolation, cancellation with a surviving request and slot reuse,
and three high-effort coding pairs. Neither lookup undergoes serving/lifecycle
qualification after its timing rejection. No candidate gets a production
soak or persisted-cache restart qualification because none meets the
performance gate.

Production was restored at **19:11:59 UTC** and independently inspected at
**2026-10-16T19:13:07 UTC**. It is healthy and idle, with no requests, queue or
native leases, sole production-unit GPU ownership, zero crash restarts,
Weightless verified on both cards and Qwen inactive.

The retained production native SHA is
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`;
its profile digest is
`2bfd7ca630f488034cfc1be093317719aeb41e227a8e8766a3e2c856e637c91c`.
The bounded controllers check fresh idle state, sole GPU ownership and Qwen
inactive before maintenance, and restore that profile on exit. A follow-up
window waits for the earlier controller to finish restoration before taking
ownership. Restore-only hooks do nothing until their own window has begun.
An independent read-only inspection checks source/native identity, leases,
queues, GPU owners, service restart count and Weightless on both cards.

[Complete synthetic receipts](../experiments/concurrent_throughput/decode_instructions_2026-10-16.json)
retain all scored answers, raw codebook comparisons, per-agent rates,
samples, cache hits, batching counters, hardware metrics and source/native
hashes. Raw profiler files and operator logs remain in private host receipts.

Reproduction tools are `prepare_decode_instructions.py --mode lookup|vabsdiff`,
`prepare_decode_cache.py`, `build_decode_instructions.py`,
`decode_instruction_oracle.cu`, and the existing GPU/numerical/serving tools in
`experiments/concurrent_throughput/`. Compile the standalone oracle for SM80
with the generated table-header directory on the include path; define
`EXL3_VERIFY_VABSDIFF` for the procedural alternative. GPU tools require an
exclusive pair. Preparation and compilation are never deployment actions.
