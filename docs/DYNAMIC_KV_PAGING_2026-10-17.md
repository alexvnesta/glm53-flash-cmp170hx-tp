# Dynamic active KV paging on two CMP170HX cards

The candidate passed the frozen short cohort and both four × 256K capacity packs.
The current production four-slot GPU profile is unchanged.
The later [production qualification](DYNAMIC_KV_PRODUCTION_2026-10-17.md)
passes four-context disk recovery, then rejects promotion at the restart gate.

## Allocation and policy

The private candidate retains the static RAM experiment's 1,187,840-token Q8
target pool, four native active sequence slots and 392,960 per-request cap.
This reserves four independent 262,144-token inputs plus 32,768 output tokens
per request and 8,192 extra slots. Output reservation is distinct from actually
generating four complete 32K answers.

Only target latent payloads and group scales use `cudaMallocManaged`, through a
scoped PyTorch `CUDAPluggableAllocator` / `MemPool`. The model weights, sparse
index planes, recurrent state, draft state and workspaces keep their normal
allocators. The pool stays alive with its cache tensors. The loader's CPU
metadata/export allocation still uses CPU tensors, and runtime checks verify
that real worker payload pointers have `cudaMemoryTypeManaged`.

CUDA manages page faults, migrations and pressure eviction between GPU and
system RAM. This is a dynamic driver policy, **not an application LRU or an
explicit per-layer GPU hot-cache budget**. CPU first touch initializes the
logical pool without migrating every unused row to GPU during allocation.
No preferred GPU is set, preserving the driver's default thrash mitigation.
Q8 representation, sparse selection, attention math and page-transfer paths
remain unchanged. The existing inactive RAM/disk prefix tiers remain separate.

The eleven layer payloads total 7,108,034,560 bytes (6.62 GiB). That is
logical managed storage, not a measurement of fixed RAM or GPU residency.
All 133 TP placement rows match the static RAM prototype. Relative to the
smaller GPU-only reference, 43 rows differ, as documented in that experiment.

Both CMP cards report managed-memory and concurrent-managed-access support.
[CUDA documents demand migration and oversubscription](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
[PyTorch documents scoped pools](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
and cautions that UVM pressure can incur both eviction and page-in transfers.
The model benchmark is necessary to establish whether this policy helps here.

## Frozen short cohort

Four submitting/active slots, 24 tasks per effort, separate low/high runs,
temperature zero, 768 output tokens. Prompts, oracles, effort levels,
speculation and native kernels are unchanged from the static RAM experiment.
The fixture SHA256 is
`38b5c9340800821e1ba5cf10f997c3831f5cc2858f23c5008b14ad916fb05ee7`.

| Backing | Effort | Aggregate tok/s | Actual agent 1 / 2 / 3 / 4 tok/s | Wall time | Completion tokens |
|---|---|---:|---|---:|---:|
| Static RAM | Low | 128.37 | 30.20 / 30.89 / 27.72 / 39.56 | 43.25 s | 5,552 |
| Demand paging | Low | 148.77 | 34.99 / 35.80 / 32.13 / 45.85 | 37.32 s | 5,552 |
| Static RAM | High | 120.73 | 33.32 / 20.66 / 28.29 / 38.47 | 48.89 s | 5,903 |
| Demand paging | High | 138.95 | 38.35 / 23.77 / 32.55 / 44.28 | 42.48 s | 5,903 |

All 48 scored candidate tasks passed. Aggregate throughput improved 15.9% low
and 15.1% high. Completion token totals match their static RAM counterparts,
so this comparison is not an artifact of shorter answers. The smaller GPU
reference measured 154.98 / 143.12 tok/s; demand paging was 4.0% / 2.9% lower
in this run. This is one scored cohort per configuration/effort, without
statistical confidence intervals.

Aggregate and actual per-agent delivery count completion tokens, including
reasoning, over the same full cohort wall time. Dynamic backfill distributes
different work to each agent. Individual rates are not aggregate divided by
four, and an agent index does not denote a fixed identical task assignment
across configurations. SSE chunks are never used as token counts.

## Four independent 256K contexts

Both packs passed all four exact answers with zero cached input tokens.
All four full contexts were observed decoding simultaneously at each effort.

| Backing | Effort | Aggregate decode tok/s | Actual agent 1 / 2 / 3 / 4 decode tok/s | Common decode interval | Cold completion |
|---|---|---:|---|---:|---:|
| Static RAM | Low | 93.53 | 24.32 / 23.33 / 23.13 / 22.74 | 10.07 s | 732.10 s |
| Demand paging | Low | 117.02 | 30.43 / 29.19 / 28.94 / 28.45 | 8.05 s | 685.75 s |
| Static RAM | High | 104.76 | 26.08 / 26.64 / 25.52 / 26.52 | 16.10 s | 738.23 s |
| Demand paging | High | 130.64 | 32.76 / 33.18 / 31.69 / 33.01 | 12.09 s | 713.31 s |

Decode improved **25.1% low / 24.7% high**. These rates use one common
interval in which all four requests were decoding, excluding prefill and the
completion tail. Low has 9 samples covering 942 observed generated tokens over
8.05 seconds; high has 13 samples covering 1,579 tokens over 12.09 seconds.
These are short observed decode intervals, not a generation soak.

Both configurations produced 1,016 completion tokens low and 1,750 high,
including reasoning. Cold completion improved from 12m12s to **11m26s** low
and from 12m18s to **11m53s** high: 6.3% / 3.4% shorter. Including the entire
cold-prefill wall time, the candidate delivered **1.482 / 2.453 completion
tok/s aggregate**. Corresponding actual per-agent delivery was
0.370 / 0.373 / 0.370 / 0.367 tok/s low and
0.613 / 0.615 / 0.613 / 0.613 tok/s high.

First-token times were 673.89–676.64 seconds low and 696.43–699.62 high.
Each pack processed 1,048,576 input tokens. The low-to-high order also includes
normal cache retention/eviction work in the later pack. The higher high-effort
decode rate reflects a different token mix with additional reasoning; it is
not a causal claim that increasing effort makes generation faster.

The synthetic marker/array oracle tests capacity and custody, not general
256K coding or reasoning quality.

## Qualification and timing control

Sixteen direct cases across both cards passed: CPU metadata/export, identical
Q8 append, sparse attention at 1/8/16/512 rows with shuffled page tables,
explicit cold migration followed by demand reads, and page round trips.
Nsight recorded 17.826 MB host-to-device and 8.913 MB device-to-host migration,
132 CPU page faults and 69 GPU page faults in this diagnostic. Its kernel
timings are not model tokens/sec.

The first full-model lifecycle pass completed 24 requests and two intended
cancellations. DrugAdopt's independent CPU tests then created contention, so
the warmup was interrupted before any scored cohort and production was
restored. Those timings are excluded; the other jobs were left running.
The quiet retry passed lifecycle/cancellation again. It requires 20 consecutive
one-second samples below 2% CPU PSI avg10 before maintenance and each cell,
and rejects a scored cell if any timing sample reaches 2%.

All 96 candidate API completions and two intended cancellations passed in
the clean window. All 81 short-cell CPU samples were 0%; the long cell peaked
at 0.18%, below the 2% threshold.

An isolated pressure diagnostic then reserved ordinary GPU memory leaving
768 MiB of headroom per card and revisited 1.5 GiB of dirty managed data.
Full-data min/max checks passed in reverse and forward order on both cards,
without explicit prefetch or eviction calls. Nsight recorded **10.87 GB to GPU
and 9.69 GB back to RAM**, 9,228 CPU page faults and 37,039 GPU page faults
across the diagnostic. This establishes migration under deliberate pressure;
its timings are not model throughput measurements.

The first diagnostic attempt exhausted its own workspace: converting a
384 MiB chunk to int64 for a checksum needed a 768 MiB temporary while only
682 MiB was free. The revised diagnostic uses full-data min/max checks with
scalar workspace. That failed diagnostic and its trace are retained; no model
benchmark or served allocator was changed or rerun to repair it.

The original four-slot GPU production profile was restored healthy and idle,
with its source/native identities verified, zero restarts, Qwen inactive and
sole ownership of both GPUs. All eleven experiment/helper units are inactive.
The desktop's local routed health endpoint also matched that restored profile.
No production promotion, soak, restart/cache recovery or vision qualification
is claimed for this allocator.

The preparer only builds an isolated candidate and manifests. It never starts
or stops hosting. All GPU checks require a coordinated exclusive window with
identity/ownership guards, foreign-traffic cancellation and restoration.

## Reproduction and source identity

The pure preparer is
[`prepare_dynamic_kv.py`](../experiments/concurrent_throughput/prepare_dynamic_kv.py).
It requires the reviewed four-slot TP Q8 profile, CUDA 13.3 headers/runtime and
both cards' concurrent-managed-access support. It builds a separate small
allocator library and keeps the qualified ExLlama native extension unchanged.
`managed_kv_oracle.py` checks numerical/copy behavior;
`managed_pressure_oracle.py` requires otherwise idle GPUs and is separate from
`sustained.py` and the frozen long input packs.

The tested base repository was `02da8a62fc31fd9590c4cd913ea45bc57d95a242`.
Experimental profile digest:
`6ce2525ba4631923850d429649157cf2f8ff2bd6339907e76e0abe15a33b22e5`.
The unchanged ExLlama native SHA256 is
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`;
the separate allocator library SHA256 is
`95fa84c848c9660016caeea51e633d26c11669ef4e2f512ba19636e25ba3ec7e`.
A matching main native hash alone does not distinguish this candidate from
the GPU or static RAM variants. Cache source, allocator library, profile and
logical pool must also match.

The [complete synthetic evidence](../experiments/concurrent_throughput/dynamic_kv_2026-10-17.json)
includes requests, common-interval samples, per-agent arithmetic, source pins,
pressure data, prior controls and private receipt hashes. A
[plain-text community summary](DISCORD_DYNAMIC_KV_RESULTS_2026-10-17.txt)
contains both tables. Local validation passed 44 HTTP tests and 55 concurrency /
preparer tests, with one existing skip, plus pinned engine reproduction and
verification. CI also parses all concurrency experiment Python sources.
