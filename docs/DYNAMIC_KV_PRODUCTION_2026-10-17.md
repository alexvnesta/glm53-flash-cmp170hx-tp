# Dynamic KV production qualification: not promoted

The managed-KV candidate **failed the real-process restart gate**. The local
backend is restored to the qualified four-slot GPU cache: 655,360 shared tokens,
392,960 per-request cap, profile
`6c5aaeaa0fc5640f3015593a2c1b97570a95b08d97df10ed905923be84459f69`.
Dynamic paging is disabled in production. The temporary managed override was
removed, Qwen is inactive, and all 18 experiment/controller/observer units
checked by this campaign are inactive.

DrugAdopt and Hermes processes were left running. GLM used bounded test and
restart windows. An arriving client interrupted the first private benchmark,
which exposed a router reload gap. That gap was fixed before the router-facing
canary. Test traffic in the canary yields to public client requests while the
same GLM backend remains available to those clients.

## Candidate and acceptance boundary

The allocator, engine, native kernels, service sources and configuration are
the unchanged candidate from [PR17's experiment](DYNAMIC_KV_PAGING_2026-10-17.md).
All 975 engine files and 66 service Python files were verified. The main native
SHA remains `f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`;
the managed allocator SHA remains
`95fa84c848c9660016caeea51e633d26c11669ef4e2f512ba19636e25ba3ec7e`.
The candidate profile is
`6ce2525ba4631923850d429649157cf2f8ff2bd6339907e76e0abe15a33b22e5`.

The new campaign reuses the earlier frozen cold low/high four-by-256K,
numerical and pressure evidence, pinned to its published receipt hash. The
remaining checks were disk recovery, a real process restart, functional/vision
regression and at least 900 seconds of completed four-agent cohort time. A
scored short cohort required CPU PSI below 2% and at least 90% of the prior GPU
control's aggregate rate. Every scored answer still had to pass its original
oracle. Clients were given priority over test traffic.

| Check | Outcome |
|---|---|
| Earlier frozen cold 4 × 256K, low/high | Passed; unchanged PR17 evidence reused |
| Four-way cancellation/survivor/reuse | Passed in the earlier private attempt of this campaign |
| New initial 4 × 256K disk recovery | **4/4 exact answers**, four paired-state reads |
| Real process restart, zero automatic retries | **Failed**: 64 MiB TP registration error, one automatic retry |
| Four-context recovery after that manual restart | Not run after the restart gate failed |
| Vision/tool regression and 15-minute soak | Not run after the restart gate failed |
| Production promotion | **Rejected; GPU profile restored** |

The per-request maximum remains 392,960 including output. The managed pool has
1,187,840 logical slots, covering four 262,144-token inputs, four 32,768-token
output reservations and 8,192 extra slots. Four full 32K outputs were not
generated. CUDA manages migration/eviction of 6.62 GiB of logical target
latent/scales; this is not a fixed residency measurement or an application LRU.

## New four-context recovery observation

The fresh managed process reads a separate copy of the previous synthetic
disk cache. Each request uses the frozen high-effort 262,144-token input and
32K output reservation, temperature zero, and its exact marker/array oracle.
All four restore **258,048 cached tokens each**, then return exact answers.
The disk counters advance from zero to **4,032 page reads**,
**7,629,963,264 bytes read**, and **four paired-state reads**, with zero read
failures or write errors. The persistent fingerprint is
`195c18e1d1098e2a91e303c0fa62989f`.

| Rate window, high effort | Aggregate tok/s | Agent 0 | Agent 1 | Agent 2 | Agent 3 |
|---|---:|---:|---:|---:|---:|
| Common interval with all four decoding | **120.93** | 30.33 | 30.71 | 29.33 | 30.56 |
| Whole pack, including disk recovery/prefill | **25.57** | 6.39 | 6.41 | 6.39 | 6.39 |

All four finish in **68.44 seconds**, after the backend is ready. Output counts
including reasoning are **437 / 439 / 437 / 437**. The shared decode rate uses
**1,579 observed token increments over one 13.0567-second interval**, with 14
samples showing all four matching requests decoding. It is not the sum of four
separately timed request rates. Each whole-pack agent rate uses its actual
completion count divided by the same 68.4373-second wall time.

This is an **as-run recovery/correctness observation**, not a controlled speed
comparison: CPU PSI reaches 7.83%, above the 2% timing guard. Available host RAM
stays above 63.83 GiB. The earlier clean cold benchmark remains the source for
allocator speed comparisons:

| Earlier PR17 cold pack | Aggregate shared decode tok/s | Actual agents 0 / 1 / 2 / 3 | Cold completion |
|---|---:|---|---:|
| Low | 117.02 | 30.43 / 29.19 / 28.94 / 28.45 | 685.75 s |
| High | 130.64 | 32.76 / 33.18 / 31.69 / 33.01 | 713.31 s |

Cold and recovered packs have different cache and I/O conditions. The new row
does not establish a speedup or regression against those cold results.

## Restart rejection and restored service

The manual restart at `2026-10-17T11:44:09Z` encounters:

```text
cudaHostRegister(0x7591e0000000, 67108864) failed: invalid argument
```

This occurs while creating the TP shared transfer arena, before managed KV
allocation. The existing automatic retry loads the candidate successfully,
but `NRestarts=1` fails the campaign's zero-retry restart gate. The gate is
retained. The underlying cause is unproven; the same failure was observed in
earlier GPU-only baseline starts, as recorded in the
[four-slot runtime report](FOUR_SLOT_RUNTIME_2026-10-16.md).

An upstream review checked dev
`448f5ea7fab03b5a52dd1072eae3d38d12a1776f`. Its shared-arena, registration
wrapper and TP-backend files have no change from the pinned source; a new
explicit-copy helper in `cuda_host.cpp` leaves registration unchanged. No
upstream fix or local registration workaround is claimed. CUDA also documents
that memory APIs can report prior asynchronous errors, so the observed call
site alone does not prove the underlying fault source.
[CUDA memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html),
[upstream comparison](https://github.com/turboderp-org/exllamav3/compare/0f685aa9fda3a98a864149b367b7ef570257572e...448f5ea7fab03b5a52dd1072eae3d38d12a1776f).

Rollback removes only the campaign's unchanged `66-managed-kv-qualified.conf`.
The existing `65-four-active-qualified.conf` restores the GPU profile. An
independent inspection verifies all 972 baseline engine files, all 66 service
Python files, native identity, healthy idle state, sole production-unit GPU
ownership, Weightless validation on both cards and `NRestarts=0` at PID 721349.

## Router fix and excluded attempts

The existing local model router is now versioned under
[`service/model_router`](../service/model_router/README.md). Before forwarding
a POST it waits for readiness of the selected backend, holding the lease
through the wait. SSE response headers arrive immediately. Unready health
responses keep their 503 status and expose inflight/waiting counts, allowing
an operator to yield to clients during a reload. A forwarded generation is
never retried. The router change preserves GLM as default and Qwen as explicit.

All **25 mocked router tests pass**. A live synthetic request during the actual
rollback reload receives headers in **1.99 ms**, remains leased while health is
503, and returns the exact answer after **9.59 seconds**, with 11 completion
tokens. The wait is shorter than the 15-second keepalive interval, so no
keepalive comments are observed in this live probe. Existing auth/proxy and
Hermes tunnel services remain active.

The first private attempt sends no model requests: a guard incorrectly treats
`journalctl --grep` exit 1 (no matches) as an error and restores production.
The corrected guard passes four injected cases. The next private attempt
passes lifecycle/warmup and completes 24 low-effort answers (5,552 tokens,
38.88 s, 142.80 tok/s), but a real client interrupts high effort after 15
answers. Its two-effort comparison is incomplete and CPU PSI reaches 2.18%.
It is retained as excluded evidence. The functional adapter was corrected and
syntax/CLI checked before the canary; that later suite was not reached.

[Synthetic receipts and operator sources](../experiments/concurrent_throughput/dynamic_kv_production_2026-10-17.json)
retain the passed disk-recovery answers, common-interval samples, failed gate,
rollback verification, excluded attempts and router probe. Unfiltered request
histories are removed; other clients' prompts and answers are not published.
CPU checks pass 44 HTTP tests, 55 concurrency/preparer tests (one existing
skip), and 25 router tests. Pinned source reproduction/verification and Python
syntax checks also pass. Those checks are separate from GPU qualification.
