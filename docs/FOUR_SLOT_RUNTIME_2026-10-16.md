# Four-slot long-context and restart qualification, October 16, 2026

The four-active-sequence row32 profile passes long-context correctness,
cancellation/reuse, a sustained four-agent soak and deep recovery after a real
server-process restart. This extends the earlier
[four-slot throughput screen](FOUR_ACTIVE_2026-10-16.md).

These results use the same two CMP170HX cards, Weightless alpha 2, Q8 target
cache, FP16 recurrent state, K7 DFlash2, hybrid-15, four active slots, fair4/512
and a 655,360-token shared GPU pool. Native SHA:
`f0a25e2c707748eed0679d7e58f92907ebd6eea4e3b2b1605037382baeb2d046`.
Profile digest:
`6c5aaeaa0fc5640f3015593a2c1b97570a95b08d97df10ed905923be84459f69`.
All 972 engine files and all 66 service Python files are checked against their
qualified source identities.

## Long-context results

The new capacity probe sends four independent prompts with distinct A/B/C
markers and requires the exact marker values plus a JSON array of integers
0–95. Inputs are calibrated with the serving tokenizer. Each request reserves
2,048 output tokens; temperature is zero and effort is low. Answers are scored
without repair. This is a separate qualification from the frozen short-task
throughput benchmark.

| Pack | Input tokens per agent | Time until all four finish | Exact answers |
|---|---|---:|---:|
| Cold balanced | 131,072 / 131,072 / 131,072 / 131,072 | 333.67 s (5:34) | 4/4 |
| Cold mixed | 389,120 / 65,536 / 65,536 / 65,536 | 390.08 s (6:30) | 4/4 |
| Same mixed pack after process restart | 389,120 / 65,536 / 65,536 / 65,536 | 39.39 s | 4/4 |

The restart row starts after the replacement API becomes ready. Model loading
itself takes 138.25 seconds in this restart and is excluded from the 39.39-second
request-pack timer. Both cold packs and the restore observe four active
sequences and four native ring leases.

| Pack | Aggregate output tok/s including prefill | Server decode tok/s, agents 0 / 1 / 2 / 3 |
|---|---:|---|
| Cold balanced | 3.04 | 23.1 / 27.0 / 26.6 / 26.6 |
| Cold mixed | 2.59 | 97.2 / 21.9 / 21.5 / 21.5 |
| Mixed after restart | 25.64 | 16.8 / 27.3 / 27.5 / 46.4 |

Aggregate delivery divides actual completion tokens by the common pack wall
time, including prefill. Each server decode rate divides that request's
`predicted_n` by `predicted_ms`; the request windows differ and these rates must
not be summed into an aggregate. In the cold mixed pack, the 389K request
decodes after the three shorter requests finish. Its 97.2 tok/s is a serial
tail, not a four-agent speed measurement.

The cold balanced first-token times are 322.6–324.4 seconds. During the cold
mixed pack, the three 64K requests first emit tokens at about 166.3 seconds,
while the 389K request first emits at 387.5 seconds. Other CPU jobs and a
single-core, low-priority, CUDA-hidden compilation run during this correctness
qualification. These are as-run capacity observations, not an uncontended
kernel comparison.

## Recovery and lifecycle evidence

The restarted process starts with zero admitted requests and the same source,
native, profile and persistent-cache fingerprint. It restores
**388,864 / 61,440 / 61,440 / 61,440 cached input tokens**, then returns all four
exact answers. Disk metrics record **2,239 page reads**, **4,236,976,128 bytes
read**, **four paired-state reads**, zero page-read failures and zero write
errors. Page writes drain before the seed process stops. This verifies paired
target/draft continuation rather than relying on KV-hit counts alone.

The four-way lifecycle probe passes 20 mixed/coding requests, two surviving
streams and two replacement requests, with two intentional cancellations.
All leases, request records and pending tickets retire. The legacy functional,
tool, vision, retention and cold-tail suite passes all 31 checks.

The same frozen 24-task throughput fixtures then run for **337.23 seconds** at
four agents, alternating low and high effort: **192/192 requests correct**.
No request bodies, token budgets or correctness oracles are changed. This is a
stability soak; its rates are retained separately from the controlled A/B.
The complete qualification contains 259 successful synthetic requests plus
the two intended cancellations. It does not establish broad task quality or
real-agent production reliability.

Two restores of the prior two-slot production profile and the first profiling
attempt encounter a startup failure: `cudaHostRegister(..., 67108864)` returns
`invalid argument` while registering a TP shared-memory arena. The production
unit's existing automatic retry recovers each time; the failed diagnostic
receives no requests and is stopped. Both four-slot runtime-qualification
launches complete without an automatic restart. Separate diagnostics pass
192 host-registration/copy checks and 12 repetitions of the API's TP startup
path that stop before loading weights. They do not reproduce the error; no
engine fix is claimed. The failures and successful production restorations
remain in the operator receipts.

## RAM capacity

The fixed 8 GiB pinned RAM KV tier dynamically retains reusable pages. A
separate 5 GiB paired-state tier and 50 GiB disk tier support resume. They do
not increase the active GPU working set: four independent 128K contexts fit,
while four 256K contexts exceed the shared pool. See
[RAM KV capacity](RAM_KV_CAPACITY.md) for page geometry, output reservations and
the distinction between inactive caching and active attention offloading.

[Synthetic qualification receipts](../experiments/concurrent_throughput/four_slot_runtime_2026-10-16.json)
retain request outputs, prompt hashes, observed token counts, individual
timings, cache/recurrent-state evidence, lifecycle results and every scored
soak request. The probe's prefill sampling field spelling was corrected after
the run; request construction and correctness checks are unchanged. The
artifact records the exact probe hash used for this run.

## Accepted production deployment

At `2026-10-16T23:50:24Z`, the local GLM backend is accepted with **four active
sequences**, retaining the row32 native above. The separate
[64-row verification experiment](VERIFICATION_ROWS_2026-10-16.md) misses its
throughput selection gate and is not deployed. The general repository launcher
continues to default to two sequences.

The router-facing four-way lifecycle probe passes, followed by **48/48 frozen
canary tasks**. A separate 5K-input cache seed drains its disk writes, the
production service restarts, and the resumed request restores **4,864 cached
tokens** with **19 disk page reads** and **one paired-state read**. Read and
write errors stay zero. The process changes from PID `3991722` to `3996563`;
both launches record `NRestarts=0`.

An independent inspection verifies all 972 engine files, all 66 service Python
files, the native hash and four-slot health, empty requests/queue/leases, sole
production-unit GPU ownership, Weightless validation on both cards, inactive
Qwen and inactive experiment/controller/watcher units. The original two-slot
service override remains available for rollback. Production receipts and
inspection are included in the
[verification comparison artifact](../experiments/concurrent_throughput/verification_rows_2026-10-16.json).
