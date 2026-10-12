# Experimental concurrent TP DFlash2 generation

This candidate adds real continuous batching behind
`--experimental-concurrent-generation`. It has CPU contract tests, not a
model/GPU qualification. Default serving keeps one admitted request, the
existing single-sequence ring and the ordinary scheduler. Tuning flags without
an explicit opt-in are rejected before model allocation.

The first guarded profile uses two active sequences, Q8 target cache with
393,216 tokens shared by both jobs, 392,960 maximum context for one request,
fixed DFlash2 K7, BC attention, and two separate 8,192-token draft ring
partitions. The native scheduler can delay a job when its pages cannot fit
alongside active jobs. Two admission slots do not promise that two nearly
384Ki requests run at once. The draft ring is sliding-window storage, not a
reduction in target context or target KV capacity.

## Source and reuse

The engine base is `c8666fbbc58d8731b97b972cedf681d0953beddd`, built on official
ExLlamaV3 dev `6cd89a908d957afba1cd1659fcf7d99a9e56d776`. The service base is
`5d8cf2571439ed54153599c77a148a4c2df39804`.

Official [Generator](https://github.com/turboderp-org/exllamav3/blob/6cd89a908d957afba1cd1659fcf7d99a9e56d776/exllamav3/generator/generator.py)
already compacts active generation jobs into a target batch, carries separate
recurrent state handles, checks each sequence's speculative acceptance, rewinds
its state independently and updates the corresponding draft rows before job
deallocation. Its [DFlash2 implementation](https://github.com/turboderp-org/exllamav3/blob/6cd89a908d957afba1cd1659fcf7d99a9e56d776/exllamav3/architecture/dflash2.py)
also supports proposal batches. This candidate uses those paths.

It backports only the `generator.py` and `job.py` hunks from DailyDisco's
[open upstream PR #448](https://github.com/turboderp-org/exllamav3/pull/448),
head `b0d0d368049191d53adec1007f941668ab2ea68e`. The opt-in profile uses four
generation rounds per mixed iteration and a 512-token prefill chunk while a
job is generating. Without the opt-in, the scheduler defaults remain one round
and its existing prefill chunk. Extra rounds require actual prefill progress
and stop when the mixed workload ends. Smaller mixed chunks can change native
checkpoint boundary coverage; paired prefix retention is excluded here.

The separately reviewed Python-only file-descriptor metadata sharing patch is
included unchanged. `EXL3_TP_SHARING_STRATEGY=file_descriptor` is required for
the proposed Linux trial. Unset sharing preserves the upstream `file_system`
default. The FD change does not alter native forward math or the six native
shared arenas. The trial still requires attested startup, per-invocation
stop/drain and before/after IPC collection. No native C++ or CUDA source changed
from the matching `f0e2cfae624390782b22dd9253b882724487c2211ed8b703895c5a170ed1511c`
build. Native source equality does not qualify new batch kernel shapes.

## Ownership and cancellation

`glm_concurrent_async.py` defers native enqueue until the persistent executor's
step lock is available. Cancellation uses the same lock. A cancelled pending
registration wakes only its consumer. A request rejected during registration
delivers its error to that job without killing unrelated jobs. Native per-job
prefill failures use the existing `reap_failed_job`; shared target/native
failures still fail the generator and are not disguised as isolated errors.

`glm_concurrent_ring.py` leases one physical draft partition when a job becomes
active. Single-job prefill and compact decode batches each map to that job's
partition while retaining absolute RoPE positions. Native target pages can be
shared when hashes match; draft partitions cannot alias. Deallocation or
cancellation releases only that lease. Its next owner writes on the same
persistent worker and fixed per-device CUDA streams after prior queued writes.
Exact source, model/cache, page objects, storage address/schema, worker thread
and stream checks reject drift before mapping.

New jobs run cold using their own `max_cached_pages=0` hint. The shared page
table is not reset. Native recurrent slots are acquired/cleared per job;
returning one slot does not reset the survivor. The existing response-owned
ASGI cleanup retains admission until native cancellation has finished, including
header-before-body failure. Teardown retains shielded executor join before
mapped/native model resources are unloaded.

## Capacity and unsupported combinations

Pass `-ambs 2` before model loading, and keep `--concurrent-max-sequences 2`
consistent. This reserves two native recurrent slots and real batch-2 BC
statics during autosplit. The capacity flag accepts 2 through 8 for experiments;
only the proposed two-sequence profile has a prepared bounded client.

For this converted model's 34 KDA layers, a second slot with seven history
positions adds a raw 1,159,233,536 bytes (about 1.080 GiB), summed across ranks.
The second Q8 draft ring adds 89,128,960 bytes (85 MiB). That is about 1.163 GiB
before additional BC statics, temporary buffers, allocator reserve and loading
margins. These are source-derived sizes, not measured free VRAM or a fit claim.
The rank weight/cache plan must be saved by the guarded model trial.

The first profile rejects previous-request prefix reuse, multi-session paired
retention, generic/target CPU page tiers, active-host migration, MTP and
adaptive draft sizing. It also rejects unsupported native job features such as
requeue, filters, embeddings, banned strings and healing. Removing those guards
would not provide a valid combined implementation.

A later paired-session extension needs per-job restore/capture plans, partition
leases acquired before draft restoration, checkpoint images independent of a
particular physical slot, and per-job hash protection. Cancellation must
invalidate only unpublished state for the cancelled job. Inactive target CPU
spill can copy only evictable unreferenced pages using the existing all-rank
transfer fences. Active-host TP needs a separate all-rank alias/return
coordinator before any checkpoint/page lifecycle operation. These extensions
are not implemented or claimed here.

## Local CPU verification

Use a Python environment with Torch and run from the service tree:

```sh
python scripts/run_concurrent_cpu_tests.py --engine-root /ABS/EXPERIMENTAL_ENGINE
python scripts/run_tp_cpu_tests.py --engine-root /ABS/EXPERIMENTAL_ENGINE
```

Both runners hide CUDA and report CUDA initialization before and after. Tests
execute the pinned native constructor, scheduler, batched verifier, recurrent
slot methods and async wrapper through CPU doubles. They exercise blocked
executor registration, pending cancellation, independent acceptance/rewind,
disjoint/wrapped rings, cancelled-slot reuse, source/owner/stream refusal,
startup defrag ordering and response admission. They do not prove GPU DMA,
real batch kernels, actual model quality or speed.

## Guarded GPU qualification

The operator supplies explicit paths and source pins in their own launch
configuration. It is a proposal; this repository does not start or stop services.
Keep the port private, set `LimitNOFILE=65536`, and run under a per-invocation
stop/IPC guard that records fresh before/after receipts.
A guard requiring exactly one active request or an old API/source hash must be
updated explicitly to this candidate, not bypassed.

Run the saved synthetic fixture with the bounded client:

```sh
python scripts/qualify_concurrent_generation.py \
  --base-url http://127.0.0.1:8013 \
  --fixture tests/fixtures/concurrent_generation.json \
  --output /ABS/FRESH_RECEIPTS --request-timeout 180 --repeats 3
```

The client requires the full shared cache geometry, two admitted sequences,
fixed K7 and no retention tier. It compares serial and concurrent marker
responses, arrives with the saved 16Ki ledger while a stream is producing,
disconnects one unfinished SSE stream, proves the other completes, and admits
a replacement into the freed partition. It rejects literal role/think pieces
and cross-sequence marker leakage. Idle health must show no native jobs or ring
leases, with real batch-2 draft and native verifier rounds, mixed prefill/generation
progress, and slot reuse observed.

Full choices/token counts are compared separately from semantic and lifecycle
checks. Speculative block size, batching and a changed autosplit plan can alter
numerical continuation. A passing bounded oracle is not general losslessness.
Request latency and first-text timing are saved for a matched measurement;
there is no speedup claim before actual results. Startup native probes, allocator
headroom, shutdown worker/IPC absence and ordinary-service rollback remain
required operator checks outside this HTTP client.
