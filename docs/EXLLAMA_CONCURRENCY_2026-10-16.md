# ExLlamaV3 concurrency investigation: October 16, 2026

Status: current-profile baseline measured; isolated sixteen-row candidate built
but not GPU-qualified or deployed. No candidate speed improvement is claimed.
The ordinary engine manifests and launchers are unchanged.

## Current two-card result

These measurements compare the same two independent tasks completed serially
versus together on the current GLM Weightless profile. Each cell is the mean
of two repeats; the second repeat reverses execution order. Time includes the
HTTP client, streaming and bounded answer validation, rather than kernel time.

| Tasks | Effort | Finish both serially | Finish both concurrently | Serial time / concurrent time |
|---|---|---:|---:|---:|
| Two structured arrays | low | 3.42 s | 3.68 s | 0.93x |
| Two structured arrays | high | 4.27 s | 4.42 s | 0.97x |
| Python function + structured array | low | 3.52 s | 3.70 s | 0.95x |
| Python function + structured array | high | 4.06 s | 4.92 s | 0.83x |

A ratio greater than one would mean the group finished sooner concurrently.
None did in this small sample. This does not establish statistical significance
or predict larger contexts, longer jobs, more agents or other prompts.

Every answer passed. The Python function passed six integer-list cases and
input-mutation checks in a restricted, resource-limited Linux child process.
Native counters increased for both batched drafting and batched target
verification in every concurrent repeat. Cache hits were zero.

Structured output work matched exactly: 385 completion tokens at low effort
and 465 at high effort, in both modes and both repeats. Mixed high effort used
366 serial versus 395 concurrent completion tokens, so its wall-time difference
includes changed output work. Mixed low effort used 344 serial and 344–345
concurrent tokens. Counts include reasoning and come from final API usage,
never SSE chunks. This is a bounded load probe, not a general coding score.

[Machine-readable baseline](../experiments/concurrent_throughput/baseline_2026-10-16.json)
includes source/native identity. Raw responses, per-request latency, draft
acceptance and health samples remain in local operator receipts.

## Candidate and evidence

The qualified source has a specialized fused decode path limited to eight
token rows. DFlash K7 verification supplies eight rows per generating request:
one request fits, while two use sixteen and take the larger-row path.
The candidate raises physical capacity to sixteen for dense and routed MLPs.
Python reads capacity from the rebuilt extension, so an old binary cannot
silently supply eight-row storage to the new code.

Only four engine files change: the two Python MLP modules, the shared native
MLP header and the binding exposing capacity. Existing CUDA arithmetic is
reused. All three C++ translation units including the changed header were
rebuilt; 188 attested objects were reused unchanged. This is a dispatch/capacity
experiment, not a newly qualified arithmetic kernel.

The variant compiled and imported with CUDA hidden in 152.35 seconds. Its local
native SHA is `6f29d6fec981e09825fe1be463ca0f2effa4e32df24ce2108a7532e3bfb52649`.
Thirteen new CPU tests passed on Linux. The existing concurrency CPU suite ran
63 cases: 62 passed and one existing skip, without initializing CUDA.
Numerical correctness at new row counts, GPU memory fit, speed and production
readiness remain untested.

The upstream dev snapshot checked during this investigation still defines an
eight-row limit. Sources:
[native capacity](https://github.com/turboderp-org/exllamav3/blob/bd4da327a8/exllamav3/exllamav3_ext/libtorch/mlp.h),
[expert dispatch](https://github.com/turboderp-org/exllamav3/blob/0f685aa9fda3a98a864149b367b7ef570257572e/exllamav3/modules/block_sparse_mlp.py).

## Reproduction tools

Tools are in [experiments/concurrent_throughput](../experiments/concurrent_throughput).
Use the qualified Weightless engine and exact manifest. Preparation refuses
source drift and an existing destination.

```sh
python experiments/concurrent_throughput/prepare_row16.py \
  --engine /ABS/QUALIFIED/engine --manifest profiles/weightless/engine.sha256 \
  --output /ABS/NEW/row16

python experiments/concurrent_throughput/build_row16.py \
  --base-profile /ABS/QUALIFIED_PROFILE --candidate /ABS/NEW/row16 \
  --python /ABS/EXISTING_VENV/bin/python --base-native-sha256 QUALIFIED_NATIVE_SHA256
```

The incremental builder requires this host profile's object-custody receipt,
Ninja command inventory and CUDA 13.3 link layout. It never starts a model or
changes a unit. `prepare_server.py` copies the service, updates the affected
source pin and writes a private launch plan on a different port with separate
disk and compilation caches. It does not execute the plan or change routing.

Run measurements against an idle qualified two-slot service:

```sh
python experiments/concurrent_throughput/measure.py \
  --base-url http://127.0.0.1:8012 --workload structured \
  --effort low high --repeats 2 --output /ABS/NEW/structured-results

# Linux is required for the resource-limited coding oracle.
python experiments/concurrent_throughput/measure.py \
  --base-url http://127.0.0.1:8012 --workload mixed \
  --effort low high --repeats 2 --output /ABS/NEW/mixed-results
```

The client checks stable idle and serving identity, registers its request IDs
from the initial SSE event, and cancels its own streams on other traffic or
identity drift. A bounded registration race is distinguished from a foreign
waiter. Interrupted waves are excluded. The first mixed attempt exposed that
race and was cancelled before output; the corrected client was tested before
collecting the reported mixed results.

## Required GPU window and decision

Both production GPUs are occupied. Coordinate an exclusive client window;
do not launch the candidate alongside the existing GLM owner.

1. Record ownership and stable idle. Preserve the production unit, tree,
   binary and caches for restoration.
2. Release the production owner for the coordinated window. Run
   `gpu_oracles.py --exclusive-window --engine ... --native ... --output ...`.
   It refuses existing GPU owners and reuses the upstream numerical oracle at
   rows 1–16, including GLM 3-bit widths, clipping and expert-range masking.
3. Start the prepared private candidate. Check exact identity, both-device
   Weightless numerics, memory headroom and healthy startup.
4. Repeat matched structured and mixed workloads by effort. Check cancellation,
   survivor completion and freed-slot reuse with existing lifecycle tools.
   Use the installed Nsight Systems for kernel attribution if needed.
5. Stop the candidate and restore the exact production profile regardless of
   outcome. Verify health and sole ownership before ending the window.

Retain the variant only after correctness and lifecycle checks pass and group
completion time improves without unacceptable individual latency. Four-slot
profiles and shorter draft windows remain later experiments; neither is
enabled or promised here.
