# Current status: rolled back; cache-budget fix staged

The initial candidate passed isolated GPU checks/canaries but its first real
request stalled in disk-cache eviction. Original production is restored and
serving; do not redeploy that candidate as qualified for continued use. The
new disk-budget patch refuses non-progressing/oversized writes before I/O and
exposes effective per-namespace quota. Twelve actual-method CPU checks plus five native policy tests pass, along with
75concurrency checks(74pass/oneexisting skip) and53HTTP checks. Candidate-v5 is
prepared, but GPU shared-root/cache-pressure qualification is pending after an
idle guard declined the18:47attempt without changes.
See the main qualification report for the incident and retained negative proof.

## Earlier headroom/tokenizer qualification and recipe

# Recovery and KDA graph headroom candidate

This candidate is compiled and CPU-tested. The small synthetic KDA GPU oracle
passes on both SM80 cards. Final full-model memory/latency, cancellation, mixed vision/cold-tail, actual
rank-failure and committed-cache recovery checks also pass. The qualified profile was deployed2026-10-19 18:17Z after the active public
request completed. Both router canaries and independent live identity checks
passed. The original profile/cache/drop-in remain available; PR22 stays unmerged.
The standard profile and engine manifest retain their existing native sources.

The service now treats typed PyTorch accelerator/distributed errors and rank
transport/timeouts during TP prefill as shared failures. Its existing B2
terminal latch bypasses per-job reaping, aborts uncertain pending persistence,
wakes all async consumers and returns HTTP 503 from health. The installed
watchdog can then apply its existing restart policy. Ordinary request errors
still use per-job containment. The existing health/readiness/request gates also detect an exited owned child
without waiting on a blocked GPU step. Registered and pending consumers are woken,
and cancellation already waiting for the step lock can observe quarantine.
No new serving supervisor or recovery policy is introduced.

The isolated native patch checks available device memory before instantiating
a **new KDA graph slot**. With `EXL3_KDA_GRAPH_MIN_FREE_MB=1024`, a slot first
encountered below 1 GiB remains on its existing eager launch sequence for the
process lifetime. Existing graphs keep replaying; the first eager call does
not query memory. Default zero retains the original capture policy. This is a
heuristic guard, not reserved memory, a promise that a graph fits above the
floor, or recovery from an already-failed CUDA operation. CUDA documents that
[`cudaMemGetInfo`](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
can surface earlier asynchronous failures and that reported free memory does
not guarantee an allocation will succeed. Those failures remain fatal.

Prepare a fresh candidate after `engine/apply_engine.sh`:

```bash
python experiments/recovery_headroom/prepare.py /absolute/new-candidate --with-tokenizer
```

The optional tokenizer applies the two reviewed patches in
[`../tokenizer`](../tokenizer/README.md). Preparation checks the standard
community profile first, updates only the exact changed-file startup pins,
and records a complete candidate manifest. The build wrapper reuses attested
row32 objects and recompiles only `libtorch/gated_delta_net.cpp`:

```bash
python experiments/recovery_headroom/build.py \
  --base-profile /absolute/attested-row32-profile \
  --candidate /absolute/new-candidate --python /absolute/venv/bin/python
```

GPU work must have exclusive ownership of the two serving cards. This
preparer/build command neither stops production nor starts a candidate.
Qualification must cover eager-versus-graph KDA results, cold mixed-prefill
tails, image/text overlap, four-slot JSON/tool/cancellation behavior, latency
and throughput, cache recovery after restart, and recovery after an injected
rank failure. Use separate candidate disk and compilation caches. Preserve
the production profile, namespace and rollback receipts.

CPU checks:

```bash
python -m unittest discover -s experiments/recovery_headroom -p 'test_*.py' -v
python service/scripts/run_concurrent_cpu_tests.py --engine-root engine-src
```

The C++ policy tests execute the actual patched guard against a modeled CUDA
memory-query boundary. They do not execute GPU kernels. The concurrency suite
executes actual pinned Generator/Job methods with CPU rank/state stand-ins;
the new fault cases also exercise the actual health handler through ASGI.

The separate `gpu_oracle.py` requires exclusive cards and checks the complete
candidate source manifest and built native identity. It runs seven KDA shapes
on each card, including four-way speculative history and the 16-token boundary.
Four repeated invocations per shape exercise first eager use, capture and
replay. A second process sets the floor to 1 TiB to force the new eager policy;
outputs, convolution state and recurrent state must all match bit for bit.
The expected 14 guard-refusal logs are checked. The October 19 receipt is
`gpu_results_2026-10-19.json`. This is a small-layer test, not proof of actual
low-memory behavior or full-model fit, performance or recovery.
