# Tensor-parallel cold graph capture candidate

Status: a source-backed deadlock candidate with a CPU scheduling reproduction.
**Not GPU-qualified, built, installed or merged into the production recipe.**

On October 14, 2026 at approximately 21:22 EDT, two active DeepSeek requests
stalled. The watchdog's first-failure nonblocking dump at 21:23:07 captured the
generation thread in `GatedMLP.forward`, `mlp.py:871`, calling
`BC_GatedMLP.run_bszN` during target prefill. It held the GIL. Health did not
answer, GPU utilization became 100%/0% with little power draw, and the existing
watchdog restarted the service at 21:25:17. Startup completed at 21:27:26.
Both DeepSeek turns failed with `backend unavailable or switch failed`.

The bounded parser fix was already active. The new incident is not evidence
against that parser correction; the captured stack is in a different path.
NCCL broken-pipe warnings began after the watchdog initiated shutdown and are
not proof of the original cause. Worker processes observed as zombies during
shutdown are also not evidence that a worker died before the stall.

## Why this synchronization is suspect

`BC_GatedMLP.run_bszN` lazily captures each row-count graph. Its capture-begin
path calls `cudaDeviceSynchronize()` before recording on a separate nonblocking
stream. With TP, another rank may already be queued in a collective waiting for
this rank's next submission. Waiting for every queued operation on the current
device before making that submission introduces a dependency cycle. Shared
experts are distributed asymmetrically in the current TP plan, so graph capture
need not occur on both ranks together.

The standard warmup does not guarantee capture of every supported 1–8-row
GatedMLP slot: its single-row families are 1,2,4,8,16,32,48,64; a first use only
sets `ready_to_record`, and a later call creates the graph. Startup's successful
18/18 warmup therefore does not exclude a cold graph during real requests.

The candidate patch replaces the device-wide synchronization with
`cudaGetLastError()`. It still checks pending launch errors. It does not eagerly
wait for asynchronous faults to complete. Capture remains on the existing
nonblocking stream, and replay remains on the original ordered stream.

This removes the identified dependency, but Python stacks cannot establish
the exact native instruction inside the call. Allocation/autotuning or another
native operation could also block there. No claim is made that the original
incident is conclusively reproduced on CUDA or that this patch alone fixes all
server stalls. Native py-spy traces require suspending the process and cannot
be combined with `--nonblocking`; they were not collected from active consumers.
The installed nonblocking observer now captures each spawned worker separately.

## CPU verification and next qualification

After the normal engine source reproduction:

```sh
python3 experiments/tp-graph-capture/test_capture_schedule.py
```

The test extracts the actual `Graph::capture_begin` method from original and
patched source and compiles both against a bounded CPU model of the CUDA calls.
The original method waits on a queued peer that needs the next host submission;
the candidate returns and allows that submission. A separate error test checks
that a pending launch error prevents capture. This is a scheduling reproduction,
not a numerical or performance benchmark and not hardware qualification.

Before adoption: build a new native extension in a separate candidate tree,
pin its new hash, run asymmetric TP cold-tail shapes (especially rows 3,5,6,7),
check normal target/draft/concurrency behavior and faults, then soak. Preserve
the current production source and binary and obtain an idle window without
DrugAdopt/Hermes requests. The patch is outside `engine/patches`, so the default
qualified build cannot apply it accidentally.
