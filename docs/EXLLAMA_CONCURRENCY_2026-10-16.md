> Follow-up: [JSON mode and final local qualification](JSON_ROW16_QUALIFICATION_2026-10-16.md)
> deployed the sixteen-row variant after additional checks. The failures and
> restored-production status below describe the original investigation window.

# ExLlamaV3 concurrency investigation: October 16, 2026

Status: an isolated sixteen-row candidate demonstrated a bounded throughput
gain, but remains experimental. High-effort strict JSON responses failed their
format contract. The original production profile was restored and verified;
ordinary engine manifests, launchers and routing are unchanged.

## Measured gain and decision

Both profiles ran GLM Weightless alpha 2 on two CMP170HX cards, with two active
slots, the same weights, K7 drafting, hybrid suffix drafting and cache budgets.
The primary comparison below uses direct backend connections for both profiles.
Each cell averages two repeats, with serial/concurrent execution order reversed
in the second repeat. Both tasks must pass before a timing comparison is valid.

| Structured tasks | Existing eight-row path, concurrent | Candidate sixteen-row path, concurrent | Pair completion time reduction |
|---|---:|---:|---:|
| Low effort | 3.28 s | 2.93 s | 10.8% |
| High effort | 3.98 s | 3.50 s | 11.9% |

On the candidate, serial execution took 3.35 s at low effort and 4.17 s at high
effort. Concurrent execution therefore increased completed-work rate by **14.3%
and 19.1%**, respectively. These percentages are throughput increases; the
corresponding time reductions versus serial execution are 12.5% and 16.0%.
Completion-token work matched exactly: 385 low-effort and 465 high-effort tokens
per task pair. Native counters confirmed batched draft and target verification,
and cached-token counts were zero.

The low-effort mixed coding/array pair also completed two valid comparisons:
3.49 s serial versus 3.14 s concurrent, an 11.2% completed-work rate increase.
Serial output used 344 tokens in both repeats; concurrent output used 344 and
345. The later high-effort failure leaves that overall mixed report incomplete;
these are its completed low-effort comparisons only.

**High-effort mixed coding is not a successful timing result.** The initial
parallel failure lost its nested exception and answer. After improving logging,
three separate unchanged diagnostic repeats each returned the code JSON inside
Markdown fences, violating the raw JSON contract. All three extracted functions
passed the unchanged six-case and nonmutation oracle afterward. That extraction
is posthoc diagnosis; all three official format checks remain failed. The
existing production backend passed all three corresponding control attempts.
This is a prompt-specific format-compliance difference, not evidence of a
general coding-intelligence loss or a demonstrated arithmetic kernel defect.

The candidate passed eighteen GPU numerical cases, 31 functional/cold-tail
requests, disconnect cleanup, survivor completion and freed-slot reuse. Both
cards verified Weightless numerics; no native leases remained after lifecycle
checks. A new long soak and realistic agent-trace qualification were not run.
The format failures and limited workload coverage block production promotion.

[Full machine-readable trial results](../experiments/concurrent_throughput/row16_2026-10-16.json)
contain per-repeat times, token counts, latency samples, identities, numerical
cases, failed answers and posthoc diagnoses. Raw operator receipts are retained
locally and identified by SHA-256 in that artifact.

## Production controls and variability

Fresh controls after restoring the original native binary gave the following
results. Direct and routed controls used the same restored production PID;
each cell averages two repeats.

| Connection | Effort | Finish both serially | Finish both concurrently | Serial / concurrent time |
|---|---|---:|---:|---:|
| Direct backend | low | 3.39 s | 3.28 s | 1.03x |
| Direct backend | high | 4.14 s | 3.98 s | 1.04x |
| Router | low | 3.34 s | 3.30 s | 1.01x |
| Router | high | 4.17 s | 4.05 s | 1.03x |

These controls are faster than the initial production snapshot below. The
measurements do not establish the cause of that variation. In particular, do
not attribute the entire initial-to-candidate difference to the kernel or to
routing. The direct-backend comparison above removes the transport mismatch,
but two repeats still do not establish statistical significance or predict
longer contexts, larger jobs, more agents or other prompts. No clocks or power
limits were changed during this trial.

## Initial production snapshot

These earlier measurements compare the same two independent tasks serially
versus together through the production router. Each cell is the mean
of two repeats; the second repeat reverses execution order. Time includes the
HTTP client, streaming and bounded answer validation, rather than kernel time.

| Tasks | Effort | Finish both serially | Finish both concurrently | Serial time / concurrent time |
|---|---|---:|---:|---:|
| Two structured arrays | low | 3.42 s | 3.68 s | 0.93x |
| Two structured arrays | high | 4.27 s | 4.42 s | 0.97x |
| Python function + structured array | low | 3.52 s | 3.70 s | 0.95x |
| Python function + structured array | high | 4.06 s | 4.92 s | 0.83x |

A ratio greater than one would mean the group finished sooner concurrently.
None did in this initial sample; the fresh controls above showed small gains.

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
includes source/native identity and the candidate's pre-trial build snapshot.
Its candidate status describes that earlier phase, not the final trial.
Raw responses, per-request latency, draft
acceptance and health samples remain in local operator receipts.

## Candidate and evidence

The qualified source has a specialized fused decode path limited to eight
token rows. Ordinary DFlash K7 verification supplies eight rows per request:
one request fits, while two use sixteen and take the larger-row path.
The candidate raises physical capacity to sixteen for dense and routed MLPs.
Python reads capacity from the rebuilt extension, so an old binary cannot
silently supply eight-row storage to the new code.
Suffix-match hybrid rounds can verify wider windows and remain outside this
sixteen-row hypothesis.

Only four engine files change: the two Python MLP modules, the shared native
MLP header and the binding exposing capacity. Existing CUDA arithmetic is
reused. All three C++ translation units including the changed header were
rebuilt; 188 attested objects were reused unchanged. This is a dispatch/capacity
experiment, not a newly qualified arithmetic kernel.

The variant compiled and imported with CUDA hidden in 152.35 seconds. Its local
native SHA is `6f29d6fec981e09825fe1be463ca0f2effa4e32df24ce2108a7532e3bfb52649`.
Nineteen new CPU tests passed on Linux. The existing concurrency CPU suite ran
63 cases: 62 passed and one existing skip, without initializing CUDA.
The upstream MoE reconstruction/dense oracle passed eighteen numerical cases
across both cards: rows 1, 2, 4, 7, 8, 9 and 16 at small dimensions, plus rows
9 and 16 at GLM's 4096/2048 widths with expert-range masking. The original
3e-3 relative-to-maximum tolerance is unchanged. The adapter expands only the
oracle's output/counter capacity and extracts its helper without installing
pytest into the serving runtime. Model-level functionality, speed and lifecycle
qualification are separate checks, reported above.

Two setup attempts failed before model readiness: an oracle-adapter anchor
assumption and omitted launcher configuration. Both were corrected and covered
by CPU regression tests; failed attempts supplied no performance data. Later,
the high-effort mixed failure exposed lost nested exception details. Failed
answers and validation errors now survive parallel exception groups, including
oracle timeouts. The original failed wave remains unclassified in the record.

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
It preserves the reviewed engine and collective configuration and refuses
unknown model configuration variables. This guards against an unmatched
comparison or a startup failure from dropping flags such as the MLA index ring.

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

## Reproducing the GPU window

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

Promotion still requires broader format/tool-calling checks, a longer shared
buffer and cancellation soak, and realistic agent traces. Preserve the recorded
prompts and validators when comparing profiles. Four-slot profiles and shorter
draft windows remain later experiments; neither is enabled or promised here.
