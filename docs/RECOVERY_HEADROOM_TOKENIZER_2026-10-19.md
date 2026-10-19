# Recovery, headroom and tokenizer qualification, October 19

**Status: CPU qualification passed; GPU qualification and deployment pending.**
Production has retained its existing process, four-slot GPU-resident profile,
native library and cache configuration throughout this work. No maintenance
window, production restart or model generation benchmark has been run.

## Observed failure and change

On October 17 at 20:09:52 UTC, a 96,526-token request with one image failed
with CUDA graph-instantiation out of memory during TP prefill. Three following
requests each waited approximately 300 seconds for a worker. NCCL reported a
600-second all-reduce timeout; the service restarted at 20:28:29 UTC. This
sequence is observed in the logs. It does not prove the initial allocation's
underlying cause or establish that every vision request risks the same fault.

The prefill path could contain that accelerator error as one request's
failure. It now sends typed accelerator/distributed/transport failures to the
existing shared terminal latch before native per-job cleanup can perform
further rank I/O. The real async failure sink wakes all consumers and the
actual health handler returns 503. The existing watchdog remains responsible
for restart decisions. Ordinary request errors retain per-job handling.

An isolated KDA native patch makes new graph captures optional below a chosen
memory floor, retaining the existing eager kernels for those slots. The
candidate floor is 1 GiB. It does not reserve that memory or guarantee freedom
from OOM, and may reduce speed for newly encountered shapes. The baseline
engine, native profile and default graph policy are preserved.

## CPU/tokenizer results

The initial, unchanged upstream tokenizer backport failed two exact-ID cases
because its one-time probe changed special-token mode. A local restoration
fix is included. The corrected tokenizer passes all **54** cases, including
actual multimodal alias replacement with small CPU tensors. No CUDA context
was initialized.

The following timings use the model's own chat template with a synthetic
code/tool prompt on the serving host, four CPU cores, four Rayon threads,
low process priority, and 12 alternating-order repeats per arm. Each timed
call checks the complete token tensor against the pinned tokenizer. Times
include tokenization and integer tensor construction; rendering is separate.

| Actual tokens | Pinned median | Corrected backport median | Time saved |
|---:|---:|---:|---:|
| 8,356 | 14.61 ms | 14.18 ms | 0.44 ms |
| 32,932 | 79.24 ms | 16.49 ms | 62.75 ms |
| 98,468 | 220.38 ms | 50.17 ms | 170.21 ms |
| 202,182 | 416.43 ms | 94.64 ms | 321.79 ms |
| 288,179 | 652.55 ms | 141.46 ms | 511.09 ms |

These are synthetic CPU preparation measurements, not end-to-end request or
decode gains. The 8K input is below the parallel path's character threshold.
The request-log median prompt was approximately 202K tokens, making the
measured large-input path relevant without establishing an exact production
latency saving.

| Check | Result |
|---|---|
| Actual Generator/Job concurrency, recovery and ASGI health contracts | 72 run, 71 passed, 1 existing optional native-SAM skip |
| HTTP contracts | 52 passed |
| Actual response-handler AST / real Starlette cleanup | 4 passed |
| Model-router contracts | 25 passed |
| Existing concurrency-experiment CPU contracts | 55 passed |
| Actual KDA guard compiled against CPU CUDA boundary | 5 passed |
| Corrected tokenizer exact-ID qualification | 54 passed |
| Native compilation | One KDA C++ object rebuilt; 190 attested objects reused; import passed with CUDA hidden |
| GPU numerical/memory/performance/soak/recovery | Not run |
| Production deployment | Not performed |

The exact pinned source and community profile reproduce before candidate
patching. Candidate manifests and startup source pins are checked separately;
the changed native extension must never be paired with unmodified sources.
New source/native identity creates a separate persistent-cache fingerprint.

## Remaining gate

Both CMP170HX cards are occupied by production. Full-model GPU qualification
requires exclusive access and cannot run alongside the loaded production
model. A router-held idle maintenance window delays new generations, so it
requires an availability decision before proceeding. No GPU test or deploy
is inferred from CPU success. Preserve the original production profile and
automatic restoration path during every authorized window.

Candidate preparation and policy limits:
[`experiments/recovery_headroom`](../experiments/recovery_headroom/README.md).
Tokenizer upstream attribution, regression and benchmark command:
[`experiments/tokenizer`](../experiments/tokenizer/README.md).
