# FP16 state, persistent cache and request progress: October 15, 2026

The Weightless profile passed qualification on two 64 GiB CMP170HX cards.
The final implementation adds profile-aware disk reuse, request timing/progress,
selected upstream recovery/guard fixes and FP16 recurrent storage. It retains
both native GIL/graph stall fixes, the same weights/vector at strength 2.0,
NCCL TP2, DFlash2 K7/hybrid15, two slots, Q8 pool 655,360, context 392,960 and
GPU0 vision. Model routing and client settings were preserved.

## Measured context reuse

Each request used low effort, temperature 0, three known markers placed near
the beginning/middle/end, and a 34-token answer. Input lengths came from the
serving tokenizer. Every answer returned all three values in order.

| Input and cache state | Time to finish | Engine prefill | Engine generation | Cached tokens |
|---|---:|---:|---:|---:|
| 32,768 cold | 21.468s | 20.713s | 0.627s | 0 |
| 98,304 cold | 60.783s | 59.810s | 0.650s | 0 |
| 270,336 cold | 170.418s | 169.068s | 0.608s | 0 |
| 270,336 after restart | 11.714s | 2.953s | 0.583s | 266,240 |

Time to finish is caller wall time and includes rendering, queueing, I/O and
response handling; engine timings use their own narrower boundaries. The
restart reused a matching namespace in a new API process, with
1,040 disk page reads and
1 paired-state read observed by the final snapshot.
This is a repeated-context benefit. It does not measure faster decoding, a new
prompt's first prefill, general task quality, or a GLM-versus-Qwen speedup.
The table uses the final source identity; an earlier intermediate run's 8.707s
restore remains in private receipts and is not substituted for this result.

## Memory and cache budgets

Each paired checkpoint shrank from 191,213,568 to 119,910,400 bytes, a 37.29%
reduction. The final seed actually retained **44 pairs** using 5,276,057,600
bytes in the same 5 GiB budget; the preceding FP32 profile retained 28.
This measures capacity, not a multi-session latency or quality improvement.
The B2-owned upstream RAM tier reserves 8,589,385,728 bytes at startup, within
its 8 GiB ceiling. Disk is bounded to 50 GiB across profile namespaces, with a
16 GiB free-space reserve. There were no disk write/read errors in this window.

## Validation

| Check | Result |
|---|---|
| HTTP/async lifecycle, metrics privacy and failure write-abort | 36 passed |
| Actual response-handler AST with real Starlette/admission/cleanup | 4 passed; normal semantic payloads unchanged |
| Concurrent CPU/custody suite | 63 run; 62 passed and 1 existing skip |
| Real upstream disk restart, corruption, geometry and allocation cases on CPU | 6 passed |
| Upstream allocation rollback cases | 52 passed |
| First-token nonfinite logit cases | 6 passed |
| Weightless format/forward-hook CPU cases | 28 passed |
| Weightless startup gate / graph scheduling CPU cases | 2 / 2 passed |
| Fresh baseline / Weightless source reproduction | 970 / 972 exact file hashes |
| GPU projection numerics / target warmup | 18 cases per card / 18 of 18 |
| Functional/cold-tail requests on final source | 31 passed |
| Long-context known-answer requests | 3 cold plus 1 after restart passed |
| Streaming disconnect | Retired exactly once; following request passed |
| Five-minute soak | 151 requests passed in 300.978s |

Functional cases covered arithmetic, Python code, a forced tool call without
executing it, vision, two simultaneous requests and prefill-boundary tails.
Short raw-tail probes check scheduling/generation rather than reasoning quality.
The soak included roughly 30K marker retrieval and repeated one/two-slot calls.
The final process stayed at PID 1032778 from restore through soak, with
zero unexpected restarts. Admission counts account for every qualification
request; no competing request was observed. No client was paused; controlled transitions used verified idle boundaries.
No broad model benchmark or upstream GPU kernel matrix was run.

## Artifact identity and operations

- Base: `0f685aa9fda3a98a864149b367b7ef570257572e` plus nine retained/selected patches;
  see [upstream provenance](../engine/NOTES.md). This is not a full v1.6.2 upgrade.
- Weightless variant manifest SHA256: `b4a1ab7258a731aed815f0130e9476d62e52f4f1f700737bf3e97720733e9bd0`.
- Native SHA256: `ee120e00c9c938c7387f51936a7ea5547557f4c9e4a386e37370f5dbb02d4c42`.
- Serving profile digest: `6a647e6e312ec005625ea35d4549bf1a88ad06b2b3a3771715a35be80a750471`.
- Disk fingerprint: `9b4ddde41f526c3ade3453aad49f39e1`.
- Build reused 188 verified native objects and rebuilt bindings, GDN CUDA and
  GDN libtorch objects with GPUs hidden. Both original native stall fixes remain.

Structured request records and health progress retain no prompt/output text or
tool arguments. A bounded read-only DeepSeek checkpoint exporter preserves the
original conversation and refuses an active/changing session. No automatic
compaction or client-wide settings change is installed.

An initial planned restart inherited a control-group-wide SIGTERM policy and
hit its 15s forced-kill timeout. `KillMode=mixed` now signals the main process
first, leaving TP workers available for cleanup, with a 60s final bound. The
next controlled shutdown completed cleanly in about four seconds. Watchdog
failure-detection thresholds are unchanged. The obsolete, inactive download
unit was disabled without stopping a process. Its definition/rollback receipt
and previous model profiles are preserved.

The host uses only the new `62-cache-observability.conf` override for this
profile. For full-build rollback, verify stable idle admissions and no active,
pending, queued or router request; stop with the current unit so cleanup can
drain, remove only that override, reload systemd and start the preserved
October 14 Weightless profile. Keep its engine/native binary together. A direct
launcher can select the prior storage policy with
`GLM_PERSISTENT_CACHE=0 RECURRENT_STATE_DTYPE=fp32`; that does not restore the
older native build. GLM remains default and Qwen explicit.

Upstream rejects unreadable pages and available checksum failures, but split TP
page payloads currently lack end-to-end CRC. The CPU corruption test covers its
checksummed non-TP fixture; it does not prove arbitrary TP payload corruption
will be detected. Five minutes and three synthetic retrieval contexts do not
establish long-term reliability or broad FP16 quality equivalence.

Configuration and timing definitions: [cache and observability](CACHE_AND_OBSERVABILITY.md).
