# Weightless GLP-44 on two CMP170HX cards

The optional profile was activated on October 14, 2026 (EDT), with API readiness
at 02:20:21UTC on October 15. It applies strength 2.0 at target layers 1–44 on both
TP ranks. Vision, DFlash 2 K7/hybrid 15, the two active slots, Q8 cache 655360,
context 392960, shared prefix/RAM retention and GLM-default/Qwen-explicit routing
are preserved. No clocks, links, drivers, model weights or client settings changed.

## Verification

| Check | Result |
|---|---|
| CPU GLP/container and actual forward-hook cases | 28 passed |
| Startup gate/current-worker receipt cases | 2 passed |
| Native capture CPU scheduling reproduction | 2 passed |
| Real Torch FD-transfer/GIL reproduction | Original wait timed out; released-GIL binding completed; both workers exited cleanly |
| GPU projection vs FP64 CPU reference | 18 cases per card; rows 1,2,7,8,31,64 × strengths 0,1,2; max absolute error 2.3853e-7 |
| Target warmup | 18/18 passed |
| Functional/cold-tail requests | 31/31 passed |
| Five-minute soak | 153/153 passed in 300.315 seconds |
| Total qualified requests | 184, no failures or stalls |

Functional checks covered exact text, arithmetic, Python code, a forced tool call
without executing a tool, a red-image vision probe, two simultaneous arithmetic
requests, repeat requests, and two cold rounds at every tail 1–8 after a 4096-token
prefix. Raw cold-tail calls intentionally had short output limits; these checked
scheduling and successful generation rather than scoring reasoning answers.
The soak included a roughly 30K-token marker retrieval and repeated single/two-slot
requests. Admission counts accounted for all 184 requests, with no competing
request observed in that window. Serving stayed on the same successful API PID.

These are bounded functional/numerical checks. They do not establish improved
intelligence, a refusal score, unchanged broad quality, a throughput gain, or a
measured steering-overhead percentage. Earlier performance/benchmark tables
describe the unsteered profile. Five minutes is not a long-duration field soak.

## Artifact identity

| Artifact | Identity |
|---|---|
| Qualified base engine | `0f685aa9fda3a98a864149b367b7ef570257572e` + four retained P16 patches |
| Additional native patches | Nonblocking graph-capture launch-error check; GIL release around GatedMLP native call |
| Variant engine manifest | 970 files; SHA256 `d55d6097fb3d31fc06cb0817cdef3b587107eb50ae04e82329df8c9be7f480ab` |
| Loaded native extension | SHA256 `c46af7b24b1d814db3ff5995b30450a399263b3a9cf36b5721d2d778eec7eb12` |
| Vector repository | `msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44`, commit `ef85b0169fd25a3c0223eedddee124cf6b1580dc` |
| Vector file | `GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf` |
| Vector SHA256 | `0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c` |
| Vector source checkpoint | `zai-org/GLM-5.3-Flash` at `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |
| Installed target quant | `turboderp/GLM-5.3-Flash-exl3`,3.05bpw at `332ab457b709b7ba30dd9a448be5de03b80a7ac9` |

The vector has 44 F32 directions of width 16384 and explicit GLP v1 project mode.
Layer 1/layer 44 mHC bases/scales match source bytes; their mixer matrices match
the source after ExLlamaV3's BF16→FP16→FP32 conversion. This samples provenance
and coordinate compatibility; it is not complete checkpoint equivalence.

The native build reused 189 verified original P16 objects and rebuilt bindings
and graph objects, with GPUs hidden. The variant source was independently
reproduced and all 970 file hashes matched. Binary rebuilds can differ by paths.
Vector assets are not redistributed; users must obtain authorized HF access.

## Operations and rollback

Four idle samples, five seconds apart, verified no active/pending/queued request,
stable admission count and no Qwen owner before activation. Only the GLM unit's
new `61-weightless.conf` override selects this candidate. The previous engine,
binary, weights and unit configuration are retained. No client was paused.

The first startup passed GPU numerics but failed because a cache copy preserved
read-only directories. Only the private candidate cache permissions were repaired;
the automatic startup retry passed all checks above. That attempt is retained in
the private receipt archive and is not counted as a successful startup.

The GIL fix addresses the captured parent/worker FD-transfer dependency. The
precise native instruction inside the historical MLP call was not observed;
native py-spy requires pausing the process and was not used. Nonblocking parent
and worker stack capture remains active for recurrence diagnosis.

For rollback, first verify no active/queued clients, remove only the Weightless
unit override, reload systemd and restart GLM. The original P16 service includes
the already-merged bounded parser fix. Keep rollback artifacts until field use
supports retiring them. Setup and launcher: [profile](../profiles/weightless/README.md).
