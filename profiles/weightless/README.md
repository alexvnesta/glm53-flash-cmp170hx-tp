# Weightless GLP-44 profile for two CMP170HX cards

This explicit variant keeps the P16 TP/DFlash2/cache/vision settings, adds the
GLP-44 post-layer projector, and uses the two native stall-hardening patches.
The launcher now defaults to FP16 recurrent storage and a bounded persistent
prefix/checkpoint cache. See [cache and metrics](../../docs/CACHE_AND_OBSERVABILITY.md)
for the configuration, identity and memory-reservation contract.
The regular launcher retains FP32 recurrent storage and opt-in persistence.
The [October 15 qualification](../../docs/CACHE_QUALIFICATION_2026-10-15.md)
records the measured checkpoint capacity, cold/restarted context timings and
bounded functional checks. Earlier decode benchmark tables retain their dated profiles.

First grant access to
[the GLP-44 repository](https://huggingface.co/msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44)
on your own Hugging Face account. Download
`GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf` at repository commit
`ef85b0169fd25a3c0223eedddee124cf6b1580dc`. The launcher pins file SHA256
`0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c`.
It downloads no assets and accepts no repository access terms for you.

After the normal environment/model setup, prepare a separate variant:

```sh
./engine/apply_engine.sh
.venv/bin/pip install gguf==0.17.1
.venv/bin/python experiments/weightless/prepare_engine.py engine-src "$PWD/engine-weightless"
git -C engine-weightless apply "$PWD/experiments/tp-graph-capture/0001-nonblocking-graph-capture.patch"
git -C engine-weightless apply "$PWD/experiments/tp-graph-capture/0002-release-gil-during-gated-mlp.patch"
(cd engine-weightless && sha256sum -c ../profiles/weightless/engine.sha256)
ENGINE_SRC="$PWD/engine-weightless" NATIVE_DIR="$PWD/native-weightless" VERIFY=0 ./engine/build_native.sh
GLM_WEIGHTLESS_VECTOR=/ABS/vector.gguf ./scripts/run_weightless.sh
```

`VERIFY=0` skips the baseline engine manifest, which intentionally differs from
this variant. The separate 972-file manifest above verifies the variant instead.
Build on the CPU with GPUs hidden as the native builder does; do not replace an
active server or run two model owners. A rebuilt extension gets its own hash,
which the launcher records and pins. The October 15 cache/FP16 build's hash is
`ee120e00c9c938c7387f51936a7ea5547557f4c9e4a386e37370f5dbb02d4c42`;
rebuild paths/compiler metadata can change binary bytes.

The native changes remove device-wide synchronization before lazy graph capture
and release the GIL during GatedMLP's native call so Torch's FD-transfer server
can run. They preserve tensor arithmetic. These are required for this profile's
captured prefill-stall regression; see the source-backed investigation under
`experiments/tp-graph-capture/`.

Startup performs 18 numerical cases on each resident GPU, during collective-free
warmup, before requests are served. The API refuses readiness unless receipts
from the current process and its worker cover both GPUs. `/health.weightless`
reports the vector hash, source revision, alpha, layer coverage and validated
devices. Checks and vector uploads happen at startup, not per-token CPU I/O.

Strength defaults to the qualified value 2.0. Layer coverage is 1–44 and width
16384. Do not infer that vectors derived for a different model/checkpoint or a
stronger dose are interchangeable. The sampled layer 1/layer 44 mHC parameters
match the canonical source after its dtype conversion; this is not a full
checkpoint-equivalence claim. EXL3 quantization changes the model, and broad
quality/speed improvements have not been established by the bounded checks.

Use the baseline launcher and its original engine/native directories to restore
the unsteered profile. Clear the GLM_WEIGHTLESS_* environment variables. An
operator must first verify no active/queued clients before replacing a healthy
running model. Retain the old directory and unit override for rollback.
