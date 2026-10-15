# Optional Weightless GLP experiment

Status: implemented and CPU-tested, **not GPU-qualified or deployed**. The normal
engine recipe, service, native binary and pin tables are unchanged. This is a
behavioral steering experiment, not a measured speed optimization. Do not use it
to address a stalled server.

The public [GLP specification](https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md)
and [EXL3-quantized vLLM hotfix](https://github.com/msuiche/weightless/blob/main/patches/hotfix-glm53-exl3-steering-projective.py)
describe the GLP operation. Despite its filename, that hotfix targets a vLLM
fork, not the ExLlamaV3 engine used here. Sources inspected at Weightless commit
`39fec2374520d7c759976a027335da21d8a4fcff`. This implementation is original code;
it does not vendor that repository or download its model/vector assets.

## Apply path and cost

GLM's 45 target layers carry FP32 `[batch, sequence, 4, 4096]` mHC streams.
The flattened HC-outer width is 16384. TP exports explicitly replicate the full
stream on each rank, so each rank projects locally without another collective:
`h -= alpha * dot(h, normalized_direction) * normalized_direction`.

The hook follows the stock TransformerBlock's DFlash export and precedes return.
The current layer's exported stream mean is therefore captured before that
layer's intervention. Earlier layers' steering naturally affects later inputs.
The hook flushes a deferred mHC update before projection. The current engine
already disables deferred updates for TP and for exported-state passes.
Prefill's last-KV-block early exit skips the post-layer hook, as it skips the
rest of the block. Draft/vision blocks without mHC are not steered.

A single Triton reduction/update kernel runs per selected layer and rank.
After the initial direction upload/JIT, it creates no full residual projection
temporary, transfers no residual to CPU and adds no host synchronization.
Directions require 64 KiB per layer per rank, at most 2.75 MiB for layers 1–44.
One block per token with 16 warps is a starting configuration, not a measured
optimum. Latency, numerical tolerance, throughput and DFlash acceptance on the
two CMP170HX cards remain unmeasured. No upstream hardware results are claimed
for this setup.

## Prepare a separate tree

From this repository, first reproduce the qualified source and run CPU checks:

```sh
./engine/apply_engine.sh
python3 -m venv .venv-weightless
.venv-weightless/bin/pip install -r experiments/weightless/requirements-test.txt
.venv-weightless/bin/python -m pytest -q experiments/weightless/test_weightless.py
.venv-weightless/bin/python experiments/weightless/prepare_engine.py engine-src /ABS/NEW/experimental-engine
```

The destination must not exist and must be outside the input tree. The script
checks the exact TransformerBlock source hash. Run the standard full engine
verification before preparing the copy. The experimental tree intentionally
fails the normal full engine manifest because it has a changed hook and two new
files. Do not overwrite the qualified engine or change its pins to conceal this.

An eventual isolated candidate uses the experimental tree on PYTHONPATH, with
its existing native extension and Torch/Triton dependencies. It also needs
`gguf==0.17.1` and NumPy. No launcher, unit, switcher or live directory is changed
by this experiment. GPU qualification requires a separate idle window with no
DrugAdopt/Hermes requests and one model owner.

## Explicit startup controls

With no `GLM_WEIGHTLESS_VECTOR`, the adapter is not imported or initialized.
An enabled experimental candidate requires all four environment variables:

```text
GLM_WEIGHTLESS_VECTOR=/ABS/LOCAL/vector.gguf
GLM_WEIGHTLESS_SHA256=<SHA256 of the complete local file>
GLM_WEIGHTLESS_BASE_REVISION=<40-character base checkpoint commit>
GLM_WEIGHTLESS_ALPHA=<explicit value from 0 to 2>
```

Match the revision to the checkpoint behind the installed quantization using
its provenance; a matching vector header alone does not prove weight identity.
There is no calibrated default alpha for this EXL3 quantization. The `[0,2]`
experiment bound is not a quality/safety guarantee. Alpha zero preserves the
original tensor operations and deferred state, though the file is still checked.

Only little-endian GGUF v3, GLP spec v1, explicit `project` mode, rank one and
`residual_stream_post_layer` are supported. Directions must be finite nonzero
1D F32 vectors of width 16384. `direction.N` applies at layer N with no offset;
only layers 1–44 are expressible. Declared layers and tensor names must agree.
Additive/legacy vectors, rank-k, spec v2, alpha scaling, model-revision mismatch
and incompatible shapes fail closed. Conversion/capture workflows are refused.

## Verification boundary

28 CPU tests cover real GGUF round trips, integrity/metadata/shape rejection,
layer mapping, zero-alpha identity and execution of the actual engine forward
method with CPU collaborators to verify capture ordering and prefill early exit.
The separate preparation test proves the input transformer remains unchanged.
The kernel also compiled for explicit SM80 using the host's existing Triton with
`CUDA_VISIBLE_DEVICES` empty, without exposing GPUs or launching any kernel:
16 warps, 64 bytes shared memory, 38081-byte PTX and 22688-byte cubin.
Reproduce on Linux with `python compile_sm80.py /ABS/NEW/output`. Compilation
does not establish numerical correctness, launch viability or runtime latency.

Before enabling a candidate: compare the fused kernel to a Torch FP32 reference
for representative rows; run target/draft/TP checks; measure acceptance and
latency at the same effort and concurrency; then complete functional and soak
checks. This is required work, not evidence already obtained.
