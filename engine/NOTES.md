# Engine: ExLlamaV3 plus nine patches

The engine under this setup is [ExLlamaV3](https://github.com/turboderp-org/exllamav3), **the work of turboderp and
its contributors, MIT licensed, Copyright (c) 2025 Turboderp.** We did not write it. We run it at one pinned upstream
commit with the patches in [patches/](patches), which add what this serving setup needs on top: tensor-parallel DFlash2,
a smaller indexer cache for GLM-5.3, safer recurrent-state replay, fair scheduling between prompt ingestion and
generation, and a few native fixes. The patches are distributed under ExLlamaV3's MIT licence; the upstream licence text
is in the checked-out tree (`engine-src/LICENSE`). Several patches port or follow other people's designs and say so in
their comments (vLLM's GLM-5 tail cache, Apache-2.0; Olt1z's tp-mla for the sharded top-k; upstream dev's hybrid
draft mode).

## Use

```bash
engine/apply_engine.sh      # engine-src/ = upstream at the pinned commit + patches, then verification
engine/build_native.sh      # native/exllamav3_ext.cpython-313-x86_64-linux-gnu.so (the interpreter's tag), about 25 min
```

The service runs with `PYTHONPATH=<repo>/native:<repo>/engine-src:<repo>/service/glm`. The service also compiles its
own small kernel (`glm53_k101.so`) on first start, from `service/kernels/k_hcfuse.py` into `service/build/k101`
(needs `ninja` and `nvcc`); that is not part of the engine build.

| file | what it is |
|---|---|
| `UPSTREAM.txt` | upstream URL and the full base commit, `3a8e2fb84cda5a87ec700f1e9935b641416b7e2b` |
| `patches/` | nine patches, applied in file-name order with `git apply` |
| `engine.sha256` | sha256 of all 975 files of the patched tree (relative paths) |
| `apply_engine.sh` | clones, checks out, patches, verifies (`--verify` re-checks an existing tree) |
| `check_engine_pins.py` | checks the engine files the service hash-pins against the tree |
| `build_native.sh` | builds the sm_80 native extension out of tree into `native/` |

All paths default to the repo root and can be overridden with environment variables (see each script's header).
`apply_engine.sh` refuses to touch an existing `engine-src/`.

## Why the patched tree must be byte-exact

The service hash-pins engine files when it starts. `glm_vision.VISION_ENGINE_SOURCE_PINS` (9 files) must match before
the vision tower loads, and `glm_concurrent_ring.ENGINE_SOURCE_PINS` (27 files) before the concurrent ring is installed;
`glm_tp_prefix.ENGINE_SOURCE_PINS` (16 modules) guards the tensor-parallel prefix tier. A tree that differs in one byte
makes the service refuse to start those features. `apply_engine.sh` therefore checks all 975 files against
`engine.sha256`, checks that no file is missing or extra, and runs `check_engine_pins.py` against `service/glm`.
The same check works on its own: `engine/check_engine_pins.py engine-src service/glm`.

`service/glm/glm_dflash_sessions.py` carries a fourth, older pin table (4 modules) for the single-GPU session cache.
Only 1 of its 4 hashes matches this engine. It is not on the tensor-parallel path, and the check reports it as
informational.

## What the patches change

The numbers are the order of application. Each patch starts with a short description of its own.

**0001, native graph capture and cuBLAS workspace** (`exllamav3_ext`). CUDA graph capture is wrapped in a scope object so
that an exception between `capture_begin()` and `capture_end()` ends the half-recorded capture, destroys the capture
stream and graph handles and clears the recorded sites, where before it left a stream stuck in capture mode. Every native
capture site uses it. A new header, `cublas_handle.cuh`, binds ExLlama's persistent per-device cuBLAS workspace to the
calling thread's handle in one place (torch 2.12 to 2.14 get the handle without torch's own workspace setup; other
versions and ROCm use the stock handle); `hgemm` and graph launch use it, `hgemm` also checks that its tensors are on one
device, and the MLA graph path binds the handle before capture because a slot can be warmed on the loader's thread.
These are the same changes as upstream dev commits `4c6a7da` and `e8da92f`, so a base commit past them would not need
this patch. Seven of its ten native files (`graph.cuh`, `hgemm.cu`, `cublas_handle.cuh` and the libtorch sources
`blocksparse_mlp.cpp`, `dsv4_attn.cpp`, `gated_delta_net.cpp`, `mlp.cpp`) are byte-identical to upstream dev `2caf277`.
The other three, `graph.cu`, `libtorch/attention.cpp` and `libtorch/mla_attention.cpp`, differ from dev tip: dev changed
the DSA top-k workspace handling in the last two later, and dev's `graph.cu` has one thread-handle line that this one lacks
(here `mla_attention.cpp` binds the handle instead).

**0002, unused NoPE query storage** (`modules/mla_attn.py`). For an MLA layer without rope, the zero-width `q_pe` slice
still views the whole projected query, so it could not be freed before the sparse kernel ran. The patch drops the unused
`q_nope` view and gives `q_pe` its own empty storage in that case, so the query returns to the allocator before the
attention output is allocated. It is the same change as upstream dev commit `716533a`.

**0003, `k_idx` ring** (`cache/mla.py`, the Triton MLA kernels, `model_init.py`). GLM-5.3's indexer keeps a per-token
`[k || gate]` plane next to the pooled keys, but only reads it to rebuild the pools the current forward pass touches.
With `EXL3_MLA_KIDX_RING=W` (default 0, off) each page keeps only `W` slots of that plane (`W` divides the page size and is
at least 32 for pool size 4), which shrinks that plane from the page size to `W` rows per page. The ring is a normal paged plane; the plane
append and pool update kernels take it as a compile-time constant; eager prefill builds pools from the chunk's own rows
before the ring write reuses the carry slots. Speculative forwards of up to 16 rows are supported and `model_init` checks
that. `EXL3_KIDX_RING_SHADOW=1` keeps full twin planes and byte-compares every forward (debug, small caches). The design
follows vLLM's `Glm5NextTailCache`. No upstream counterpart.

**0004, recurrent replay safety and the extension ABI marker** (`gdn.cu`, `bindings.cpp`, `ext.py`,
`modules/gated_delta_net.py`, `cache/recurrent_util.py`, `generator/generator.py`). A failure while rewinding or replaying a
speculative GDN/Mamba2 pass now quarantines that state: it refuses rewind, stash and free, and the generator re-raises
instead of recycling the slot, because shared target state is no longer trustworthy. A speculative pass must keep at least
one history token, a draft window larger than the recurrent cache's history is refused, and the 4-way value split of the
batched replay scan is skipped for Mamba2. The extension exports `glm_replay_state_abi = 1`, and `ext.py` requires it and
no longer falls back to JIT compilation. That is why `build_native.sh` is a required step, and why an extension built from
unpatched ExLlamaV3 is refused at import. No upstream counterpart.

**0005, tensor-parallel DFlash2 and TP plumbing** (`architecture/dflash2.py`, `modules/arch_specific/dflash2.py`,
`model/model*.py`, `model_init.py`, `util/candidate_topk.py`, `util/tp_sharing.py`, `util/tp_lazy_cache.py`). The DFlash2
drafter can attach to a tensor-parallel target when `experimental_tensor_parallel` is set (`--experimental-dflash2-tp`,
validated before any allocation). Each rank crops its LM-head shard to the real vocabulary, applies the selector's scale and
softcap, takes a local top-k, and the shards are gathered and merged into paired (score, global token id) candidates that
the selector walks; ties break by descending score, then ascending token id, not the dense kernel's lane order.
`EXL3_TP_WORKER_TIMEOUT` (default 20 s) sets how long the parent waits for a rank on cold compile caches.
`EXL3_TP_SHARING_STRATEGY` (`file_system` default, or `file_descriptor` on Linux) picks torch's multiprocessing sharing
strategy for the parent and every rank, and each rank prints a one-line JSON report when it is set. `tp_lazy_cache.py` is
an experimental target-only host page tier for TP (pinned arenas allocated on overflow, batched store and fetch) that the
serving layer uses. Upstream dev added its own DFlash2 tensor-parallel support later (`9cc3bd1`, `e98785b`); these
patches do not include or depend on it.

**0006, fair scheduling** (`generator/generator.py`, `generator/job.py`). New `Generator` arguments `fair_gen_rounds`
(default 1) and `fair_chunk_size` (default off). While some jobs still ingest a prompt and others are generating, the
generator runs several generation steps per prefill chunk and can shrink the chunk, so streaming jobs keep producing
tokens during a long prefill. The defaults reproduce the old behaviour. No upstream counterpart.

**0007, hybrid drafting wrapper** (`generator/hybrid_draft.py`, a new file). A port of upstream dev's hybrid draft mode
(`e72a906d69`, long suffix-match drafts over the regular drafter) packaged as `install_hybrid_drafting(generator, H)`, which
wraps the generator instance instead of editing `generator.py`, `job.py` and `model_init.py`. The module docstring lists
the differences from upstream.

**0008, converter flag.** `conversion/quant_config.py` writes `tie_word_embeddings = True`. The deployed engine predates
upstream's `8962e94` fix, so this reverts it. It affects only the `config.json` that `convert.py` writes, not loading or
serving. It stays in the series so the tree is byte-identical to the one that was measured; drop it (and update
`engine.sha256`) if you convert models with this tree.

**0009, file modes.** Upstream marks `model_init.py` and `architecture/solar_open_moe.py` executable; the deployed tree has
no executable files. Mode change only, no hash changes.

## Build notes

`build_native.sh` runs the engine's own `setup.py build_ext` with `--build-lib native --build-temp native/.build`:
`TORCH_CUDA_ARCH_LIST=8.0`, `MAX_JOBS=4`, CUDA toolkit 13.3, torch 2.14.1+cu130, nvcc flags from
`exllamav3/util/cuda_flags.py` (`-O3 --use_fast_math --compress-mode=size`), host compiler `-Ofast`, GPUs hidden, `nice -n 19`.
It does not load the result. The 349 native source files of the patched tree are the ones the measured extension was
built from.

Compile check of this script on 2026-10-12 (Linux x86-64, Python 3.13.7, torch 2.14.1+cu130, nvcc 13.3.73, `MAX_JOBS=4`,
`nice -n 19`, CPU only, extension not loaded): 191 build steps, 900 seconds, no errors. Result:
`exllamav3_ext.cpython-313-x86_64-linux-gnu.so`, 157,455,312 bytes, sha256
`e30430db247242ec280f77f7ba905a7e2d350fe1101419ba6927b800b101b26d`. `engine-src` was byte-for-byte unchanged afterwards.

That is not the hash of the extension the setup was measured with (`ddb08fec47303681cf96489b072bef0aa2364f4a07385ae653de68df31dc66da`,
157,455,960 bytes), and a rebuild of yours will not match either: the object files embed the source and build paths.
Everything else we compared is the same: 1,707 exported dynamic symbols (identical after demangling), the same `.text`
size (6,100,178 bytes), the same shared-library dependencies, and the `glm_replay_state_abi` marker; the embedded GPU code
differs by 408 bytes, the length of the paths. If you need to compare your build with another one, compare
`nm -D --defined-only <so> | c++filt | sort`, not the file hash.

## Limits

- The tree was reproduced and hashed, and the patches were applied on macOS and Linux. No GPU test was run for this
  repo. The tests the patches add were not run either: `tests/native_lifecycle_cpu`, `native_workspace_cpu`,
  `mla_query_lifetime_cpu`, `tp_dflash_cpu` and `tp_service_cpu` are CPU tests, `tests/modules/mla/test_mla_kpool_ring.py`
  needs a CUDA device.
- `tp_sharing.py` reads two environment variables for its report, `INVOCATION_ID` (set by systemd) and
  `GLM53_TRIAL_RUN_ID` (optional, ours).
- The docstring of `tests/modules/mla/test_mla_kpool_ring.py` gives its commands with an `engine/` prefix
  (`engine/tests/...`), from the layout it was written in; inside `engine-src` drop that prefix.
