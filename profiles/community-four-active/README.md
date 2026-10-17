# Four active sequences on two CMP 170HX cards

This explicit source profile packages the GPU-resident row32 configuration
measured in the community benchmark. It retains TP2, Q8 target cache 655,360,
392,960 total tokens per request, DFlash2 K7, hybrid-15, FP16 recurrent state,
Weightless GLP-44 alpha2, native broadcast/gather and NCCL reductions. Active
system-RAM KV paging is disabled. The shared cache does not provide four
simultaneous 392,960-token requests.

The general launcher continues to default to two active sequences. This
profile adds the measured engine and service overlays, verifies all 972 engine
files and the startup pin tables, and explicitly selects four slots. The
native sources are the existing production row32 sources; this package adds
no kernel optimization. Rebuilt library hashes vary with source/build paths.

After cloning the repository at the recipe's pinned commit:

```bash
./scripts/setup_venv.sh
.venv/bin/pip install gguf==0.17.1
./engine/apply_engine.sh
./profiles/community-four-active/build.sh "$PWD/community-profile"
GLM53_ENGINE_SRC="$PWD/community-profile/engine" GLM53_NATIVE_DIR="$PWD/community-profile/native" \
  ./scripts/get_models.sh
```

Supply your authorized GLP-44 L1–44 alpha2 GGUF vector as described in
[the Weightless profile](../weightless/README.md), from the gated HF repository
`msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44` at revision
`ef85b0169fd25a3c0223eedddee124cf6b1580dc`.
The wrapper verifies SHA-256
`0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c` and runs the
existing GPU numerical checks. The vector and converted drafter are not
redistributed in this repository. Attribution and upstream licenses remain in
[CREDITS.md](../../CREDITS.md) and [NOTICE.md](../../NOTICE.md).

```bash
export GLM53_COMMUNITY_PROFILE="$PWD/community-profile"
export GLM_WEIGHTLESS_VECTOR=/absolute/path/to/GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf
# Optional production-style inactive RAM/disk cache; use your own absolute path.
export GLM_PERSISTENT_CACHE_DIR="$PWD/session-cache"
export GLM_PERSISTENT_CACHE_GIB=50
export GLM_PERSISTENT_CACHE_MIN_FREE_GIB=16
./profiles/community-four-active/run.sh
```

Use model IDs `GLM-5.3-Flash-low`, `GLM-5.3-Flash-high` or
`GLM-5.3-Flash-max` in requests. They select the same loaded model at the
stated effort. An explicit `reasoning_effort` request field overrides the
alias. The default model ID remains `GLM-5.3-Flash` at high effort.

The benchmark uses the upstream tool's original synthetic prompts and exact
API completion token counts, including reasoning. It measures inference
throughput; it does not establish coding accuracy, intelligence or steering
quality. Per-agent contributions come from the same upstream measurement
window and are reconciled against its aggregate token count.
