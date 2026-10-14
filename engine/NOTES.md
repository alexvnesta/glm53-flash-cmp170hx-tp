# Pinned upstream engine and retained patches

The engine is ExLlamaV3 by turboderp and its contributors, MIT licensed. This
recipe pins upstream dev commit `0f685aa9fda3a98a864149b367b7ef570257572e` and
applies four patches. It is a reproducible pinned snapshot, not a claim to track
the moving upstream head.

| Patch | Purpose |
|---|---|
| 0001 MLA k_idx ring | Retains only the needed index rows and supports the 655,360-token shared pool on the two 64 GiB cards. |
| 0002 fair scheduling | Bounds long-prefill interference with a decoding peer using 512-token chunks and four generation rounds. |
| 0003 hybrid draft-id guard | Prevents drafting image-placeholder IDs after the image embeddings have been released. |
| 0004 SWA stash fill | Fixes checkpoint stash/restore indexing in the pinned upstream version: use ring fill rather than absolute sequence position. |

Upstream now supplies tensor-parallel DFlash2, hybrid drafting, graph/cuBLAS
changes and recurrent paired states. The earlier custom implementations and
native ABI marker are no longer required. The fourth patch is a correctness
fix, not a card-specific speed claim. Its CPU regression cases distinguish the
broken and corrected ring state.

`apply_engine.sh` fetches the exact commit, applies patches in filename order,
and verifies all 968 files and the exact file set against `engine.sha256`.
The service additionally checks vision and native-concurrency engine-source
pins. `build_native.sh` builds from that verified tree for SM80. A locally
rebuilt binary can have a different hash because build paths are embedded;
`run_tp.sh` records the actual rebuilt extension hash for service custody.

The native source is unchanged between the stage-1 and stage-2 engine trees.
The deployed setup reused the stage-1 extension; source-only changes do not
mean a new GPU binary. Source reproduction and CPU tests are distinct from
GPU qualification. The final GPU evidence and bounded measurements are summarized in the
README and [qualification record](../docs/QUALIFICATION_2026-10-13.md).

Production's FLA autotune choices are pinned on the measured host to avoid
both TP ranks overwriting the same cache entries. That cache depends on the
host/compiler/software configuration. The recipe can accept an operator's
`TRITON_CACHE_DIR`; it does not distribute the measured host's binary cache or
claim identical floating-point output on every machine.

Credits and upstream license notices remain in `CREDITS.md`, `NOTICE.md` and
the engine license. No converted drafter weights or native binaries are
redistributed here.
