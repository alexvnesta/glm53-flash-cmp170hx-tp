# Pinned upstream engine and retained patches

The engine is ExLlamaV3 by turboderp and its contributors, MIT licensed. This
recipe pins upstream dev commit `0f685aa9fda3a98a864149b367b7ef570257572e` and
applies nine patches. It is a reproducible pinned snapshot, not a claim to track
the moving upstream head.

| Patch | Purpose |
|---|---|
| 0001 MLA k_idx ring | Retains only the needed index rows and supports the 655,360-token shared pool on the two 64 GiB cards. |
| 0002 fair scheduling | Bounds long-prefill interference with a decoding peer using 512-token chunks and four generation rounds. |
| 0003 hybrid draft-id guard | Prevents drafting image-placeholder IDs after the image embeddings have been released. |
| 0004 SWA stash fill | Fixes checkpoint stash/restore indexing in the pinned upstream version: use ring fill rather than absolute sequence position. |
| 0005 upstream FP16 recurrent state | Backport of `02446529b9c1bd57032b857b7d7dd78649c03cb3`; adds FP16 GDN/KDA storage and dtype-aware checkpoints. The standard service launcher explicitly retains FP32. |
| 0006 upstream allocation recovery | Backport of `b0c700ceaf1a44c30920c922a94efdd4d43e4f65`; restores popped slots and frees partially created target/draft pairs. |
| 0007 upstream nonfinite logits | Backport of `2540263065d203f53d735e963c52a21471d16d65`; introduces explicit nonfinite-logit failure. |
| 0008 budget and first-token guards | Adapts `f7c893ee86b04b728f09e02e43de202dd559fc56` to this checkpoint cache and follows `c844534a7111ffb86e0f984af3ebd75cf21daea6` with an unconditional first-token check. |
| 0009 cache profile identity | Optional additional identity in upstream's disk fingerprint; separates steering, source/native builds and runtime precision without introducing another cache format. |

Upstream now supplies tensor-parallel DFlash2, hybrid drafting, graph/cuBLAS
changes and recurrent paired states. The earlier custom implementations and
native ABI marker are no longer required. The fourth patch is a correctness
fix, not a card-specific speed claim. Its CPU regression cases distinguish the
broken and corrected ring state.

`apply_engine.sh` fetches the exact commit, applies patches in filename order,
and verifies all 970 files and the exact file set against `engine.sha256`.
The service additionally checks vision and native-concurrency engine-source
pins. `build_native.sh` builds from that verified tree for SM80. A locally
rebuilt binary can have a different hash because build paths are embedded;
`run_tp.sh` records the actual rebuilt extension hash for service custody.

The historical P16 stage-1/stage-2 native sources were identical. The FP16
backport changes native recurrent-state interfaces and requires a new native
build; never pair these Python sources with the old extension. The Weightless
variant also retains both stall-hardening patches. Source reproduction and CPU
tests are distinct from GPU qualification. The historical evidence is summarized in the
README and [qualification record](../docs/QUALIFICATION_2026-10-13.md).

`refresh_manifests.py --engine engine-src --weightless-engine PATH --write` is an
explicit maintainer operation for reviewed source changes. It updates both
manifests and startup pin tables; it is not a replacement for reproducing and
verifying the source through `apply_engine.sh`.

Production's FLA autotune choices are pinned on the measured host to avoid
both TP ranks overwriting the same cache entries. That cache depends on the
host/compiler/software configuration. The recipe can accept an operator's
`TRITON_CACHE_DIR`; it does not distribute the measured host's binary cache or
claim identical floating-point output on every machine.

Credits and upstream license notices remain in `CREDITS.md`, `NOTICE.md` and
the engine license. No converted drafter weights or native binaries are
redistributed here.
