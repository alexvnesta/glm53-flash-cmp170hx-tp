# Experimental TP DFlash2 and completed-session overflow

Use explicit `-tp --experimental-dflash2-tp`. The typed engine configuration
remains default off. `--dflash-prefix-cache` retains the previous request;
`--dflash-session-cache` enables bounded completed-session retention. Choose one
of those two options. Only one request runs at a time.

The native engine stashes target recurrent checkpoints in every TP rank. The
parent holds the native integer handle and an epoch cookie, paired with the
exact token prefix and raw packed draft-ring window. It does not clone the
worker checkpoint buffers. Recreated native hashes cannot reuse old draft
snapshots. Budget replacement dispatches deletion of the exact handle to every
rank after draining pending replies. Aliased native keys keep their shared
handle until the last key is removed. An unrelated miss preserves completed
sessions. Cancellation/errors discard unpublished work. Native state or target
page eviction can cause a cold retry.

`--target-cpu-cache-gib N` requires session retention. The target-only tier is
attached after a fresh native generator is validated and defragmentation is
disabled in an owned installation transaction. Generic `--cpu_cache_size` is
refused for TP DFlash2: it would also attach the small modulo draft ring using
target page identities. No draft pages enter this CPU tier.

No host mapping is allocated at installation. Native eviction of a complete,
unreferenced GPU page prepares a slot on every rank. Each rank uses
`PinnedArena(required=True)`, an anonymous mapping registered with CUDA and
rounded to `mmap.PAGESIZE`. This avoids Torch's pinned-allocator power-of-two
slab charges. Registration failure closes partly prepared peer slabs, disables
further growth, and lets missing prefixes run cold. Existing stored pages
remain usable. Budget counts the sum of rank slab sizes, including replicated
planes on every rank. Health reports rank sizes, logical slots and committed
mapping bytes. Compare allocation counters at an idle boundary. Token copies,
Python metadata, driver bookkeeping, draft snapshots and native recurrent
HostPool RAM are separate costs; this is not a process-RAM cap. Paired native
checkpoint accounting overlaps native cache accounting and is not additive.

Every slot byte range belongs to a fixed rank/device/current CUDA stream. A
previous H2D read completes before its next D2H overwrite on that same stream;
different devices operate on disjoint mappings. Transfers refuse tensor
identity/pointer/layout or thread/stream drift. The parent pins the generator,
cache-owner serial, page-table geometry and physical page objects. A duplicate
host image is recycled only after exact token, parent, native page ownership,
complete referenced GPU residency and non-revertible state checks. Sole
protected host images remain intact; an incoming spill may be skipped when all
host entries are protected. Teardown synchronizes recorded rank streams before
unregistering mappings.

Shutdown shields the whole responsive async close until its native step and
executor settle. It releases session state, then separately attempts ack drain
and model unload. A failed ack still permits native worker destruction, and
cleanup never hides an original lifespan exception. Use a stop guard
(ExecStop/ExecStopPost with KillMode=mixed), a bounded stop time and before/
after IPC capture. Do not signal TP children concurrently with the parent.

Base is official dev `6cd89a908d957afba1cd1659fcf7d99a9e56d776`. It includes
published default-off TP, typed loader plumbing, reviewed graph/BLAS patches,
and sparse NoPE query-lifetime cleanup. The new slot transport is Python-only.
Use an extension built from the matching C++ tree, rather than historical
stock1.5.4 binaries. The source manifest identifies exact inputs, and this TP
helper gates nine relevant installed engine files before attaching.

The existing LS experimental tier keeps its separate stock1.5.4 source fences.
They have not been broadened to current dev. LS tests use the original external
fixture; TP tests use this new engine tree. Ordinary default-off LS serving is
unchanged.

CPU contracts use real native page/checkpoint/queue bodies and two rank doubles
with CPU tensors. They establish ownership, hook order and raw-copy contracts,
not asynchronous DMA, graphs, full-model parity or performance. The TP HTTP
pilot saves exact requests and raw responses, then performs cold A/B, resume A,
unrelated cold C, resume B, real SSE cancellation D and recovery B. It requires
actual host spill/restoration and matching full choices/semantic token counts.
The earlier 65,536-token target-cache pilot is semantically unsupported after fresh controls reproduced its marker failure before pressure.
For 393,216/220,000 use a 2 GiB target budget initially and confirm that reported
slot capacity exceeds the required overflow pages. The current-dev engine/API has bounded prefix-mode qualification below and a separate [full 384Ki completed-session result](tp_full384_session_qualification.md). The 64Ki failure remains documented and is not qualified by either full-profile result.

Active sparse-latent host migration remains unsupported in TP. The LS adapter
binds actual layer tensors/native attention objects in one process; TP places
those objects in ranks and exposes only exports in the parent. It needs a
per-rank adapter/coordinator, rank-local registered latent arenas and pointer
leases, return/rebind before checkpoint/page lifecycle operations,
collective-safe failures and separate live qualification. This inactive tier
preserves completed sessions while target working tensors remain on GPU. It
does not increase simultaneous active requests or the configured context limit.

## Reproducible portable tests and qualification limits

```sh
python scripts/run_tp_cpu_tests.py --engine-root /PATH/C8666FB_ENGINE
python scripts/run_cpu_tests.py --engine-root /PATH/STOCK_16A_ENGINE
python scripts/run_http_cpu_tests.py
```

The TP suite passes 61 contracts on the exact current engine source. The separate
LS regression passes 112 contracts on stock16a; those original source fences
have not been loosened to accept dev. HTTP-only checks use a separate pinned
Starlette/AnyIO environment, documented in `http_lifecycle.md`. All Torch CPU
runs confirm CUDA remains uninitialized.

Engine base is official dev `6cd89a908d957afba1cd1659fcf7d99a9e56d776`;
experimental engine head is `c8666fbbc58d8731b97b972cedf681d0953beddd`.
Service plumbing head before admission/response cleanup is
`7b526ec`, based on published service `f18c0a7`. New helper source pins are
recorded in `docs/publication_source_scope.json`. Use an extension compiled from
the exact engine C++ tree. A successful native build or CPU suite alone is not
a full-service ABI, CUDA worker, IPC cleanup or inference qualification.

TP DFlash2 stays explicit and default-off. Historical request-matched profile
comparisons used TP+BC1+joint GEMV settings versus LS+BC0+ordinary GEMV settings,
with different whole choices and reasoning/output allocation. Those are profile
throughput observations, not isolated-kernel speedups or equivalent work.
Target-only versus DFlash continuation has stable full-output mismatches on
bounded fixtures. Actual-layer diagnostics localized some Q1/Q8 arithmetic
differences but did not establish a universal parity fix. The 64Ki profile still needs diagnosis. The later full 393216-token profile has separate bounded host-transfer/lifecycle qualification; older legacy TP receipts alone do not qualify it. No lossless
upstream TP PR is proposed by this branch.

## Current-dev prefix-mode live result, 2026-10-09

The exact current-dev engine and matching native extension loaded the full
393216-token model profile. Eight real native GPU checks, six alternating
1024-output throughput requests (three per prompt, all with cached_tokens=0), five short/16K
semantic checks, a 15-step prefix/ring-wrap/cancel-recovery sequence, and the
queue/orphan-response canary passed. Native extension SHA256 is
`f0e2cfae624390782b22dd9253b882724487c2211ed8b703895c5a170ed1511c`.
This is current-dev bounded model/native evidence, not only the historical
stock1.5.4 result or CPU build evidence.

Median decode rates were 123.229 tok/s for counting and 44.115 tok/s for the
concurrency LRU prompt, against fresh LS profile medians 96.223 and 34.231 tok/s.
The saved requests/settings matched, but whole choices differed and the profiles
also differed in TP, BC and GEMV settings. The roughly 28–29% observed profile
gain is descriptive; it does not establish isolated-kernel speedup, equal work,
complete count/LRU answers, general quality or speculative losslessness.

This campaign used previous-request prefix mode. The combined TP multi-session target CPU-pressure tier was not qualified by this prefix-mode pass alone. A subsequent full 384Ki lifecycle passed separately; its 64Ki profile still failed as recorded below. The operator supplied the live result. Sanitized geometry
and source/build pins are in [qualification_summary.json](qualification_summary.json);
raw machine logs, IPC paths, binaries and process receipts are excluded.

## 64Ki multi-session pilot failed a following semantic check

A separate target-only RAM pressure pilot used 65536 cache tokens. Cold A/B each
rendered 40018 prompt tokens; retained A/B and post-cancel B recovery reported
39936 cached tokens and passed their exact saved answer checks. That retention
result is narrower than the overall service behavior.

The next five short/16K probe requests completed HTTP successfully, but the cold
`short_markers` request failed semantic validation. It had 66 prompt tokens,
79 completion tokens, cached_tokens=0 and finish reason `stop`. Its visible
answer repeated the four markers and included literal assistant/think control
pieces. This is an observed malformed answer, not a transport failure or a
successful semantic check. The full 384Ki pressure stage was skipped, and the
original service was restored at 2026-10-09T17:16:46Z.

The cause is unresolved. Fresh matched64Ki controls with retention disabled and enabled produced the same complete malformed marker answer before pressure and afterward. Retained state or CPU-page restore is therefore not a necessary cause. This does not isolate a numerical/layout cause.
Do not promote multi-session TP as generally qualified, lossless or a universal
performance upgrade from the prefix-mode and retention passes. Keep this
failure visible during diagnosis and require matched fresh controls before
attributing it to a particular layer of the implementation.

## Full 384Ki completed-session follow-up

The separate [full 384Ki follow-up](tp_full384_session_qualification.md) preserves the fresh64Ki cold-control failure and earlier malformed distributed fixture. Its calibrated replacement and complete pressure/semantic/throughput/IPC/restoration replay passed the final audit. No runtime source change or general losslessness claim is introduced.
