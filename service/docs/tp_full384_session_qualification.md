# Full 384Ki TP completed-session qualification

Status: bounded full 384Ki completed-session lifecycle qualified on 2026-10-09. The complete copied-receipt audit passed 1195 checks, and 16 additional saturation/fresh-restoration checks passed. The runtime source and native binary are unchanged. This proof applies to the saved profile and requests below.

The tested profile uses 393216 target-cache tokens, context 392960, Q8 target/draft caches, the 8192-token draft ring and fixed K7. It enables TP and DFlash2 explicitly, with BC1 and both GEMV flags set to `0`. Optional completed-session retention uses a 2 GiB target-only inactive tier, 1 GiB paired-state budget, 4 GiB native recurrent budget and eight checkpoints. Requests remain serialized. Weights and active attention remain on GPU. This feature does not enable active-latent TP migration, simultaneous active generation or a larger configured context.

The original current-dev prefix-mode campaign passed its native/model/ring/cancellation/queue lifecycle, summarized in [tp_session_cache.md](tp_session_cache.md). Its medians were 123.229 count/44.115 LRU tok/s against fresh LS 96.223/34.231. Whole choices and TP/BC/GEMV profiles differed, so those observations do not compare equivalent generated work or isolate a kernel speedup. General speculative continuation equivalence remains unqualified.

## Historical failures remain visible

The 65,536-token TP profile failed a 66-token marker request with 79 completion tokens and literal assistant/think control text. Fresh no-retention and retention controls produced the same complete failure before pressure and after pressure. This rules out retained state or CPU-page restore as a necessary cause; it does not identify the cause. That geometry remains semantically unsupported. The full 393216 profile has a different actual TP allocation plan and rank-cache geometry and is qualified separately.

The earlier full 384Ki campaign passed its six pressure replies, four preprobes and seven known semantic oracles, but its distributed marker request was only 1,476 tokens despite declaring 110,000 to 140,000. The geometry gate correctly failed. Throughput and queue stages were unreached, so that campaign did not pass overall. Owned shutdown/restoration still completed. The malformed historical fixture is retained as [tp_distributed_markers_v1_invalid_geometry.json](../tests/fixtures/tp_distributed_markers_v1_invalid_geometry.json).

## Corrected fixture and complete audit

The separately versioned [tp_distributed_markers_v2.json](../tests/fixtures/tp_distributed_markers_v2.json) retains the genuine measured 133K synthetic ledger. The exact template/tokenizer renders 133299 tokens, independently confirmed by the endpoint owner. Its four unique marker values are at token positions 52, 44,440, 88,845 and 133,257. The other six probe requests, runtime source/configuration, saved 220K pressure requests and complete downstream validation sequence are unchanged.

A fresh ordinary LS control passed this exact corrected request with the four ordered values, 57 completion tokens and stop. This is a bounded known-answer observation, not general long-context task quality. The TP replay passed all six pressure/resume/cancellation-recovery replies, four short preprobes, seven postprobes including two genuine 133K inputs, measured throughput, queue/disconnect behavior, exact owned IPC cleanup and actual original generation/idle restoration. All eight post-command phases returned 0 under one fresh guarded invocation.

Public artifacts include source/tests/synthetic fixtures and sanitized scope. No weights, native binaries, copied engine, raw machine receipts, private paths or service controllers are redistributed. The runtime source and MIT notice remain unchanged.

## Retention, capacity and known-answer result

Cold A and B each rendered 219992 prompt tokens. Resume A, resume B and post-cancel recovery B reused 219904 tokens and matched their own cold complete choices and semantic token counts. All six pressure replies also matched both saved LS pressure arms and the earlier full 384Ki TP pressure replies. Speculative/cache usage metadata was compared separately and was not universally identical. The unchanged saved pressure fixture SHA256 is `2ef2a93f520505c73a37ee13ce108689ee4390b8a04d47625e950da35dd80cbf`.

The first A had zero target host slabs. Pressure from B caused 183 target-page D2H pushes. The pressure phase ended with 383 H2D restores and no paired checkpoint pruning. Later probes/throughput/queue work saturated 680 target slots at 2144665600 registered mapping bytes, within the 2 GiB requested target budget. Rank slot sizes were 1,720,320 and 1,433,600 bytes, summing 3,153,920 per logical page. Final counters recorded 1542 pushes, 383 restores and 862 finite-capacity evictions, with zero pin failures or skipped spills. One completed checkpoint was pruned during the later work. Finite retention may therefore fall back cold; it does not retain every session indefinitely.

The byte ceiling covers exact requested registered rank slabs. Final token snapshots used 1,392,640 additional bytes; paired snapshots used 861919232 bytes within 1 GiB, and native recurrent cache used 746455040 bytes within 4 GiB. Paired native accounting overlaps the native cache and must not be added twice. Python/driver/other process RAM remains separate. No process-RAM cap, reserved-GPU-memory or external available-VRAM claim is made.

All eleven pre/post known-answer observations passed their semantic, transport and geometry gates. The independent ledger and distributed-marker requests measured 133153 and 133,299 tokens. The corrected TP distributed-marker full choices and semantic token counts matched the ordinary LS control; entire speculative usage differed. These are known synthetic retrieval/logic answers, not general long-context quality or blanket LS/TP losslessness.

Cold A/B recorded native `prompt_ms` of 166422/165851; resume A/B and recovered B recorded 276/264/254 ms. These intervals exclude separate queue/restore timing and are not whole HTTP or first-token latency. They describe one saved lifecycle rather than a repeated latency benchmark.

## Throughput and lifecycle

| Saved 1024-token request | Session-profile median tok/s | Earlier modern-prefix median tok/s |
|---|---:|---:|
| Counting |123.316531|123.229443|
| Concurrent LRU reasoning |44.084995|44.115378|

One warmup pair preceded three alternating measured rounds. All six measured replies had cached_tokens=0 and 1024 output tokens; complete choices and entire usage matched both the earlier modern-prefix and legacy TP results. The observed differences were +0.071% and -0.069%, so no meaningful throughput improvement from retention is claimed. Different TP/LS whole choices still prevent an equivalent-work LS throughput claim. The cap does not prove a completed 1000-integer answer or ten complete LRU sections.

An unfinished generated-text pressure stream was disconnected and B recovered exactly. Pressure cancellation can occur during reasoning or visible text. The separate queue canary closed an unfinished visible-content holder, removed a disconnected waiter, produced three exact queued replies and returned idle. Startup inventory recorded six native arenas plus one 68-byte Torch object before the first request. Final same-invocation capture found those seven objects, and main-only graceful shutdown drained all owned processes in 4.280 seconds with all seven exact names absent. Fresh independent receipts then matched the restored original stock-native/source/argv/environment and exclusive resource ownership, plus actual `4`/`stop` and idle generation.

Runtime engine head remains `c8666fbbc58d8731b97b972cedf681d0953beddd` on official dev `6cd89a908d957afba1cd1659fcf7d99a9e56d776`; service runtime matches `39ae7078860c1f4a18ec9b02c3319096d08b9cd1`. Native SHA256 remains `f0e2cfae624390782b22dd9253b882724487c2211ed8b703895c5a170ed1511c`. New docs/fixture commits do not change these runtime bytes.

This result does not qualify the failed 64Ki layout, general speculative continuation equivalence, TP active-latent migration, simultaneous active requests, larger configured context or an enabled boot/restart transition. Those remain separate gates.
