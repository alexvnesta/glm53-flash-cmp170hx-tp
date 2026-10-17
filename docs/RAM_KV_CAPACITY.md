# RAM KV caching and four-agent context capacity

The host tier dynamically retains and restores **inactive, reusable KV pages**.
Its pinned RAM arenas have a fixed budget at startup. It does not extend the
context working set of requests that are decoding at the same time.

The local FP16 Weightless profile has these separate budgets:

| Tier | Budget | Purpose |
|---|---:|---|
| GPU target Q8 KV | 655,360 tokens shared across requests | Pages needed by active requests, plus reusable cached pages |
| Pinned system RAM KV | 8 GiB | Reusable hashed pages evicted from GPU storage |
| Paired recurrent state in RAM | 5 GiB | Matching target and draft checkpoints needed for correct resume |
| Persistent disk cache | 50 GiB | Reusable pages and paired checkpoints across process restarts |

These budgets are not additive active-context capacity. Model weights, draft
state, vision, work buffers and recurrent state also occupy GPU memory.

The measured target page geometry is 1,892,352 bytes per 256-token page across
the two TP ranks, or 7,392 bytes/token. The 8 GiB KV arena holds 4,539 complete
pages (8,589,385,728 bytes), roughly 1.16 million token-equivalents of inactive
pages. Page and checkpoint eviction, shared-prefix reuse and simultaneous
sessions determine useful retention. The recurrent-state budget is separate.

## Current GPU pool while four agents decode

| Four independent input contexts | Input total | Within the nominal GPU pool? |
|---|---:|---|
| 4 × 131,072 | 524,288 | Yes, with output and scheduler headroom |
| 389,120 + 3 × 65,536 | 585,728 | Yes, with output and scheduler headroom |
| 4 × 262,144 | 1,048,576 | No |

The per-request cap is 392,960 tokens **including output**. Dividing the shared
pool by four gives 163,840 tokens each before output reservations and other
headroom; that is a planning estimate, not a guaranteed per-agent allocation.
Identical shared prefixes can reduce unique page use. Independent prompts do
not receive that benefit.

The long-context qualification uses 2,048 output tokens per request and checks
rounded page reservations against the shared pool before sending requests.
It tests exact marker retrieval and an independently checked JSON array; it is
separate from the frozen short-task throughput cohort.

## What RAM offloading currently does

`CPUPageCache` stores complete hashed pages when they become unreferenced and
are evicted from the GPU page cache. It restores them during page allocation.
The paired B2 layer retains compatible target/draft recurrent state as well;
KV pages alone cannot establish a correct recurrent continuation. Disk caching
can preserve those records across server restarts.

Active requests retain their required pages in GPU memory. Native concurrency
explicitly refuses active-host migration. Increasing `RAM_TIER_GIB` can retain
more parked histories, but does not make four 256K independent active contexts
fit this GPU pool.

Supporting active context beyond GPU capacity requires a different cache
execution path: page-aware attention transfers or host-aware attention,
bounded scheduling and prefetch, and matching target/draft/recurrent-state
ownership during preemption, cancellation and resume. Its effect on aggregate
and per-agent tok/s must be measured on these PCIe Gen2 cards.

Upstream's [Tier 1 discussion](https://github.com/turboderp-org/exllamav3/issues/269)
confirms idle-session CPU caching. The
[vLLM KV offloading documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md)
also describes completed-block prefix caching and promotion back to GPU. A
reported GPU-plus-CPU cache pool alone does not establish that oversized active
contexts decode concurrently, or establish performance on this two-card TP
configuration.

The later [active RAM KV investigation](ACTIVE_KV_OFFLOAD_2026-10-17.md)
demonstrates four independent 262,144-token inputs in an isolated allocator
variant, using 6.62 GiB of fixed active host backing for Q8 latents/scales and
retaining index planes on GPU. It reports aggregate and actual per-agent tok/s,
cold-prefill time, correctness and the 16–17% short-task throughput cost.
This is an experiment with static backing; the production behavior and budgets
described above remain unchanged.

The [dynamic paging follow-up](DYNAMIC_KV_PAGING_2026-10-17.md) keeps the same
logical pool and four × 256K inputs, with CUDA-managed latent/scales instead
of fixed host backing. Hot pages can migrate to GPU and cold pages can move
to RAM under the driver's policy. Both cold packs passed, with shared decode
at 117.0 / 130.6 tok/s low/high and about 25% improvement over static RAM.
The same short cohort improved 15–16%. This is a driver policy, without an
application LRU or explicit hot-tier size. It remains an isolated experiment;
the production tiers and active capacity above are unchanged.
