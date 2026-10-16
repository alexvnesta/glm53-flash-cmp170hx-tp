# Cache retention and request observability

The serving recipe retains its pinned upstream base and four original P16
patches. Selected upstream changes add FP16 KDA recurrent-state storage,
allocation recovery, an oversized-checkpoint guard and first-token nonfinite
logit detection. This is a selected backport, not an upgrade to all of v1.6.2.
Both native GIL/graph stall fixes remain required by the Weightless variant.

## Request records

Each admitted request emits one `GLM_REQUEST` JSON record on completion,
disconnect, cancellation or error, including cleanup before the response body
starts. Admission rejection is also recorded. Records contain identifiers,
configuration identity, effort, token counts and timings. They retain no prompt,
generated text, image bytes or tool arguments. The journal's normal retention
policy applies; the health endpoint keeps only 32 recent records.

`preparation_ms` covers rendering/tokenization before admission. `http_queue_ms`
is the HTTP admission wait; `queue_ms`, `prompt_ms` and `predicted_ms` are the
engine's own measurements. `engine_ttft_ms` starts at entry to the completion
handler, including preparation and queueing. `visible_ttft_ms` additionally
reflects parser buffering before the first parsed response delta (nonempty for
chat; raw completions may emit an empty delta). Rates on very short
answers are unsuitable as throughput benchmarks. Native timing definitions and
raw counts remain present so downstream reports can choose their denominator.

`/health.request_metrics.active` reports phase, prefill token progress, generated
token progress and age of the latest progress event. These reads use CPU
metadata and do not synchronize a GPU. The existing watchdog policy is unchanged.

## FP16 recurrent state

The standard `run_tp.sh` launcher retains FP32 recurrent storage. The Weightless
launcher selects FP16 unless `RECURRENT_STATE_DTYPE=fp32` is supplied. This changes
state storage and rounding, not weight bitrate, the FP32 mHC residual or the
Weightless projection's precision. Long-context retrieval and speculative rewind
must be qualified for the selected profile; do not assume identical output to
FP32 storage.

For this model, halving the 34 KDA recurrent matrices saves 71,303,168 bytes per
paired checkpoint. The calculated pair size changes from 191,213,568 to
119,910,400 bytes, allowing 44 rather than 28 pairs within 5 GiB. The drafter's
FP16 sliding-window state remains part of every pair.

## Persistent prefix cache

Set an absolute `GLM_PERSISTENT_CACHE_DIR` to enable upstream's `DiskPageCache`.
It uses a B2-owned upstream `CPUPageCache` within `RAM_TIER_GIB` (8 GiB by default).
Unlike the nonpersistent lazy tier, this mode registers its bounded RAM arenas
at startup. The disk budget is 50 GiB by default, shared by fingerprints under
the same root, with a 16 GiB free-space reserve. Low space pauses writes;
upstream rejects unreadable pages and failed available checksums. Its split TP
page payloads currently have no end-to-end CRC, so arbitrary corruption of those
payloads is not guaranteed to be detected. The cache root belongs to the service
account and is private (mode 0700).

The fingerprint includes upstream's model/cache identity and an additional
serving identity: engine and service source digests, native-extension hash,
Torch/Triton versions, kernel flags, recurrent precision and Weightless vector
hash/base revision/alpha/layers/width. Changing any of these creates a separate
namespace. A `serving-profile.json` beside the upstream manifest makes it
inspectable. Source or steering changes deliberately invalidate reuse.

Upstream performs page serialization, available CRC checking, LRU budgeting and paired
target/draft checkpoint I/O. Image-placeholder chains are excluded by upstream's
`persistable` flag. Enqueue-time prefetch is disabled in this adapter: promotion
runs during native allocation on the serialized executor. Idle flushing also
runs there, in bounded steps, keeping the HTTP event loop available.
Generator or B2 custody failures disable further disk writes for that process.

Persistence helps a repeated compatible context after restart or eviction. It
cannot avoid computing a new context's first prefill. Inspect
`/health.persistent_cache` for pages, checkpoints, pending writes, read/write
counts, errors and fingerprint. Confirm both disk page reads and checkpoint
reads, a new process, matching identity and a deep `cached_tokens` value when
qualifying restart reuse.

The Weightless launcher enables persistence at `.cache/prefix` by default. Use
`GLM_PERSISTENT_CACHE=0 RECURRENT_STATE_DTYPE=fp32` for its prior storage policy.
Stop or replace a healthy serving process only after checking active, pending,
queued and router requests. Keep the previous engine, native binary and unit
override as rollback artifacts.

For a systemd deployment, use `KillMode=mixed` so SIGTERM reaches the main
process first while its TP workers can still drain cache I/O. The example unit
allows 60 seconds before killing the entire cgroup. The old default of sending
SIGTERM to all ranks interrupted shutdown and exhausted its 15-second timeout.
This changes shutdown ordering; watchdog failure detection remains unchanged.

## Deliberate conversation checkpoints

`scripts/export_dsh_checkpoint.py --projection SESSION.json --log session.v4.jsonl.zstd
--output checkpoint.json` exports the latest terminal summary, its previous
completed summary and log locators. It refuses an active or changing session
and never rewrites history or sends prompts. Commands and tool-result bodies
are omitted. Summaries are model-reported context, so verify checkout state and
tests before acting on their claims.

For a new explicitly requested task, use that bounded checkpoint with the
existing artifacts rather than automatically importing an entire old transcript.
Keep system/tool definitions stable, append changing task details to user
messages and preserve original history. No automatic compaction or client-wide
provider/settings changes are installed by this recipe.
