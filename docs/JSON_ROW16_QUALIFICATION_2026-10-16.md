# JSON-object mode and sixteen-row decode qualification

> Follow-up: [sustained traffic profiling and the 32-row variant](SUSTAINED_CONCURRENCY_2026-10-16.md).

Status: **qualified and deployed on the local two-card host**, October 16,
2026, at 12:45 UTC. The sixteen-row binary and opt-in JSON-object capability
are active. Original unit files and the previous profile remain available for
rollback. The general repository launcher still uses its standard engine.

## What changes

The October 16 candidate sometimes wrapped a requested code object in Markdown.
The original strict-format failures remain recorded in
[the concurrency investigation](EXLLAMA_CONCURRENCY_2026-10-16.md).

The API now offers explicit `response_format: {"type":"json_object"}` on
`/v1/chat/completions`, enabled at startup by `--experimental-json-mode`.
It uses the pinned engine's upstream `LLGuidanceFilter`, with an independent
parser per job and an object-root grammar activated after `</think>`. Reasoning
can remain free-form. Complete answers must parse as JSON objects; no fences are
stripped and no outputs are repaired. Streaming and nonstreaming requests share
that contract. A token-limited response retains `finish_reason: "length"` and
may contain incomplete JSON.

No schema API is exposed. JSON mode rejects function tools, custom stops,
`ignore_eos`, `min_tokens`, and the text-completions endpoint before admission.
Ordinary text and tool calls keep their previous behavior. Request JSON mode
explicitly; prompt wording alone does not enable a grammar.

Install the optional dependency with the serving venv's Python:

```bash
python -m pip install -r service/requirements-json.txt
./scripts/run_tp.sh --experimental-json-mode
```

Example body:

```json
{
  "model": "GLM-5.3-Flash",
  "messages": [{"role": "user", "content": "Return JSON with key code containing a Python add(a, b) function."}],
  "reasoning_effort": "high",
  "response_format": {"type": "json_object"}
}
```

`/health.json_output` identifies whether the mode is enabled. Its source paths
are pinned along with the concurrent engine hooks. Only issued, unmodified,
per-job JSON-object filters pass the concurrency guard.

## Matched comparison

Both the original eight-row binary and the sixteen-row candidate use the same
new API. Speed qualification retains the original plain-array structured
fixture, two repeats each at low and high effort, with no JSON filter requested.
This tests the native dispatch change without conflating it with grammar cost.

The separate `structured_json` cohort wraps arrays in objects; `mixed_json`
retains the original coding prompt and adds the explicit format field. These
have independent fixture hashes. A low-effort control returned `numbers`
instead of `values`, so that failed cell has no valid timing comparison. The
counting prompt and strict scoring remain unchanged. A separate functional
check runs the original coding prompt in both active slots three times at high
effort, using the original six-case behavior and nonmutation oracle. The soak
uses exact JSON-object copying plus the same coding oracle; it does not replace
the scored counting cohort. Failed answers remain failures.

Rates count completion tokens reported by the API, including reasoning.
Aggregate tokens/s is total completion tokens divided by the complete pair's
wall time. Individual end-to-end rates use each request's own wall time.
Engine decode rates use reported `predicted_ms`, excluding other request phases;
they are not kernel-only GPU rates. A faster aggregate rate does not mean each
agent receives that same rate. The server still has two active slots.

## Measured speed

Both arms use both cards. These medians cover two repetitions of the unchanged
64-integer structured pair, with no cached prompt tokens. Each serial and
concurrent pair generated exactly 385 completion tokens at low effort and 465
at high effort, including reasoning.

| Effort | Candidate C1 total tok/s | Candidate C2 total tok/s | Throughput gain | C2 median individual tok/s |
|---|---:|---:|---:|---:|
| Low | 115.8 | 133.0 | 14.9% | 76.3 |
| High | 111.9 | 133.6 | 19.4% | 70.8 |

C1 means one active request, with the pair served serially. C2 means two active
requests. Individual C1 medians were 114.2 and 111.6 tok/s. Pair finish time fell
from 3.326 to 2.895 seconds at low effort and 4.156 to 3.480 seconds at high.
This is an aggregate throughput gain; each simultaneous agent is slower than
one agent using the backend alone.

Fresh eight-row controls produced 121.4 tok/s at C2 low effort and 113.5 tok/s
at C2 high. The candidate's C2 rate was 9.6% and 17.8% higher in this small
sample. Earlier control timings varied, so these percentages describe this
window rather than a universal improvement.

The separate high-effort `mixed_json` pair passed strict JSON and the code
oracle on both builds. On the candidate, its serial/concurrent rates were
78.0/89.7 tok/s and pair times 4.041/3.768 seconds. Token counts differed
315/338; this single pair does not support the same matched-work speed claim
as the structured cohort. The low-effort `structured_json` cell failed on both
builds because of the wrong field name; it has no valid speed comparison.

## Qualification

The CPU checks cover the request contract, actual-tokenizer grammar masks,
independent/interleaved parser states, per-job ownership, HTTP response cleanup,
streamed errors, and explicit truncation. Both engine variants pass the
concurrent CPU suite (64 tests, one pre-existing skip).

GPU qualification covers JSON semantics at low/high/max effort, filtered and
unfiltered jobs together, cancellation with a surviving stream and slot reuse,
the existing tool/vision/cold-tail checks, matched throughput, and a bounded
concurrent soak. The native candidate binary is unchanged from the previously
validated sixteen-row build: only its decode capacity differs; no arithmetic
CUDA kernel was changed in this API qualification.

Upstream implementations:
[ExLlamaV3 LLGuidanceFilter](https://github.com/turboderp-org/exllamav3/blob/dev/exllamav3/generator/filter/llguidance.py),
[LLGuidance JSON-schema grammar](https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md).

## Retained qualification attempts

The first lifecycle attempt on the eight-row control reached its 2,048-token
limit. The next attempt, with a 4,096-token limit, completed a valid JSON object
but used numbered keys instead of an array. That was a semantic failure, not a
successful lifecycle qualification. Both receipts are retained. The custom
lifecycle wording was replaced with the existing structured fixture's explicit
array wording for a separately versioned qualification. Its expected answer
remains exact. The throughput fixtures, their 768-token limits and correctness
oracles were not changed between profiles or attempts. JSON-object mode
constrains syntax; it does not guarantee task semantics.

## Final runtime receipts and limits

The production canary completed **144 requests in 309.1 seconds**. Every request
terminated normally, with no engine errors or leaked leases. Sixty pairs passed
the exact-output/code oracle; twelve pairs failed because their max-effort code
request reached the unchanged 768-token limit. These remain failed task scores.
The [compact receipts](../experiments/concurrent_throughput/json_row16_2026-10-16.json)
include those responses, independent fixture hashes, per-request timings and
raw-receipt digests. This is a runtime qualification with explicit model limits.

Additional verified gates:

- Both profiles: nine exact JSON objects across low/high/max effort, independent
  filtered/unfiltered jobs, cancellation, survivor correctness and slot reuse.
- Both profiles: six concurrent high-effort coding requests passed raw JSON and
  the original six-case behavior/nonmutation oracle; the separate high-effort
  mixed fixture also passed.
- Candidate: 31 existing tool, vision, arithmetic and cold-tail requests passed.
- Candidate: six temperature-1 requests returned the exact JSON objects despite
  conflicting requests for Markdown fences; 18 batched verification rounds
  were observed.
- Disk recovery: after an intentional restart, a matching long prompt restored
  **4,864 tokens**, read a persisted recurrent state, and returned the expected
  JSON object.
- Independent final inspection: healthy, idle, zero native leases, zero service
  crash restarts, only this unit's processes on the GPUs; Qwen inactive.

Deployed native SHA-256:
`6f29d6fec981e09825fe1be463ca0f2effa4e32df24ce2108a7532e3bfb52649`.
Profile digest:
`0b5b7d3419302d0816530a9292d99dbb682a37cbdd1939d6715a1d5858f1cb6d`.
The same Weightless alpha 2 projector remains verified on both devices; cache
pool, context cap, two active slots and GLM routing are unchanged. The new
local systemd override is `63-json-row16.conf`; removing only that override in
a coordinated idle window restores the earlier unit configuration.

JSON mode guarantees an object-shaped complete response or an error. It does
not guarantee the requested keys, correct code or completion within a token
limit. Request it explicitly with `response_format`; ordinary prompt-only
Markdown behavior remains possible. Schema output is unsupported. This sample
also does not measure sustained queued traffic, large-context pressure or a
general coding-quality score for the new profile.

## Where to optimize next

The useful next measurement is a profile of the existing two-slot workload.
Keep two slots occupied with queued mixed traffic, then separate draft
generation, target verification, sampling, CPU gaps and NCCL communication
before choosing a larger kernel change. DFlash already batches
speculative tokens within a single request, so doubling request count does not
start from an unbatched target forward pass. Both C1 and C2 use both cards.

Ordinary K7 verification can require sixteen rows for two jobs; hybrid drafting
can exceed that capacity. Instrument how often dispatch crosses the fast-path
limit, then compare concurrency-aware draft width with wider fused decode.
Expert reuse and CPU overhead are additional hypotheses. Four active slots
require separate memory, context, fairness and cancellation qualification. None
of these hypotheses is a measured promise of a twofold gain.

The first candidate soak stopped on a maximum-effort coding request that spent
its entire 768-token limit in reasoning, returned an empty visible answer and
correctly reported `finish_reason: "length"`. Its failed task score is retained.
The continued runtime soak records known-answer failures separately while
preserving their scores. Protocol errors, missing usage, unhealthy identity,
engine errors and unreleased jobs still fail runtime qualification. This
separates backend stability from model task quality and does not change the
benchmark prompts, limits, validators or failed results.
