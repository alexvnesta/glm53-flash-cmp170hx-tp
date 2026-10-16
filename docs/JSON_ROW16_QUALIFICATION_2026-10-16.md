# JSON-object mode and sixteen-row decode qualification

Status: CPU checks passed; GPU qualification in progress. The original
production backend remains the automatic rollback target. This report will be
updated with the qualification receipts before any promotion.

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
new API and explicit JSON-mode fixtures. `structured_json` wraps the arrays in
objects; `mixed_json` retains the original coding prompt and adds the explicit
format field. These are separate cohorts from the earlier prompt-only results.
Low, high and max effort are reported separately. The original frozen oracles
are retained; failed waves are not timing wins.

Rates count completion tokens reported by the API, including reasoning.
Aggregate tokens/s is total completion tokens divided by the complete pair's
wall time. Individual end-to-end rates use each request's own wall time.
Engine decode rates use reported `predicted_ms`, excluding other request phases;
they are not kernel-only GPU rates. A faster aggregate rate does not mean each
agent receives that same rate. The server still has two active slots.

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
