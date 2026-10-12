# Serialized request admission and response lifetime

The API admits a bounded FIFO queue before creating a native generation job.
One caller owns the existing GPU lock. Full queues return 429, expired waits
return 429, and disconnected queued callers are removed without cancelling the
current owner. `GLM53_REQUEST_QUEUE_SIZE` defaults to 8; `GLM53_REQUEST_WAIT_SECONDS`
defaults to 120. Validation rejects invalid or unbounded policies.

An admitted streaming job now shares one `OwnedJobCleanup` between the iterator
and the entire ASGI response. Header/send failure can happen before an iterator
runs its `finally`. The response wrapper covers that boundary, drains the owned
native cancellation despite repeated asyncio/AnyIO cancellation, and releases
the GPU lock once. Cancellation errors remain observable. Ordinary successful
SSE text and semantic usage stay unchanged in the CPU API-function comparison.

## Portable CPU checks

```sh
python -m pip install -r requirements-http-test.txt
python scripts/run_http_cpu_tests.py
```

This runs eight admission contracts, twelve real Starlette/AnyIO ASGI contracts,
and four API-function comparisons: historical/candidate, normal/header failure.
The immutable historical fixture contains only the actual MIT-licensed
`complete` and `stream_text` function source, with provenance and SHA256. The
candidate functions are read from this repository's actual `glm/glm_api.py`.
Only model/tokenizer/job objects are CPU stand-ins; no engine, Torch, GPU,
network or service is imported or contacted.

The standalone comparator accepts multiple explicit candidate files. Running it
once against both publication branches yields the original six comparisons:

```sh
python scripts/check_response_lifetime_ast.py \
  --candidate /PATH/LS_SERVICE/glm/glm_api.py \
  --candidate /PATH/TP_SERVICE/glm/glm_api.py --output /PATH/NEW_RESULT.json
```

These tests cover ownership and ASGI response boundaries. They do not establish
CUDA cancellation/DMA completion or full-model quality. Runtime qualification
requires an owned real SSE disconnect, idle recovery, and a following successful
request on the exact source and native build being deployed. Raw machine
receipts, service controllers, native binaries and model files are excluded.
