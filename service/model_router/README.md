# GLM-default model router

This is the existing local GLM/Qwen switcher, captured from the deployed
October 13 implementation with the October 17 backend-reload fix. GLM remains
the default; Qwen requires an explicit model ID. The controller holds a request
lease through the last streamed byte and serializes model ownership changes.

An operator restart can reload the selected GLM backend without changing the
router's selected model. Before forwarding a POST, the router now waits for that
backend's existing readiness check. Streaming headers arrive immediately and
15-second comments keep the connection active while loading. A generation is
forwarded once; forwarded POSTs are never retried. Disconnecting during the wait
releases the lease.

Unready `/health` responses retain status 503 and expose the switcher's inflight
and waiting counts. This lets an operator yield test load to arriving clients
even when backend health is unavailable.

The live source before this fix had SHA256
`47b5a6929fa261996a08698f238742577eec876ed3c4cc3da2f865c26c0da86a`;
the fixed `switcher.py` has SHA256
`51a1008307b004d7fa49971b428f6cff84e0d181cd4d18bd919a84e59860103d`.
The controller and the existing auth/header boundaries are preserved. Caller
bearer tokens are removed before the private GLM backend; the Qwen backend key
is read from the configured file, never embedded in this source.

Run `python -m unittest discover -s service/model_router -v` with the HTTP
dependencies used by CI. The 25 tests use mocked services and transports; they
check admission, cancellation, reload waits, streaming and ownership ordering.
They do not start either GPU model.
