# CPU contracts and response lifecycle checks

Run from the repository root after `engine/apply_engine.sh` and environment
setup. These tests use CPU tensors or source ASTs and model stand-ins. They do
not load weights, initialize CUDA, start a serving port, or replace GPU
qualification of cache writes and model continuation.

```bash
export CUDA_VISIBLE_DEVICES=
export EXLLAMAV3_ENGINE_ROOT="$PWD/engine-src"
export PYTHONPATH="$PWD/service/glm:$PWD/service/tests"
.venv/bin/python -m unittest test_dflash_options test_prefill_chunk -v
.venv/bin/python -m unittest test_tp_options test_tp_lifecycle test_tp_distributed_fixture -v
.venv/bin/python -m unittest discover -s service/tests/concurrent_cpu -v
.venv/bin/python -m unittest discover -s service/tests/http -v
.venv/bin/python service/scripts/check_response_lifetime_ast.py
```

The October 13 checks ran 19 source-contract cases, 27 TP cases, 63 concurrency
cases (one skip), 20 HTTP cases and four response-lifetime AST cases. CUDA
remained uninitialized in the tensor/source suites; the HTTP/AST checks did not
import Torch. GitHub CI runs source reproduction, syntax and the HTTP/AST checks
without downloading model weights or building a GPU extension.

`service/docs/qualification_summary.json` is the historical October 9 one-active TP
fixture record used by provenance tests. Its 64K failure and bounded scope stay
visible. It is not the qualification record for the current native-SWA,
two-active profile; use the root recipe's dated qualification documentation.

The October 16 four-slot experiment adds independent state/reuse, fifth-job
rejection, capacity-drift and shared-prefix admission cases. Its exact row32
CPU run executes 68 contracts with one native-only SAM skip and leaves CUDA
uninitialized. See the root recipe's `docs/FOUR_ACTIVE_2026-10-16.md` for the
separate GPU benchmark and bounded lifecycle evidence; four-slot production
soak, long-context and cache-restart qualification remain outstanding.
