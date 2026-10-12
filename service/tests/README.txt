CPU tests for the service. Run from service/ with the patched engine on PYTHONPATH and CUDA hidden, e.g.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=../native:../engine-src:glm python -m unittest discover -s tests -v

Removed before publication: test_duplicate_recycle, test_sessions, test_session_startup, test_tp_sessions, test_tp_startup and
test_install_hooks. They predate the final engine (stale test doubles and pinned stock-engine sources) and failed identically on
the unmodified production tree, so they would only report false breakage.
