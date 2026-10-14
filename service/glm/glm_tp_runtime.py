"""TP host settings that engine patch 0005 used to carry, now applied by the service (P16).

Upstream dev keeps a fixed 20 s parent wait for TP worker replies (model_tp.DISPATCH_TIMEOUT) and always selects
torch's file_system sharing strategy in the parent (Model_TPMixin.create_tp_context). Production qualified a 300 s wait
(cold Triton/compiled-tensor caches) and file_descriptor sharing (the pilot's shutdown leaked shm objects under
file_system). The unit's environment variables keep their names and meaning:

- EXL3_TP_WORKER_TIMEOUT: seconds the parent waits for a worker reply. Set on the module global before load; upstream
  reads it at call time in tp_worker_dispatch_single/tp_worker_result. Unset leaves upstream's 20 s.
- EXL3_TP_SHARING_STRATEGY: file_system or file_descriptor (Linux). The parent strategy is installed inside
  create_tp_context, in place of upstream's file_system, before any worker is spawned. Spawned ranks start with
  torch's platform default (file_descriptor on Linux); after load each rank is set to the requested strategy and every
  rank reports it, so a mismatch stops the start. Unset leaves upstream's behavior untouched.

Importing does not load Torch or the engine.
"""
import math
import os
import sys

TIMEOUT_ENV = "EXL3_TP_WORKER_TIMEOUT"
SHARING_ENV = "EXL3_TP_SHARING_STRATEGY"
STRATEGIES = ("file_system", "file_descriptor")


def worker_timeout(environ=None):
    environ = os.environ if environ is None else environ
    raw = environ.get(TIMEOUT_ENV)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise ValueError(TIMEOUT_ENV + " must be finite and positive")
    return value


def sharing_strategy(environ=None, platform=None):
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    strategy = environ.get(SHARING_ENV)
    if strategy is None:
        return None
    if strategy not in STRATEGIES:
        raise ValueError(SHARING_ENV + " must be file_system or file_descriptor")
    if strategy == "file_descriptor" and platform != "linux":
        raise ValueError("Explicit TP file_descriptor mode requires Linux")
    return strategy


def settings(environ=None):
    """Validate both settings without importing the engine (CLI parse time)."""
    return dict(worker_timeout=worker_timeout(environ), sharing_strategy=sharing_strategy(environ))


def install(environ=None):
    """Before model load: apply the timeout and arrange the parent strategy. Returns the validated settings."""
    chosen = settings(environ)
    from exllamav3.model import model_tp
    if chosen["worker_timeout"] is not None:
        model_tp.DISPATCH_TIMEOUT = chosen["worker_timeout"]
    strategy = chosen["sharing_strategy"]
    if strategy is not None:
        _install_parent_strategy(model_tp.Model_TPMixin, strategy)
    return chosen


def _install_parent_strategy(mixin, strategy):
    import torch.multiprocessing as tmp
    original = mixin.create_tp_context
    installed = getattr(original, "_glm_tp_sharing", None)
    if installed is not None:
        if installed != strategy:
            raise ValueError("TP sharing strategy was already installed with a different value")
        return
    if strategy not in tmp.get_all_sharing_strategies():
        raise ValueError("Requested TP sharing strategy is unavailable: " + strategy)

    def create_tp_context(self, *args, **kwargs):
        real = tmp.set_sharing_strategy
        def select(name):
            # Upstream's only call here is set_sharing_strategy("file_system"); anything else means a changed engine.
            if name != "file_system":
                raise RuntimeError("Unexpected TP sharing strategy request from the engine: " + repr(name))
            real(strategy)
        tmp.set_sharing_strategy = select
        try:
            return original(self, *args, **kwargs)
        finally:
            tmp.set_sharing_strategy = real
    create_tp_context._glm_tp_sharing = strategy
    mixin.create_tp_context = create_tp_context


def mp_sharing_strategy(local_context, requested):
    """On every TP rank (and the output pseudo-rank in the parent): set the requested strategy, return what is set."""
    import torch.multiprocessing as tmp
    if requested is not None and tmp.get_sharing_strategy() != requested:
        tmp.set_sharing_strategy(requested)
    return tmp.get_sharing_strategy()


def verify(model, chosen):
    """After load: the timeout is in force, and the parent and every rank use the requested strategy."""
    import torch.multiprocessing as tmp
    from exllamav3.model import model_tp
    report = dict(worker_timeout_s=float(model_tp.DISPATCH_TIMEOUT), sharing_strategy=chosen["sharing_strategy"],
                  parent_sharing=tmp.get_sharing_strategy(), rank_sharing=None)
    if chosen["worker_timeout"] is not None and model_tp.DISPATCH_TIMEOUT != chosen["worker_timeout"]:
        raise RuntimeError("TP worker timeout was not applied")
    strategy = chosen["sharing_strategy"]
    if strategy is None or not getattr(model, "loaded_tp", False):
        return report
    if report["parent_sharing"] != strategy:
        raise RuntimeError("TP parent sharing strategy is not the requested one")
    model.tp_drain_acks()
    devices = tuple(model.active_devices)
    for device in devices:
        model.tp_worker_dispatch(device, mp_sharing_strategy, (strategy,))
    ranks = [model.tp_worker_result(device) for device in devices]
    if any(rank != strategy for rank in ranks):
        raise RuntimeError(f"TP rank sharing strategies {ranks} differ from the requested {strategy}")
    report["rank_sharing"] = ranks
    return report
