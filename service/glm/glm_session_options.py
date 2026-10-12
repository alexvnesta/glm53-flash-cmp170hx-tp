"""Pure CLI validation for the isolated LS multi-session candidate."""
from dataclasses import dataclass

GIB = 1024**3


@dataclass(frozen=True)
class SessionOptions:
    enabled: bool
    target_cpu_bytes: int
    recurrent_bytes: int
    paired_bytes: int
    max_checkpoints: int


def session_options(args):
    enabled = getattr(args, "dflash_session_cache", False)
    target = getattr(args, "target_cpu_cache_gib", 0)
    recurrent = getattr(args, "recurrent_cache_gib", 4)
    paired = getattr(args, "session_cache_gib", 1)
    checkpoints = getattr(args, "session_cache_max_checkpoints", 8)
    for name, value, minimum in (("--target-cpu-cache-gib", target, 0),
                                  ("--recurrent-cache-gib", recurrent, 1),
                                  ("--session-cache-gib", paired, 1)):
        if type(value) is not int or not minimum <= value <= 64:
            raise ValueError(f"{name} must be an integer from {minimum} through 64")
    if type(checkpoints) is not int or not 2 <= checkpoints <= 128:
        raise ValueError("--session-cache-max-checkpoints must be an integer from 2 through 128")
    if enabled and getattr(args, "dflash_prefix_cache", False):
        raise ValueError("Choose --dflash-session-cache or --dflash-prefix-cache")
    if enabled and (getattr(args, "mtp", False) or not getattr(args, "draft_model_dir", None)):
        raise ValueError("--dflash-session-cache requires a DFlash2 drafter")
    if (enabled and getattr(args, "tensor_parallel", False)
            and getattr(args, "experimental_dflash2_tp", False) is not True):
        raise ValueError("TP session retention requires the explicit experimental DFlash2 TP opt-in")
    if target and not enabled:
        raise ValueError("--target-cpu-cache-gib requires --dflash-session-cache")
    return SessionOptions(bool(enabled), target * GIB, recurrent * GIB, paired * GIB, checkpoints)
