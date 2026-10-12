"""Explicit default-off TP+DFlash2 service option. No engine or GPU imports."""


def tp_dflash_options(args):
    enabled = getattr(args, "experimental_dflash2_tp", False)
    if type(enabled) is not bool:
        raise ValueError("experimental_dflash2_tp must be a bool")
    target_tp = getattr(args, "tensor_parallel", False)
    dflash = bool(getattr(args, "draft_model_dir", None)) and not getattr(args, "mtp", False)
    if target_tp and dflash and not enabled:
        raise ValueError("TP DFlash2 requires the explicit --experimental-dflash2-tp opt-in")
    if enabled:
        if target_tp is not True or not dflash:
            raise ValueError("--experimental-dflash2-tp requires -tp and a DFlash2 drafter")
        if getattr(args, "cpu_cache_size", 0):
            raise ValueError("Generic CPUPageCache also includes the draft ring; TP profile requires cpu_cache_size=0")
    return enabled


def tp_dflash_health(generator):
    draft = getattr(generator, "draft_model", None)
    return bool(getattr(generator.model, "loaded_tp", False)
                and getattr(generator, "dflash_draft", False)
                and draft is not None
                and getattr(draft.config, "experimental_tensor_parallel", False) is True)
