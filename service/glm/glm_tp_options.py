"""Explicit default-off TP+DFlash2 service option. No engine or GPU imports at import time.

P16: upstream dev runs DFlash2 against a tensor-parallel target natively (9cc3bd1, e98785b: sharded LM-head top-k,
always on, no opt-in). Engine patch 0005 and its model_init flag are gone, so --experimental-dflash2-tp is now a
service flag with the same name and dest: the operator's explicit opt-in to that profile. It changes nothing in the
engine; the service checks the same preconditions before load and, after the generator attaches the drafter, that
the drafter is attached to this TP target.
"""
from pathlib import Path

FLAG = "--experimental-dflash2-tp"


def add_tp_dflash_argument(parser):
    """The flag now belongs to the service parser; an engine that still adds it is not the P16 engine."""
    if any(FLAG in action.option_strings for action in parser._actions):
        raise ValueError(f"The engine argument parser already defines {FLAG}: this service needs the P16 engine "
                         "(upstream native DFlash2-TP), not engine patch 0005")
    parser.add_argument(FLAG, dest="experimental_dflash2_tp", action="store_true", default=False,
                        help="Explicit opt-in to DFlash2 drafting for a tensor-parallel target (upstream native "
                             "DFlash2-TP); requires -tp and a separate DFlash2 drafter")


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
        model_dir = getattr(args, "model_dir", None)
        if model_dir and Path(args.draft_model_dir).resolve() == Path(model_dir).resolve():
            raise ValueError("--experimental-dflash2-tp cannot enable MTP/shared target weights")
        if getattr(args, "cpu_cache_size", 0):
            raise ValueError("Generic CPUPageCache also includes the draft ring; TP profile requires cpu_cache_size=0")
    return enabled


def require_native_tp_dflash():
    """Before load: the engine must carry upstream's native DFlash2-TP path (sharded LM-head top-k)."""
    from exllamav3.model.model_tp import Model_TPMixin
    from exllamav3.architecture.dflash2 import DFlash2Model
    if not callable(getattr(Model_TPMixin, "tp_dispatch_lm_head_topk", None)) or not hasattr(DFlash2Model, "attach_to"):
        raise ValueError("The engine has no native tensor-parallel DFlash2 path (upstream 9cc3bd1/e98785b)")


def require_tp_dflash_attached(generator):
    """After the generator attached the drafter: it is a DFlash2 drafter attached to this exact TP target."""
    if not tp_dflash_health(generator):
        raise ValueError("The engine did not attach the DFlash2 drafter to the tensor-parallel target")


def tp_dflash_health(generator):
    draft = getattr(generator, "draft_model", None)
    attached = getattr(draft, "attached_model", None)
    return bool(getattr(generator.model, "loaded_tp", False)
                and getattr(generator, "dflash_draft", False)
                and draft is not None and draft.caps.get("dflash_draft")
                and callable(attached) and attached() is generator.model)
