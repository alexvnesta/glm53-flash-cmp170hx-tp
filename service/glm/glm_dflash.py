"""Local DFlash2 integration: keep the drafter on one device.

P16 removed preserve_fused_exports (k_hcfuse) and bounded_draft_cache (the single-sequence draft ring); upstream's
sliding-window draft state bounds the drafter now, also in the concurrent profile (glm_native_concurrency, stage 2).
"""


def pin_drafter(device=1):
    """Keep the small drafter beside the target head, avoiding vocab-logit copies."""
    from exllamav3.model.model import Model
    import torch

    original = Model.load
    if getattr(original, "_glm_dflash_device", False):
        return

    def load(self, *args, **kwargs):
        if self.caps.get("dflash_draft"):
            kwargs.pop("use_per_device", None)
            kwargs.pop("reserve_per_device", None)
            kwargs["device"] = torch.device(f"cuda:{device}")
        return original(self, *args, **kwargs)

    load._glm_dflash_device = True
    Model.load = load
