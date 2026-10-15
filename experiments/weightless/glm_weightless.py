"""Experimental rank-one GLP projector. Not imported by the production recipe.

Original implementation of the public GLP v1 container/apply contract.
GPU dependencies are imported only when an enabled target block executes.
"""
import hashlib
import logging
import math
import os
from pathlib import Path
import re

WIDTH = 4 * 4096
LAYERS = 45
HOOK = "residual_stream_post_layer"
LOG = logging.getLogger(__name__)


def load_vector(path, expected_sha256, base_revision):
    import numpy as np
    from gguf import GGUFReader, GGMLQuantizationType

    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("GLM_WEIGHTLESS_SHA256 must pin the complete vector file")
    if not re.fullmatch(r"[0-9a-f]{40}", base_revision or ""):
        raise ValueError("GLM_WEIGHTLESS_BASE_REVISION must pin the base checkpoint")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("GLP file exceeds the 16 MiB experiment limit")
    data = path.read_bytes()
    if data[:8] != b"GGUF\x03\x00\x00\x00":
        raise ValueError("only little-endian GGUF version 3 is supported")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("GLP file SHA256 mismatch")
    # The reader maps the file. Check again after parsing to detect replacement.
    reader = GGUFReader(str(path))
    meta = {name: field.contents() for name, field in reader.fields.items()}
    if any(name.startswith("dspark.") for name in meta):
        raise ValueError("legacy dspark metadata requires re-export")
    if meta.get("glp.mode") != "project":
        raise ValueError("only explicit glp.mode=project is supported")
    if meta.get("glp.spec_version") != 1:
        raise ValueError("only GLP spec version 1 is supported")
    if meta.get("glp.rank", 1) != 1 or any(
        key in meta for key in ("glp.dir_scales", "glp.layer_scales")
    ):
        raise ValueError("rank-k and alpha scaling require a different reader")
    if meta.get("glp.hook_point") != HOOK:
        raise ValueError("GLP hook must be residual_stream_post_layer")
    if meta.get("general.base_model.0.version") != base_revision:
        raise ValueError("GLP base checkpoint revision mismatch")
    if meta.get("glp.derived_at", HOOK) != HOOK:
        LOG.warning("GLP direction was derived at %s, applied at %s", meta["glp.derived_at"], HOOK)
    vectors = {}
    for tensor in reader.tensors:
        match = re.fullmatch(r"direction\.([1-9][0-9]*)", tensor.name)
        if not match:
            raise ValueError(f"unsupported tensor {tensor.name!r}")
        layer = int(match[1])
        if layer >= LAYERS or layer in vectors:
            raise ValueError("duplicate or out-of-range GLM layer")
        if tensor.tensor_type != GGMLQuantizationType.F32 or list(tensor.shape) != [WIDTH]:
            raise ValueError("direction must be a 1D F32 HC-outer vector of width 16384")
        value = np.asarray(tensor.data, dtype=np.float32).reshape(-1).copy()
        norm = float(np.linalg.norm(value.astype(np.float64)))
        if not np.isfinite(value).all() or not math.isfinite(norm) or norm == 0:
            raise ValueError("direction must be finite and nonzero")
        vectors[layer] = (value / norm).astype(np.float32)
    if not vectors:
        raise ValueError("GLP contains no directions")
    declared = meta.get("glp.layer_ids_zero_based")
    if declared is None or sorted(int(x.strip()) for x in declared.split(",")) != sorted(vectors):
        raise ValueError("declared layers must match direction.N tensor names exactly")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("GLP changed during loading")
    return vectors


def make_projector(environ=None):
    environ = os.environ if environ is None else environ
    path = environ.get("GLM_WEIGHTLESS_VECTOR")
    if not path:
        return None
    alpha = float(environ.get("GLM_WEIGHTLESS_ALPHA", "nan"))
    if not math.isfinite(alpha) or not 0 <= alpha <= 2:
        raise ValueError("explicit GLM_WEIGHTLESS_ALPHA in [0, 2] is required")
    vectors = load_vector(path, environ.get("GLM_WEIGHTLESS_SHA256"),
                          environ.get("GLM_WEIGHTLESS_BASE_REVISION"))
    LOG.warning("EXPERIMENTAL Weightless GLP enabled: sha256=%s alpha=%s layers=%s",
                environ["GLM_WEIGHTLESS_SHA256"], alpha, sorted(vectors))
    return Projector(vectors, alpha, environ.get("GLM_WEIGHTLESS_VALIDATE") == "1",
                     environ.get("GLM_WEIGHTLESS_RECEIPTS"))


class Projector:
    def __init__(self, vectors, alpha, require_validation=False, receipts=None):
        self.vectors = vectors
        self.alpha = alpha
        self.devices = {}
        self.require_validation = require_validation
        self.validated = set()
        self.receipts = receipts

    def validate_device(self, device):
        """Run bounded numerical checks once, during collective-free warmup."""
        import json
        import numpy as np
        import torch
        from exllamav3.util.weightless_kernel import project_in_place
        rng = np.random.default_rng(20261014)
        direction = next(iter(self.vectors.values()))
        gpu_direction = torch.from_numpy(direction).to(device)
        reference_direction = direction.astype(np.float64)
        cases = []
        for rows in (1, 2, 7, 8, 31, 64):
            original = rng.standard_normal((rows, WIDTH), dtype=np.float32)
            for alpha in (0.0, 1.0, 2.0):
                value = torch.from_numpy(original.copy()).to(device)
                project_in_place(value, gpu_direction, alpha)
                actual = value.cpu().numpy()
                reference = original.astype(np.float64)
                reference -= alpha * (reference @ reference_direction)[:, None] * reference_direction
                np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
                cases.append({"rows": rows, "alpha": alpha,
                              "max_abs_error": float(np.max(np.abs(actual-reference)))})
        self.validated.add(device)
        if self.receipts:
            root = Path(self.receipts)
            root.mkdir(parents=True, exist_ok=True)
            receipt = {"pid": os.getpid(), "device": str(device), "passed": True,
                       "cases": cases, "operation": "rank_one_projection_fp32"}
            (root / f"numerics-{device.index}-{os.getpid()}.json").write_text(json.dumps(receipt, indent=2)+'\n')
        LOG.warning("Weightless GPU numerical check passed on %s: %d cases", device, len(cases))

    def __call__(self, block, x, params):
        # DFlash and vision blocks do not have the target mHC architecture.
        if self.alpha == 0 or block.attn_hc is None or block.layer_idx not in self.vectors:
            return x
        if params.get("layer_instance", 0) != 0:
            return x
        if "capture" in params or "quant_preserve" in params:
            raise ValueError("steering is not supported during model conversion")
        import torch
        if (x.ndim != 4 or tuple(x.shape[-2:]) != (4, 4096)
                or x.dtype != torch.float32 or not x.is_contiguous() or x.device.type != "cuda"):
            raise ValueError("steering requires CUDA contiguous FP32 [B,S,4,4096] residuals")
        if self.require_validation and x.device not in self.validated:
            if "backend" in params and not params.get("tp_warmup"):
                raise RuntimeError("Weightless numerical validation must precede TP serving")
            self.validate_device(x.device)
        # Stock TP/export paths already materialize the residual. This also
        # handles the engine's deferred mHC path if it is enabled in the future.
        from exllamav3.modules.hyperconnections import hc_flush
        hc_flush(params)
        key = (block.layer_idx, x.device)
        direction = self.devices.get(key)
        if direction is None:
            direction = torch.from_numpy(self.vectors[block.layer_idx]).to(x.device)
            self.devices[key] = direction
        from exllamav3.util.weightless_kernel import project_in_place
        project_in_place(x, direction, self.alpha)
        return x
