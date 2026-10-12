"""Default-off image input options, validated before model allocation.

--experimental-vision accepts base64 data: image parts in user messages (glm_vision.py). Without the flag every vision
setting is refused, and with it every value is range-checked here, so a bad value stops the service at start.
"""
import re
import sys
from dataclasses import dataclass

MIB = 1024 * 1024
# processor_config.json of the deployed checkpoint: min_image_tokens / max_image_tokens per image.
MODEL_MIN_IMAGE_TOKENS, MODEL_MAX_IMAGE_TOKENS = 16, 8000
# Pillow 12.3.0 is the minimum with the decoder-bomb and out-of-bounds fixes this decode path relies on.
MIN_PILLOW = (12, 3, 0)
# Option name: (default, minimum, maximum, help). Defaults are conservative starting values; set the token cap and the
# cache from measured VRAM and host RAM.
LIMITS = {
    "vision_max_images": (8, 1, 32, "images per request"),
    "vision_max_image_bytes": (8 * MIB, 1024, 32 * MIB, "decoded bytes per image"),
    "vision_max_request_image_bytes": (24 * MIB, 1024, 128 * MIB, "decoded image bytes per request"),
    "vision_max_image_pixels": (16777216, 64, 89478485, "width x height per image, checked before decoding"),
    "vision_max_image_side": (8192, 28, 16384, "longest image side in pixels"),
    "vision_max_image_tokens": (MODEL_MAX_IMAGE_TOKENS, MODEL_MIN_IMAGE_TOKENS, MODEL_MAX_IMAGE_TOKENS,
                                "vision tokens per image; larger images are scaled down to fit"),
    "vision_cache_mib": (512, 0, 8192, "content-keyed embedding cache in host RAM, MiB; 0 disables reuse"),
    "vision_decode_timeout": (10, 1, 60, "seconds for one isolated image decode"),
    "vision_decode_memory_mib": (1024, 256, 8192, "address-space limit of one decode process, MiB"),
    "vision_decode_workers": (2, 1, 8, "concurrent decode processes"),
    "vision_device": (1, 0, 7, "CUDA device of the single-device vision tower"),
}
# auto: run each decode in a bubblewrap sandbox (no network, only the interpreter and the worker mounted) when a probe at
# start succeeds, else rlimits only; bwrap: refuse to start without it; none: rlimits only.
SANDBOXES = ("auto", "bwrap", "none")


@dataclass(frozen=True)
class VisionOptions:
    enabled: bool = False
    max_images: int = 0
    max_image_bytes: int = 0
    max_request_image_bytes: int = 0
    max_image_pixels: int = 0
    max_image_side: int = 0
    max_image_tokens: int = 0
    cache_bytes: int = 0
    decode_timeout: int = 0
    decode_memory_bytes: int = 0
    decode_workers: int = 0
    device: int = 0
    # The decode process must confirm its RLIMIT_AS; only tests on a non-Linux host relax this.
    require_memory_limit: bool = True
    decode_sandbox: str = "none"
    # Image embeddings are set aside once a job's prompt is prefilled, so decode steps stop carrying them.
    keep_embeddings: bool = False

    def body_allowance(self):
        """Request-body bytes added for images: base64 (4 per 3 bytes) of the request image budget, plus framing."""
        return -(-self.max_request_image_bytes // 3) * 4 + 1024 * self.max_images if self.enabled else 0

    def limits(self):
        return {"max_images": self.max_images, "max_image_bytes": self.max_image_bytes,
                "max_request_image_bytes": self.max_request_image_bytes, "max_image_pixels": self.max_image_pixels,
                "max_image_side": self.max_image_side, "max_image_tokens": self.max_image_tokens,
                "formats": ["png", "jpeg", "webp"], "url_schemes": ["data"], "roles": ["user"],
                "decode_sandbox": self.decode_sandbox,
                "embeddings_after_prefill": "kept" if self.keep_embeddings else "dropped"}


def add_vision_arguments(parser):
    parser.add_argument("--experimental-vision", action="store_true",
                        help="Opt in to image input: base64 data: URLs in user messages only, never fetched")
    for name, (default, low, high, text) in LIMITS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            help=f"Vision: {text} (default {default}, {low}..{high}); requires --experimental-vision")
    parser.add_argument("--vision-decode-sandbox", help="Vision: auto (default; bubblewrap when it works, else rlimits "
                        "only), bwrap (required) or none; requires --experimental-vision")
    parser.add_argument("--vision-keep-embeddings", action="store_true", default=None,
                        help="Vision: keep image embeddings on every decode step instead of dropping them after "
                        "prefill (measurement only); requires --experimental-vision")


def pillow_version():
    try:
        import PIL
        from PIL import features
    except ImportError:
        return None, False
    return PIL.__version__, bool(features.check("webp"))


def vision_options(args, pillow=None, platform=sys.platform):
    enabled = getattr(args, "experimental_vision", False)
    if type(enabled) is not bool:
        raise ValueError("experimental_vision must be a bool")
    given = {name: getattr(args, name, None) for name in LIMITS}
    sandbox = getattr(args, "vision_decode_sandbox", None)
    keep = getattr(args, "vision_keep_embeddings", None)
    if keep not in (None, True):
        raise ValueError("vision_keep_embeddings must be a flag")
    if not enabled:
        if any(value is not None for value in given.values()) or sandbox is not None or keep is not None:
            raise ValueError("Vision settings require --experimental-vision")
        return VisionOptions()
    values = {}
    for name, (default, low, high, _) in LIMITS.items():
        value = default if given[name] is None else given[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"--{name.replace('_', '-')} must be an integer from {low} through {high}")
        values[name] = value
    sandbox = "auto" if sandbox is None else sandbox
    if sandbox not in SANDBOXES:
        raise ValueError(f"--vision-decode-sandbox must be one of {', '.join(SANDBOXES)}")
    if values["vision_max_image_bytes"] > values["vision_max_request_image_bytes"]:
        raise ValueError("--vision-max-image-bytes must not exceed --vision-max-request-image-bytes")
    # Python, Pillow and about four RGBA copies of the largest allowed raster must fit under the decode limit.
    if values["vision_decode_memory_mib"] * MIB < 256 * MIB + 16 * values["vision_max_image_pixels"]:
        raise ValueError("--vision-decode-memory-mib must be at least 256 MiB plus 16 bytes per allowed pixel")
    if not platform.startswith("linux"):
        raise ValueError("--experimental-vision isolates decoding with RLIMIT_AS and needs Linux")
    version, webp = pillow_version() if pillow is None else pillow
    if version is None or tuple(int(p) for p in re.findall(r"\d+", version)[:3]) < MIN_PILLOW:
        raise ValueError(f"--experimental-vision needs Pillow >= {'.'.join(map(str, MIN_PILLOW))} (found {version})")
    if not webp:
        raise ValueError("--experimental-vision needs Pillow built with WebP support")
    return VisionOptions(
        True, values["vision_max_images"], values["vision_max_image_bytes"], values["vision_max_request_image_bytes"],
        values["vision_max_image_pixels"], values["vision_max_image_side"], values["vision_max_image_tokens"],
        values["vision_cache_mib"] * MIB, values["vision_decode_timeout"], values["vision_decode_memory_mib"] * MIB,
        values["vision_decode_workers"], values["vision_device"], decode_sandbox=sandbox, keep_embeddings=keep is True)
