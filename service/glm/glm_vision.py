"""Image input for the GLM API, off unless --experimental-vision (glm_vision_options.py).

Only base64 data: URLs in user messages are accepted; nothing is ever fetched. Before admission an image costs no decoder
work: the payload size, the format (PNG/JPEG/WebP magic bytes) and the header dimensions are checked in pure Python, and
the vision-token cost comes from the engine's own canvas arithmetic, so the prompt length is exact and an oversized
request is refused before it takes a slot. After admission each new image is decoded in a throwaway process
(glm_vision_worker.py) under memory, CPU and wall-clock limits, then encoded by the tower on the generator's GPU thread
under its step lock, so it never interleaves with a native step.

Tower outputs are cached by content. The engine gives every MMEmbedding fresh token ids from a process-global counter
and pages are hashed by token id, so only a reused embedding object lets a re-sent image keep its ids, its page hashes
and its B2 prefix plans. Equal ids always mean equal content. The converse need not hold: two requests that bring the
same new image at the same moment may each encode it, and the second copy is used uncached.

Each image part becomes a text part holding a per-content alias that the tokenizer expands to [begin] + ids + [end];
the model's chat template therefore never reaches its "unable to process this image" branch.
"""
import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "glm_vision_worker.py"
# 1x1 images the start-time sandbox probe decodes, one per accepted format (each codec must load inside the sandbox).
PROBE_IMAGES = {fmt: base64.b64decode(data) for fmt, data in (
    ("PNG", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="),
    ("JPEG", "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgy"
             "PC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA"
             "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0Kx"
             "wRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip"
             "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
             "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldY"
             "WVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3"
             "+Pn6/9oADAMBAAIRAxEAPwDkKKKK8U/TD//Z"),
    ("WEBP", "UklGRjoAAABXRUJQVlA4IC4AAACwAQCdASoBAAEAAUAmJaACdLoABDAAAP7x3I/4DdfFtMv/vYL/3YL/3YL/WwAA"))}
# Preprocessing of the deployed checkpoint (processor_config.json, engine Glm5NextVisionModel.preprocess). load_tower
# refuses to start when the loaded config differs, because image_tokens() depends on these values.
PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE, MIN_IMAGE_TOKENS = 14, 2, 2, 16
FACTOR = PATCH_SIZE * MERGE_SIZE      # one vision token per 28x28 px of the resized canvas
BOUNDARY_TOKENS = 2                   # <|begin_of_image|> and <|end_of_image|> around every image
DECLARED_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/webp")
OTHER_FORMATS = ((b"GIF8", "GIF"), (b"BM", "BMP"), (b"II*\0", "TIFF"), (b"MM\0*", "TIFF"), (b"%PDF", "PDF"),
                 (b"8BPS", "PSD"), (b"\0\0\0\x0cjP  ", "JPEG 2000"), (b"\xff\x4f\xff\x51", "JPEG 2000"),
                 (b"SIMPLE", "FITS"), (b"\0\0\x01\0", "ICO"), (b"qoif", "QOI"), (b"<", "SVG/XML"))
JPEG_FRAME_MARKERS = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# Header scans run on the event loop before admission, so hostile layouts must not cost more than a few hundred steps.
# Real files reach IDAT or the frame header within a few dozen chunks or segments.
MAX_HEADER_CHUNKS, MAX_JPEG_FILL_BYTES = 256, 64
# The engine files the vision path executes, checked before the tower loads.
VISION_ENGINE_SOURCE_PINS = {
    'exllamav3/tokenizer/mm_embedding.py': '7d2a0a2a01aa6d292fcc4e3fa64216f75ae65170a2d455c4b14435135a57b0a4',
    'exllamav3/tokenizer/tokenizer.py': '12b1115b806aedd8dea2eb652a7f9830d50a0056ced8603c1306d311e28e4c3a',
    'exllamav3/architecture/glm4v.py': '6814630cfd7404ed1bca73b3fc169c8afc5b985a49e01a8191878f0dab054091',
    'exllamav3/architecture/glm5_next.py': '680f95b2a090bc51e370664366e8222734981d73a0d93a730b6c09d8f541b066',
    'exllamav3/architecture/mm_processing/glm5_next.py': 'a83c72bbb348e97c36453b06b27f92ad4e8b5b3a2b07ea2623a8529a1bb0b5c3',
    'exllamav3/architecture/mm_processing/common.py': 'b499d13f6ccb26b3d57cc585fb7c5d871fddef1ca3adc7044b8eb72fa8b97da5',
    'exllamav3/modules/embedding.py': '409a1f60526aa44d1c56b7d578c2d258ba657b98f8b471135590f568442e5c1b',
    # glm_concurrent_ring.drop_prefilled_embeddings relies on iterate_gen collecting job.embeddings per decode
    # step and on the requeue path staying unreachable for ring jobs; a changed generator must be re-qualified.
    'exllamav3/generator/generator.py': 'add0fa6c149c38eaa55ab7ed6cc43505ca32778d61bac9af8ca63e48b7f11181',
    'exllamav3/generator/job.py': 'b587612432e482d262776afb79a6a007355b2cad3ae4b8aaaf52e567d7c4df8e'}


class ImageRejected(ValueError):
    """The request's image cannot be used; the API answers 400 with this message."""


class VisionFailure(RuntimeError):
    """The server failed to encode an acceptable image; the API answers 500."""


@dataclass(frozen=True)
class ImagePart:
    key: str        # sha256 of the encoded bytes and the preprocessing budget
    alias: str      # prompt placeholder the tokenizer expands; secret per process, stable per content
    data: bytes
    format: str
    width: int
    height: int
    tokens: int     # vision tokens plus the two boundary tokens


def decode_data_url(url, max_bytes):
    if not isinstance(url, str):
        raise ImageRejected("image_url.url must be a string")
    if url[:5].lower() != "data:":
        raise ImageRejected("Only base64 data: image URLs are accepted; this server does not fetch http(s), file or other URLs")
    comma = url.find(",", 5, 5 + 256)
    fields = [field.strip().lower() for field in url[5:comma].split(";")] if comma > 0 else []
    if len(fields) < 2 or fields[-1] != "base64":
        raise ImageRejected("Image data: URLs must be base64-encoded (data:image/png;base64,...)")
    if fields[0] not in DECLARED_TYPES:
        raise ImageRejected(f"Unsupported image type {fields[0][:40]!r}; send PNG, JPEG or WebP")
    limit = -(-max_bytes // 3) * 4
    payload = url[comma + 1:]
    if len(payload) > limit + limit // 32 + 4:      # room for MIME line breaks only
        raise ImageRejected(f"Image exceeds {max_bytes} bytes")
    payload = re.sub(r"[ \t\r\n]", "", payload)
    if len(payload) > limit:
        raise ImageRejected(f"Image exceeds {max_bytes} bytes")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):       # ValueError: non-ASCII characters
        raise ImageRejected("Image data is not valid base64") from None
    if len(data) > max_bytes:
        raise ImageRejected(f"Image exceeds {max_bytes} bytes")
    return data


def sniff(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    known = next((name for magic, name in OTHER_FORMATS if data.startswith(magic)), None)
    if known is None and data[4:8] == b"ftyp":
        known = "HEIF/AVIF"
    raise ImageRejected(f"Image format {known} is not supported; send PNG, JPEG or WebP" if known
                        else "Image data is not PNG, JPEG or WebP")


def png_size(data):
    if len(data) < 33 or data[8:16] != b"\0\0\0\rIHDR":
        raise ImageRejected("Truncated or malformed PNG header")
    width, height = struct.unpack(">II", data[16:24])
    position = 33
    for _ in range(MAX_HEADER_CHUNKS):      # an APNG declares acTL before its first IDAT
        if position + 8 > len(data):
            return width, height
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        if kind == b"acTL":
            raise ImageRejected("Animated images are not supported; send a single frame")
        if kind in (b"IDAT", b"IEND"):
            return width, height
        position += 12 + length
    raise ImageRejected(f"PNG has more than {MAX_HEADER_CHUNKS} chunks before its image data")


def jpeg_size(data):
    position = 2
    for _ in range(MAX_HEADER_CHUNKS):
        if position + 4 > len(data) or data[position] != 0xFF:
            break
        fill = 0
        while data[position + 1] == 0xFF and fill < MAX_JPEG_FILL_BYTES and position + 5 <= len(data):
            position += 1                   # fill bytes before a marker
            fill += 1
        marker = data[position + 1]
        if marker == 0xFF:
            raise ImageRejected(f"JPEG has more than {MAX_JPEG_FILL_BYTES} fill bytes before a marker")
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            position += 2
            continue
        if marker in (0xD9, 0xDA):          # end of image or start of scan before any frame header
            break
        length = struct.unpack(">H", data[position + 2:position + 4])[0]
        if marker in JPEG_FRAME_MARKERS and length >= 7 and position + 9 <= len(data):
            height, width = struct.unpack(">HH", data[position + 5:position + 9])
            if not height:
                raise ImageRejected("JPEG images that declare their height after the scan are not supported")
            return width, height
        if length < 2:
            break
        position += 2 + length
    else:
        raise ImageRejected(f"JPEG has more than {MAX_HEADER_CHUNKS} segments before its frame header")
    raise ImageRejected("Truncated or malformed JPEG header")


def webp_size(data):
    if len(data) < 30:
        raise ImageRejected("Truncated or malformed WebP header")
    chunk = data[12:16]
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        if data[20] & 0x02:
            raise ImageRejected("Animated images are not supported; send a single frame")
        return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    raise ImageRejected("Truncated or malformed WebP header")


HEADER_SIZE = {"PNG": png_size, "JPEG": jpeg_size, "WEBP": webp_size}


def vision_canvas(height, width, max_tokens, min_tokens=MIN_IMAGE_TOKENS):
    """Copy of the engine's glm5_vision_canvas (exllamav3/architecture/mm_processing/glm5_next.py) for one still
    image, which preprocess() sizes as temporal_patch_size frames. test_vision.py checks it against the pinned file."""
    num_frames, temporal_factor, factor = TEMPORAL_PATCH_SIZE, TEMPORAL_PATCH_SIZE, FACTOR
    pixels_per_token = temporal_factor * factor ** 2
    min_pixels = min_tokens * pixels_per_token
    max_pixels = max_tokens * pixels_per_token

    def align(value, f):
        return math.ceil(value / f) * f

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    budget = aligned_frames * aligned_height * aligned_width
    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        budget = aligned_frames * aligned_height * aligned_width
    if budget > max_pixels:
        low, high = 1, height
        best_h = best_w = factor
        while low <= high:
            ch = (low + high) // 2
            cw = max(1, math.floor(width * ch / height))
            cand_h, cand_w = align(ch, factor), align(cw, factor)
            if aligned_frames * cand_h * cand_w <= max_pixels:
                best_h, best_w = cand_h, cand_w
                low = ch + 1
            else:
                high = ch - 1
        aligned_height, aligned_width = best_h, best_w
    return aligned_height, aligned_width


def image_tokens(width, height, max_tokens):
    canvas_height, canvas_width = vision_canvas(height, width, max_tokens)
    return canvas_height * canvas_width // (FACTOR * FACTOR) + BOUNDARY_TOKENS


def embedding_bytes(embedding):
    tensor = embedding.embeddings
    return tensor.numel() * tensor.element_size()


class EmbeddingCache:
    """Byte-bounded LRU of tower outputs keyed by content. Entries the current request uses are never evicted while
    it inserts its other images; an evicted entry a running job still holds stays alive with that job."""
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key, embedding, protected=()):
        size = embedding_bytes(embedding)
        if key in self.entries or size > self.max_bytes:
            return
        self.entries[key] = (embedding, size)
        self.bytes += size
        for victim in list(self.entries):
            if self.bytes <= self.max_bytes:
                break
            if victim != key and victim not in protected:
                self.bytes -= self.entries.pop(victim)[1]
                self.evictions += 1

    def stats(self):
        return {"entries": len(self.entries), "bytes": self.bytes, "budget_bytes": self.max_bytes,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions}


def verify_vision_sources(root):
    root = Path(root).resolve(strict=True)
    for relative, expected in VISION_ENGINE_SOURCE_PINS.items():
        path = root / relative
        if path.resolve(strict=True) != path or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Vision engine source mismatch: {relative}")


def sandbox_prefix():
    """bubblewrap argv for the decode worker: every namespace unshared (no network: the host's loopback services and
    LAN are unreachable), a cleared environment, and only the interpreter, its venv and the worker directory mounted,
    read-only. None when bwrap is not installed."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return None
    # Private /proc, /dev and /tmp first: the read-only binds below then stay visible even when one lies under /tmp.
    argv = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--setenv", "LC_ALL", "C",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    mounts = []
    for path in sorted({os.path.realpath(p) for p in ("/usr", sys.base_prefix, sys.prefix, str(WORKER.parent),
                                                      os.path.dirname(os.path.realpath(sys.executable)))}, key=len):
        if not any(path == m or path.startswith(m + "/") for m in mounts):
            mounts.append(path)
            argv += ["--ro-bind", path, path]
    for path in ("/bin", "/sbin", "/lib", "/lib32", "/lib64"):  # merged-/usr hosts link these into /usr
        if os.path.islink(path):
            argv += ["--symlink", os.readlink(path), path]
    return argv + ["--chdir", "/"]


def load_tower(config, options):
    """Load the single-device vision component after the TP text model (GPU). The tower's resize budget is
    set to the per-image token cap, so the engine scales images to exactly what image_tokens() predicts."""
    import exllamav3
    verify_vision_sources(Path(exllamav3.__file__).resolve().parent.parent)
    pp = config.vision_pp
    if ((pp.patch_size, pp.merge_size, pp.temporal_patch_size, pp.min_image_tokens)
            != (PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE, MIN_IMAGE_TOKENS)
            or pp.max_image_tokens < options.max_image_tokens):
        raise ValueError("The checkpoint's image preprocessing differs from the vision token estimate")
    pp.max_image_tokens = options.max_image_tokens
    tower = exllamav3.Model.from_config(config, component="vision")
    tower.load(device=f"cuda:{options.device}")
    return tower


class Vision:
    def __init__(self, options, tower, tokenizer, secret=None):
        self.options = options
        self.tower = tower
        self.tokenizer = tokenizer
        self.secret = secrets.token_bytes(32) if secret is None else secret
        self.cache = EmbeddingCache(options.cache_bytes)
        self.decode_slots = asyncio.Semaphore(options.decode_workers)
        self.sandbox, self.isolation = self.probe_sandbox()
        self.counters = dict.fromkeys(("images_accepted", "image_rejects", "decode_failures", "tower_failures",
                                       "estimate_mismatches", "shared_prefix_skipped"), 0)

    def reject(self, message):
        self.counters["image_rejects"] += 1
        return ImageRejected(message)

    def probe_sandbox(self):
        """(argv prefix, isolation name). Decodes a 1x1 PNG through bubblewrap once, at start; --vision-decode-sandbox
        bwrap refuses to start when that fails, auto falls back to rlimits only (one log line)."""
        o = self.options
        if o.decode_sandbox == "none":
            return [], "rlimits"
        prefix, reason = sandbox_prefix(), "bwrap is not installed"
        if prefix is not None:
            try:
                for fmt, data in PROBE_IMAGES.items():
                    done = subprocess.run(prefix + [sys.executable, "-I", "-B", str(WORKER), fmt, "1", "1", "64",
                                                    str(o.decode_memory_bytes), "1024", "5"],
                                          input=data, capture_output=True, timeout=30, env={"LC_ALL": "C"}, cwd="/")
                    header = json.loads(done.stdout.split(b"\n", 1)[0] or b"{}")
                    header = header if isinstance(header, dict) else {}
                    if not (done.returncode == 0 and header.get("ok") is True and header.get("bytes") == 3
                            and (header.get("memory_limited") is True or not o.require_memory_limit)):
                        reason = f"{fmt} probe exit {done.returncode}: {header.get('error') or done.stderr.decode(errors='replace').strip()[-160:]}"
                        break
                else:
                    return prefix, "bwrap"
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                reason = f"probe failed: {error!r}"[:200]
        if o.decode_sandbox == "bwrap":
            raise ValueError(f"--vision-decode-sandbox bwrap: the decode sandbox does not work here ({reason})")
        print(f"Vision decode sandbox unavailable ({reason}); decoding with rlimits only", flush=True)
        return [], "rlimits"

    def parse(self, part):
        o = self.options
        source = part.get("image_url")
        try:
            data = decode_data_url(source.get("url") if isinstance(source, dict) else source, o.max_image_bytes)
            fmt = sniff(data)
            width, height = HEADER_SIZE[fmt](data)
        except ImageRejected as error:
            raise self.reject(str(error)) from None
        if not width or not height:
            raise self.reject("Image has zero width or height")
        if width * height > o.max_image_pixels:
            raise self.reject(f"Image of {width}x{height} exceeds {o.max_image_pixels} pixels")
        if max(width, height) > o.max_image_side:
            raise self.reject(f"Image of {width}x{height} exceeds {o.max_image_side} pixels on a side")
        digest = hashlib.sha256(b"glm-vision-1|%d|%d|" % (o.max_image_tokens, MIN_IMAGE_TOKENS))
        digest.update(data)
        key = digest.hexdigest()
        alias = "<$GLM_IMAGE_" + hmac.new(self.secret, key.encode(), "sha256").hexdigest()[:32] + "$>"
        return ImagePart(key, alias, data, fmt, width, height, image_tokens(width, height, o.max_image_tokens))

    def rewrite(self, messages):
        """Each image_url part of a user message becomes a text part holding its alias. A text-only request returns
        the same messages object and no parts. Other media parts stay untouched, so render_prompt still refuses them."""
        o = self.options
        out, parts, total = [], [], 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list) or not any(part.get("type") == "image_url" for part in content):
                out.append(message)
                continue
            if message.get("role") != "user":
                raise self.reject(f"Image content is supported only in user messages, not in {message.get('role')!r}")
            rewritten = []
            for part in content:
                if part.get("type") != "image_url":
                    rewritten.append(part)
                    continue
                if len(parts) == o.max_images:
                    raise self.reject(f"At most {o.max_images} images per request")
                image = self.parse(part)
                total += len(image.data)
                if total > o.max_request_image_bytes:
                    raise self.reject(f"Images in one request exceed {o.max_request_image_bytes} bytes")
                parts.append(image)
                rewritten.append({"type": "text", "text": image.alias})
            out.append({**message, "content": rewritten})
        if not parts:
            return messages, []
        self.counters["images_accepted"] += len(parts)
        return out, parts

    def count_tokens(self, prompt, parts, tokenizer):
        """Exact prompt length before the tower runs. The engine splits the prompt at each alias and tokenizes the text
        between them separately (Tokenizer.encode_special_or_unspecial), then splices in [begin] + ids + [end]."""
        expected = Counter(part.alias for part in parts)
        if any(prompt.count(alias) != count for alias, count in expected.items()):
            raise self.reject("The chat template did not place every image exactly once")
        by_alias = {part.alias: part for part in parts}
        total = 0
        for piece in re.split("(" + "|".join(map(re.escape, by_alias)) + ")", prompt):
            if piece in by_alias:
                total += by_alias[piece].tokens
            elif piece:
                total += tokenizer.encode(piece, encode_special_tokens=True).shape[-1]
        return total

    async def embed(self, parts, generator, disconnected=None):
        """After admission: one embedding per distinct image, in first-use order, for Tokenizer.encode and the Job.
        Images are decoded and encoded one at a time, so a request holds at most one decoded raster."""
        unique = list({part.key: part for part in parts}.values())
        found = {part.key: self.cache.get(part.key) for part in unique}
        for part in unique:
            if found[part.key] is not None:
                continue
            if disconnected is not None and await disconnected():
                raise ImageRejected("Client disconnected before its images were encoded")
            image = await self.decode(part)
            if disconnected is not None and await disconnected():
                raise ImageRejected("Client disconnected before its images were encoded")
            embedding = await self.encode_image(image, part, generator)
            del image
            if embedding.text_alias != part.alias:
                raise VisionFailure("Image encoder returned a different alias")
            if embedding.full_length != part.tokens:
                self.mismatch(part.tokens, embedding.full_length, f"{part.width}x{part.height}")
            self.cache.put(part.key, embedding, protected=found)
            found[part.key] = embedding
        return [found[part.key] for part in unique]

    def mismatch(self, estimate, actual, what="prompt"):
        self.counters["estimate_mismatches"] += 1
        print(f"Vision token estimate differs for {what}: estimated {estimate}, encoded {actual}", flush=True)

    async def encode_image(self, image, part, generator):
        loop = asyncio.get_running_loop()
        async with generator._step_lock:
            future = loop.run_in_executor(generator._gpu_worker, self.tower.get_image_embeddings,
                                          self.tokenizer, image, part.alias)
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # Never release the step lock under a running GPU call, however often this task is cancelled.
                while not future.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait({future})
                if not future.cancelled() and future.exception() is not None:
                    self.counters["tower_failures"] += 1
                raise
            except Exception as error:
                self.counters["tower_failures"] += 1
                print(f"Image encoding failed: {error!r}", flush=True)
                raise VisionFailure("Image encoding failed; see service log") from error

    async def decode(self, part):
        from PIL import Image
        o = self.options
        async with self.decode_slots:
            process = await asyncio.create_subprocess_exec(
                *self.sandbox, sys.executable, "-I", "-B", str(WORKER), part.format, str(part.width), str(part.height),
                str(o.max_image_pixels), str(o.decode_memory_bytes), str(o.max_image_bytes), str(o.decode_timeout + 1),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                cwd="/", env={"LC_ALL": "C"})
            try:
                pixels = await asyncio.wait_for(self.exchange(process, part), o.decode_timeout)
            except asyncio.TimeoutError:
                raise self.decode_failed(part, f"timed out after {o.decode_timeout} s") from None
            finally:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
        return Image.frombytes("RGB", (part.width, part.height), pixels)

    async def exchange(self, process, part):
        expected = part.width * part.height * 3
        try:
            process.stdin.write(part.data)
            await process.stdin.drain()
            process.stdin.close()
            header = json.loads(await process.stdout.readline())    # one line, bounded by the stream limit
        except (OSError, ValueError) as error:
            raise self.decode_failed(part, type(error).__name__) from None
        if not (isinstance(header, dict) and header.get("ok") is True and header.get("mode") == "RGB"
                and header.get("bytes") == expected):
            raise self.decode_failed(part, header.get("error") if isinstance(header, dict) else "no header")
        if self.options.require_memory_limit and header.get("memory_limited") is not True:
            raise self.decode_failed(part, "memory limit not applied")
        try:
            pixels = await process.stdout.readexactly(expected)
        except asyncio.IncompleteReadError:
            raise self.decode_failed(part, "short output") from None
        if await process.stdout.read(1) or await process.wait() != 0:
            raise self.decode_failed(part, "unexpected output or exit status")
        return pixels

    def decode_failed(self, part, reason):
        self.counters["decode_failures"] += 1
        print(f"Image decode failed ({part.format} {part.width}x{part.height}): {reason}", flush=True)
        return ImageRejected(f"Image could not be decoded ({part.format}, {part.width}x{part.height})")

    def health(self):
        return {"limits": self.options.limits(), "decode_isolation": self.isolation, **self.counters,
                "cache": self.cache.stats()}
