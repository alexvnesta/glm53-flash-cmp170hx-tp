"""Decode one PNG/JPEG/WebP image in a throwaway process (stdlib + Pillow only).

glm_vision.py runs this file as `python -I -B glm_vision_worker.py FORMAT WIDTH HEIGHT MAX_PIXELS MEMORY_BYTES
MAX_INPUT_BYTES CPU_SECONDS` for each new image whose format, size and header dimensions it has already checked. The
encoded bytes arrive on stdin. Address space, CPU time, core dumps and file writes are limited before Pillow is
imported. The only output is one JSON line, then WIDTH*HEIGHT*3 RGB bytes on success. A decoder crash, bomb or hang
ends this process, never the serving process.
"""
import json
import resource
import sys

FORMATS = ("PNG", "JPEG", "WEBP")


def limit(kind, value):
    try:
        resource.setrlimit(kind, (value, value))
        return True
    except (ValueError, OSError):
        return False


def to_rgb(image):
    """The engine's convert_to_rgb (exllamav3/architecture/mm_processing/common.py): transparent pixels become white."""
    if image.mode == "RGB":
        return image
    from PIL import Image
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, "WHITE")
    canvas.paste(image, (0, 0), image)
    return canvas.convert("RGB")


def decode(data, fmt, width, height, max_pixels):
    import warnings
    from io import BytesIO
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = max_pixels
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    with Image.open(BytesIO(data), formats=[fmt]) as image:
        # A multi-picture JPEG (MPO: phone photos with a gain map or depth image) opens at its primary image, the one
        # whose frame header the parent read; the other pictures are never decoded.
        if image.format not in ((fmt, "MPO") if fmt == "JPEG" else (fmt,)) or image.size != (width, height):
            raise ValueError("decoder disagrees with the checked header")
        if image.format != "MPO" and (getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1):
            raise ValueError("animated image")
        image.load()
        return to_rgb(image).tobytes()


def main(argv):
    fmt = argv[1]
    width, height, max_pixels, memory, max_input, cpu = map(int, argv[2:8])
    memory_limited = limit(resource.RLIMIT_AS, memory)
    limit(resource.RLIMIT_CPU, cpu)
    limit(resource.RLIMIT_CORE, 0)
    limit(resource.RLIMIT_FSIZE, 0)
    data = sys.stdin.buffer.read(max_input + 1)
    out = sys.stdout.buffer
    try:
        if fmt not in FORMATS or len(data) > max_input or width * height > max_pixels:
            raise ValueError("input outside the checked limits")
        pixels = decode(data, fmt, width, height, max_pixels)
    except MemoryError:
        out.write(json.dumps({"ok": False, "error": "decode memory limit"}).encode() + b"\n")
        return 3
    except Exception as error:
        out.write(json.dumps({"ok": False, "error": type(error).__name__}).encode() + b"\n")
        return 2
    out.write(json.dumps({"ok": True, "mode": "RGB", "width": width, "height": height, "bytes": len(pixels),
                          "memory_limited": memory_limited}).encode() + b"\n")
    out.write(pixels)
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
