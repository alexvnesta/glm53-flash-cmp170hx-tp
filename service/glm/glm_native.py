"""Native extension custody before the engine is imported (P16).

Engine patch 0004(e) made exllamav3/ext.py refuse a JIT build and refuse an extension without its ABI marker, so a
start could never mix Python sources with another build's native code. Upstream ext.py has neither: without a
precompiled exllamav3_ext on sys.path it compiles one at import time (about 15 minutes, unpinned). This module replaces
the marker with a check the service runs before its first engine import:

- exllamav3_ext must resolve (importlib find_spec, which does not import exllamav3) to a precompiled extension file;
- with GLM53_NATIVE_EXT_SHA256 set, that file's sha256 must equal it (the unit pins the qualified build this way).

Importing this module does not load Torch or the engine.
"""
import hashlib
import importlib.machinery
import importlib.util
import os

ENV = "GLM53_NATIVE_EXT_SHA256"
NAME = "exllamav3_ext"


def require_prebuilt_extension(environ=None, find_spec=importlib.util.find_spec):
    environ = os.environ if environ is None else environ
    spec = find_spec(NAME)
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or not origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        raise RuntimeError(f"{NAME} is not a precompiled extension on sys.path (the engine would JIT-build one at "
                           "import); put the qualified native build first on PYTHONPATH")
    digest = hashlib.sha256()
    with open(origin, "rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    actual = digest.hexdigest()
    expected = environ.get(ENV)
    if expected is not None and actual != expected.strip().lower():
        raise RuntimeError(f"{NAME} at {origin} has sha256 {actual}, not the pinned {ENV}")
    return {"path": origin, "sha256": actual, "pinned": expected is not None}
