"""Profile identity and explicit options for upstream's persistent cache tier.

Importing this module never imports Torch or initializes a GPU.
"""
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class PersistenceOptions:
    directory: str | None = None
    budget_bytes: int = 0
    min_free_bytes: int = 0

    @property
    def enabled(self):
        return self.directory is not None


def persistence_options(args, retention):
    directory = getattr(args, "disk_cache_dir", None)
    if not directory:
        return PersistenceOptions()
    size = getattr(args, "disk_cache_size", 50)
    reserve = getattr(args, "persistent_cache_min_free_gib", 16)
    if (not retention.enabled or not retention.target_cpu_bytes
            or isinstance(size, bool) or not isinstance(size, (int, float))
            or not math.isfinite(size) or not 1 <= size <= 256
            or type(reserve) is not int or not 1 <= reserve <= 256):
        raise ValueError("Persistent cache requires the B2 RAM tier, a 1..256 GiB disk budget and 1..256 GiB free-space reserve")
    p = Path(directory).expanduser()
    if not p.is_absolute():
        raise ValueError("Persistent cache directory must be absolute")
    return PersistenceOptions(str(p), int(size*1024**3), reserve*1024**3)


def source_digest(directory):
    """Hash executable sources, independent of checkout path and pycache/JIT artifacts."""
    root = Path(directory).resolve(strict=True)
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".py", ".cpp", ".cu", ".h", ".cuh") or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError("Profile sources must not be symlinks")
        h.update(path.relative_to(root).as_posix().encode()+b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def profile_identity(engine_root, native_sha, weightless, recurrent_dtype, environ=os.environ):
    if len(native_sha) != 64 or any(c not in "0123456789abcdef" for c in native_sha):
        raise ValueError("A pinned native SHA256 is required for profile identity")
    if recurrent_dtype not in ("fp16", "fp32"):
        raise ValueError("Unsupported recurrent-state precision")
    # Weightless readiness already verified these fields against current-process receipts.
    steering = {k: weightless.get(k) for k in
                ("enabled", "vector_sha256", "base_revision", "alpha", "layers", "width")}
    return {"schema": 1, "engine_sha256": source_digest(engine_root),
            "service_sha256": source_digest(Path(__file__).parent), "native_sha256": native_sha,
            "torch": version("torch"), "triton": version("triton"),
            "recurrent_dtype": recurrent_dtype, "weightless": steering,
            "kernel_flags": {k: v for k, v in sorted(environ.items()) if k.startswith("EXL3_")}}


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def attach_persistence(coordinator, options, identity):
    """Attach once, at startup, to the exact upstream tier owned by this B2 coordinator."""
    if not options.enabled:
        return None
    from exllamav3.generator.disk_cache import DiskPageCache
    g, tier = coordinator.generator, coordinator.tier
    coordinator.check_owner()
    if (tier is None or not getattr(tier, "persistent_capable", False)
            or g.disk_page_cache is not None or coordinator.native.disk_tier is not None):
        raise ValueError("Persistence requires the freshly attached B2 upstream CPU tier")
    root = Path(options.directory)
    if root.is_symlink():
        raise ValueError("Persistent cache root must not be a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.geteuid():
        raise ValueError("Persistent cache root must belong to the service account")
    root.chmod(0o700)
    class OwnedDiskPageCache(DiskPageCache):
        def prefetch(self, hashes, serial):
            # Native enqueue runs on the HTTP thread. Allocation-time promotion runs on
            # the serialized GPU executor; keep all rank I/O on that existing owner.
            return None

    disk = OwnedDiskPageCache(str(root), [g.cache], tier, options.budget_bytes,
                         min_free=options.min_free_bytes, extra_identity=identity)
    try:
        disk.recurrent_sides = {"target": (g.model, g.cache), "draft": (g.draft_model, g.draft_cache)}
        tier.attach_disk(disk)
        g.disk_page_cache = coordinator.native.disk_tier = coordinator.disk = disk
        g.disk_cache_flush_on_idle = True
        (Path(disk.dir)/"serving-profile.json").write_text(json.dumps(identity, sort_keys=True, indent=2)+"\n")
        coordinator.check_owner()
        return disk
    except BaseException:
        disk.abort()
        disk.close()
        if tier.disk_tier is disk:
            tier.disk_tier = None
        if g.disk_page_cache is disk:
            g.disk_page_cache = None
        if coordinator.native.disk_tier is disk:
            coordinator.native.disk_tier = None
        coordinator.disk = None
        raise


def persistence_health(disk):
    if disk is None:
        return {"enabled": False}
    with disk._lock:
        return {"enabled": True, "fingerprint": disk.fingerprint, "budget_bytes": disk.root_size,
                "resident_bytes": disk.total_size, "pages": len(disk.pages), "checkpoints": len(disk.states),
                "pending_pages": len(disk.pending_pages), "pending_checkpoints": len(disk.pending_states),
                "writes_disabled": disk.write_disabled, "metrics": dict(disk.metrics)}
