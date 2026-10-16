"""Copy the service and prepare a private launch plan; never start or stop it."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare_row16 import verify

ENV_NAMES = {
    "PYTHONPATH", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "EXL3_TP_SHARING_STRATEGY",
    "EXL3_RECURRENT_DTYPE", "EXLLAMA_RECURRENT_STATE_DTYPE",
    "GLM_WEIGHTLESS_VECTOR", "GLM_WEIGHTLESS_SHA256", "GLM_WEIGHTLESS_BASE_REVISION",
    "GLM_WEIGHTLESS_ALPHA", "GLM_WEIGHTLESS_VALIDATE", "GLM_WEIGHTLESS_RECEIPTS",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "GLM53_NATIVE_EXT_SHA256", "PYTORCH_CUDA_ALLOC_CONF",
    "EXL3_BC_ATTN", "EXL3_MLA_KIDX_RING", "EXL3_INT8_GEMV", "EXL3_GEMV",
    "EXL3_DSA_QC_STAGE", "EXL3_DSA_QC_STAGE_MIN_R", "EXL3_TP_WORKER_TIMEOUT",
    "EXL3_AUTOSPLIT_MARGIN_MB", "GLM53_PREFILL_CHUNK_SIZE",
    "GLM53_DYNAMIC_DRAFT", "GLM53_RECORD_DRAFT_STATS", "GLM53_TP_ACTIVE_HOST",
    "NCCL_SOCKET_IFNAME",
}


def configuration_environment(raw_env, base, candidate):
    env = {}
    for entry in raw_env:
        if b"=" not in entry:
            continue
        key, value = entry.decode().split("=", 1)
        if key in ENV_NAMES:
            env[key] = value.replace(str(base), str(candidate))
        elif key.startswith(("EXL3_", "EXLLAMA_", "GLM53_", "NCCL_")):
            # An omitted engine flag can invalidate a comparison or prevent
            # startup. Fail without printing its potentially sensitive value.
            raise ValueError("Unreviewed model configuration variable: " + key)
    return env


def prepare(base, candidate, pid, port):
    verify(candidate / "engine", candidate / "engine.sha256")
    build = json.loads((candidate / "build-receipt.json").read_text())
    preparation = json.loads((candidate / "preparation.json").read_text())
    so, = (candidate / "native").glob("exllamav3_ext*.so")
    if hashlib.sha256(so.read_bytes()).hexdigest() != build["native_sha256"]:
        raise ValueError("Candidate native identity changed")
    argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if str(base / "service/glm/glm_api.py") not in argv:
        raise ValueError("PID does not own the qualified GLM service")
    if argv[argv.index("--port") + 1] == str(port):
        raise ValueError("Candidate must use a different loopback port")
    target = candidate / "service"
    if target.exists():
        raise ValueError("Preserve an existing candidate service")
    shutil.copytree(base / "service", target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    pins = {}
    for path in target.rglob("*.py"):
        text = path.read_text()
        updated = text
        for rel, change in preparation["changed_sources"].items():
            if change["before"] in updated:
                updated = updated.replace(change["before"], change["after"])
                pins.setdefault(str(path.relative_to(target)), []).append(rel)
        if updated != text:
            path.write_text(updated)
    raw_env = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
    # Read only explicit non-credential configuration; never serialize a full
    # service environment or inherit API/Hugging Face credentials into receipts.
    env = configuration_environment(raw_env, base, candidate)
    env["GLM53_NATIVE_EXT_SHA256"] = build["native_sha256"]
    env["GLM_WEIGHTLESS_RECEIPTS"] = str(candidate / "receipts/numerics")
    if not env.get("PYTHONPATH", "").startswith(str(candidate / "native") + ":"):
        raise ValueError("Candidate import path does not lead with its native binary")
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if key not in env:
            continue
        old_path = Path(env[key].replace(str(candidate), str(base)))
        new_path = Path(env[key])
        if new_path == old_path or candidate not in new_path.parents:
            raise ValueError("Candidate compilation cache must be isolated")
        if old_path.exists():
            shutil.copytree(old_path, new_path)
    argv = [value.replace(str(base / "service"), str(candidate / "service")) for value in argv]
    argv[argv.index("--port") + 1] = str(port)
    if "--disk_cache_dir" in argv:
        argv[argv.index("--disk_cache_dir") + 1] = str(candidate / "disk-cache")
    plan = {
        "status": "prepared_not_executed", "production_pid": pid,
        "argv": argv, "env": env, "cwd": str(candidate),
        "changed_service_pins": pins, "native_sha256": build["native_sha256"],
        "requires_exclusive_gpu_window": True,
        "qualification": "Numerics, GPU memory fit, functional, concurrency, cancellation and rollback remain untested",
    }
    output = candidate / "launch-plan.json"
    output.write_text(json.dumps(plan, indent=2) + "\n")
    output.chmod(0o600)
    print(json.dumps({"prepared": True, "port": port, "changed_service_pins": pins,
                      "native_sha256": build["native_sha256"], "launched": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--production-pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=8013)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.production_pid <= 1:
        parser.error("Require a user port and a production PID greater than one")
    prepare(args.base_profile.resolve(), args.candidate.resolve(), args.production_pid, args.port)


if __name__ == "__main__":
    main()
