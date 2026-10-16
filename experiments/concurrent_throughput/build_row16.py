"""CPU-only incremental rebuild; no model loading or service operations.

Reuses attested native objects and rebuilds every C++ translation unit that
includes the changed class header. Both engines must share all other sources.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from prepare_row16 import verify

NATIVE_SUBDIR = Path("exllamav3/exllamav3_ext")
REBUILD = ("bindings", "libtorch/mlp", "libtorch/blocksparse_mlp")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_command(cmd, rel, engine, build_dir):
    sources = [arg for arg in cmd if arg.endswith(str(rel))]
    if len(sources) != 1 or not Path(sources[0]).is_absolute() or "-c" not in cmd:
        raise ValueError("Unexpected compiler command")
    old_engine = sources[0][:-len(str(rel))]
    object_rel = rel.with_suffix(".o")
    old_object = cmd[cmd.index("-o") + 1]
    if not old_object.endswith(str(object_rel)):
        raise ValueError("Unexpected object path")
    old_build = old_object[:-len(str(object_rel))]
    # Includes, output and dependency files must all use the isolated copy.
    return [arg.replace(old_engine, str(engine) + "/")
            .replace(old_build, str(build_dir) + "/") for arg in cmd]


def build(base, candidate, python, expected_native):
    so, = (base / "native").glob("exllamav3_ext*.so")
    if sha(so) != expected_native:
        raise ValueError("Qualified native SHA mismatch")
    verify(candidate / "engine", candidate / "engine.sha256")
    receipt = json.loads((candidate / "preparation.json").read_text())
    rows = receipt['decode_max_rows']
    if type(rows) is not int or rows not in (16, 32):
        raise ValueError('Unsupported native decode capacity')
    changes = receipt["changed_sources"]
    base_files = {p.relative_to(base / "engine").as_posix(): sha(p)
                  for p in (base / "engine").rglob("*")
                  if p.is_file() and not any(x in (".git", "__pycache__", ".pytest_cache")
                                            for x in p.relative_to(base / "engine").parts)}
    for rel, digest in base_files.items():
        expected = changes.get(rel, {}).get("after", digest)
        if sha(candidate / "engine" / rel) != expected:
            raise ValueError("Unreviewed candidate source change: " + rel)
    if any(base_files.get(rel) != values["before"] for rel, values in changes.items()):
        raise ValueError("Candidate was prepared against a different base")
    base_build = base / "native/.build"
    base_objects = sorted(base_build.glob("exllamav3/**/*.o"))
    custody = json.loads((base / "build-receipt.json").read_text())["object_hashes"]
    if {str(p.relative_to(base_build)): sha(p) for p in base_objects} != custody:
        raise ValueError("Qualified object custody mismatch")
    ninja = python.parent / "ninja"
    commands = subprocess.check_output([
        str(ninja), "-f", str(base_build / "build.ninja"), "-t", "commands"
    ], text=True).splitlines()
    native = candidate / "native"
    native.mkdir(exist_ok=True)
    if list(native.glob("exllamav3_ext*.so")):
        raise ValueError("Preserve the already completed candidate binary")
    build_dir = native / ".build"
    if not build_dir.exists():
        shutil.copytree(base_build, build_dir)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TORCH_CUDA_ARCH_LIST": "8.0", "MAX_JOBS": "1"}
    started = time.monotonic()
    used = []
    for name in REBUILD:
        rel = NATIVE_SUBDIR / (name + ".cpp")
        matches = [shlex.split(c) for c in commands if str(rel) in c]
        if len(matches) != 1:
            raise ValueError(f"Expected one compile command: {name}")
        cmd = compile_command(matches[0], rel, candidate / "engine", build_dir)
        print("compile " + name, flush=True)
        subprocess.run(["nice", "-n", "19", "ionice", "-c3", *cmd],
                       cwd=candidate / "engine", env=env, check=True)
        used.append(cmd)
    objects = sorted(build_dir.glob("exllamav3/**/*.o"))
    rebuilt = {str(NATIVE_SUBDIR / (name + ".o")) for name in REBUILD}
    for obj in objects:
        rel = str(obj.relative_to(build_dir))
        if rel not in rebuilt and sha(obj) != custody[rel]:
            raise ValueError("Reused object changed: " + rel)
    suffix = subprocess.check_output([str(python), "-c",
        'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'], text=True).strip()
    output = native / ("exllamav3_ext" + suffix)
    lib = subprocess.check_output([str(python), "-c",
        'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")'],
        text=True, env=env).strip()
    cmd = ["c++", "-shared", *(str(p) for p in objects), "-L" + lib,
           "-L/usr/local/cuda-13.3/lib64", "-lc10", "-ltorch", "-ltorch_cpu",
           "-ltorch_python", "-lcudart", "-lc10_cuda", "-ltorch_cuda", "-o", str(output)]
    subprocess.run(["nice", "-n", "19", "ionice", "-c3", *cmd], env=env, check=True)
    used.append(cmd)
    check_env = {**env, "PYTHONPATH": str(native)}
    subprocess.run([str(python), "-c", "import torch; import exllamav3_ext as e; "
                    f"assert e.decode_max_rows == {rows}; assert not torch.cuda.is_initialized(); "
                    f"print('native import OK; decode_max_rows={rows}; CUDA not initialized')"],
                   env=check_env, check=True)
    result = {
        "compiled": True, "gpu_visible": False, "gpu_tests_run": False, "decode_max_rows": rows,
        "base_native_sha256": expected_native, "native_sha256": sha(output),
        "rebuilt_objects": sorted(rebuilt), "reused_objects": len(objects) - len(rebuilt),
        "commands": used, "seconds": time.monotonic() - started,
        "object_hashes": {str(p.relative_to(build_dir)): sha(p) for p in objects},
    }
    (candidate / "build-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("commands", "object_hashes")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--base-native-sha256", required=True)
    args = parser.parse_args()
    build(args.base_profile.resolve(), args.candidate.resolve(), args.python.absolute(),
          args.base_native_sha256)


if __name__ == "__main__":
    main()
