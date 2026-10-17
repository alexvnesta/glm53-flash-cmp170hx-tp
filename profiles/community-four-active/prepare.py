"""Reproduce the measured four-slot source profile without importing CUDA.

First run engine/apply_engine.sh. Then: python prepare.py OUTPUT_DIRECTORY
This copies the pinned base engine and service, applies the exact measured
source overlays, and checks the full engine manifest and service startup pins.
Native compilation is a separate, explicit step.
"""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    profile = Path(__file__).resolve().parent
    repo = profile.parents[1]
    output = Path(sys.argv[1]).expanduser().resolve()
    if output.exists():
        raise ValueError("Output directory already exists; use a fresh directory")
    subprocess.run([str(repo / "engine/apply_engine.sh"), "--verify"], check=True)
    output.mkdir(parents=True)
    for source, name in ((repo / "engine-src", "engine"), (repo / "service", "service")):
        shutil.copytree(source, output / name, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        patch = profile / f"{name}.patch"
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=output / name, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=output / name, check=True)
    expected = {}
    for line in (profile / "engine.sha256").read_text().splitlines():
        digest, relative = line.split(None, 1)
        expected[relative.strip()] = digest
    actual = {p.relative_to(output / "engine").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (output / "engine").rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Prepared engine differs from the measured manifest")
    subprocess.run([sys.executable, str(repo / "engine/check_engine_pins.py"), str(output / "engine"),
                    str(output / "service/glm")], check=True)
    for folder in ("native", "receipts"):
        (output / folder).mkdir()
    print(f"Verified {len(actual)} engine files and service startup pins: {output}")


if __name__ == "__main__":
    main()
