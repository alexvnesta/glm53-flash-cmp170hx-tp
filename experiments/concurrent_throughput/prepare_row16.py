"""Prepare an isolated 16-row fused-decode variant of the qualified engine.

The original engine, normal manifests and serving configuration are unchanged.
The variant requires a separately rebuilt extension exporting decode_max_rows.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(engine, manifest):
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual = {}
    for path in engine.rglob("*"):
        rel = path.relative_to(engine)
        if any(part in (".git", "__pycache__", ".pytest_cache") for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Source symlink refused: {rel}")
        if path.is_file():
            actual[rel.as_posix()] = sha(path)
    if actual != expected:
        mismatches = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise ValueError(f"Qualified source identity mismatch: {mismatches[:12]}")
    return expected


def prepare(engine, output, manifest):
    engine = engine.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output == engine or engine in output.parents:
        raise ValueError("Destination must be new and outside the qualified engine")
    expected = verify(engine, manifest)
    source = "exllamav3/"
    replacements = {
        source + "modules/mlp.py": ("MAX_BSZN = 8  #", "MAX_BSZN = ext.decode_max_rows  #"),
        source + "modules/block_sparse_mlp.py": ("MAX_BSZN = 8  #", "MAX_BSZN = ext.decode_max_rows  #"),
        source + "exllamav3_ext/libtorch/mlp.h": ("#define MAX_BSZN 8 ", "#define MAX_BSZN 16 "),
        source + "exllamav3_ext/bindings.cpp": (
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n{",
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n{\n    m.attr(\"decode_max_rows\") = MAX_BSZN;",
        ),
    }
    # Validate every anchor before creating the destination.
    for rel, (old, _) in replacements.items():
        if (engine / rel).read_text().count(old) != 1:
            raise ValueError(f"Expected one source anchor in {rel}")
    output.mkdir(parents=True)
    target = output / "engine"
    shutil.copytree(engine, target, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    changed = {}
    for rel, (old, new) in replacements.items():
        path = target / rel
        path.write_text(path.read_text().replace(old, new))
        changed[rel] = {"before": expected[rel], "after": sha(path)}
    hashes = {rel: sha(target / rel) for rel in expected}
    (output / "engine.sha256").write_text("".join(
        f"{digest}  {rel}\n" for rel, digest in sorted(hashes.items())
    ))
    receipt = {
        "status": "unqualified_candidate", "source_files": len(hashes),
        "base_manifest_sha256": sha(manifest), "changed_sources": changed,
        "decode_max_rows": 16, "native_rebuild_required": True,
        "gpu_tests_run": False, "serving_changed": False,
        "hypothesis": "Keep two K7 requests (16 verification rows) on the fused small-row path",
    }
    (output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.engine, args.output, args.manifest)))


if __name__ == "__main__":
    main()
