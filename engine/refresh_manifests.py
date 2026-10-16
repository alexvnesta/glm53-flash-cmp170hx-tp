#!/usr/bin/env python3
"""Explicit maintainer operation: regenerate source manifests and startup pin tables.

Run only on a reviewed, reproduced source tree. Verification remains apply_engine.sh --verify.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import pprint

from check_engine_pins import TABLES, read_table, relative_path


def manifest(root):
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(x in (".git", "__pycache__", ".pytest_cache") for x in rel.parts) or not p.is_file():
            continue
        if p.is_symlink():
            raise ValueError("A source manifest must not follow symlinks: "+str(p))
        lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+rel.as_posix()+"\n")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weightless-engine", type=Path)
    parser.add_argument("--write", action="store_true", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    (repo/"engine/engine.sha256").write_text(manifest(args.engine))
    for filename, variable, required in TABLES:
        if filename == "glm_concurrent_ring.py":
            filename = "glm_native_concurrency.py"
        source = repo/"service/glm"/filename
        if not source.exists():
            if required:
                raise ValueError("Required pin table missing: "+filename)
            continue
        table = read_table(source, variable)
        values = {key: hashlib.sha256((args.engine/relative_path(key)).read_bytes()).hexdigest() for key in table}
        text = source.read_text()
        tree = ast.parse(text)
        assignment, = [n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets)]
        lines = text.splitlines(True)
        lines[assignment.lineno-1:assignment.end_lineno] = [variable+" = "+pprint.pformat(values, sort_dicts=False)+"\n"]
        source.write_text("".join(lines))
    if args.weightless_engine:
        (repo/"profiles/weightless/engine.sha256").write_text(manifest(args.weightless_engine))


if __name__ == "__main__":
    main()
