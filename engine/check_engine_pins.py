#!/usr/bin/env python3
"""Check the engine files that the GLM service hash-pins at startup against an engine tree.

The service refuses to start vision or the concurrent ring when a pinned engine file differs, so a rebuilt or
re-patched engine must pass this check before it is used. The pin tables are read (never imported or executed)
from the service sources with ast.

usage: check_engine_pins.py ENGINE_SRC_DIR SERVICE_GLM_DIR

Exit status: 0 when every REQUIRED table matches, 1 when a required table has a mismatch, 2 on a usage or parse error.
"""
import ast
import hashlib
import sys
from pathlib import Path

# (source file, variable, required). Keys are either a relative path (exllamav3/x/y.py) or a dotted module name.
TABLES = (
    ("glm_vision.py", "VISION_ENGINE_SOURCE_PINS", True),
    ("glm_concurrent_ring.py", "ENGINE_SOURCE_PINS", True),
    ("glm_tp_prefix.py", "ENGINE_SOURCE_PINS", False),
    ("glm_dflash_sessions.py", "ENGINE_SOURCE_PINS", False),
)


def read_table(path: Path, variable: str) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                raise ValueError(f"{path.name}: {variable} is not a str->str dict")
            return value
    raise ValueError(f"{path.name}: {variable} not found")


def relative_path(key: str) -> str:
    # A path (contains "/") is used as is; a dotted module name (exllamav3.cache.recurrent) maps to its .py file.
    return key if "/" in key else key.replace(".", "/") + ".py"


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    engine, glm = Path(argv[1]), Path(argv[2])
    failed_required = False
    for filename, variable, required in TABLES:
        source = glm / filename
        if not source.is_file():
            print(f"SKIP  {filename}: not found")
            failed_required |= required
            continue
        try:
            table = read_table(source, variable)
        except (ValueError, SyntaxError) as error:
            print(f"ERROR {error}")
            return 2
        bad = []
        for key, expected in table.items():
            target = engine / relative_path(key)
            actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
            if actual != expected:
                bad.append((key, expected, actual))
        label = "required" if required else "informational"
        if bad:
            failed_required |= required
            if required:
                print(f"FAIL  {filename}:{variable} ({label}): {len(table) - len(bad)}/{len(table)} match")
            else:
                print(f"NOTE  {filename}:{variable} ({label}): {len(table) - len(bad)}/{len(table)} match; this table belongs to a "
                      f"code path the tensor-parallel profile does not use, and it does not stop the server")
                continue
            for key, expected, actual in bad:
                print(f"        {key}\n          pinned {expected}\n          actual {actual}")
        else:
            print(f"OK    {filename}:{variable} ({label}): {len(table)}/{len(table)} match")
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
