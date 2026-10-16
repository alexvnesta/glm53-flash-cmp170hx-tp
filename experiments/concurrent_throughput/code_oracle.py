"""Restricted, resource-bounded oracle for the synthetic compress_runs task.

Only function definitions are accepted, with no imports, classes, reflection,
global mutation, decorators or attribute access except list.append. The child
has a small builtin set and receives only public synthetic integer fixtures.
"""

import ast
import json
from pathlib import Path
import subprocess
import sys


def validate(answer):
    data = json.loads(answer)
    if not isinstance(data, dict) or set(data) != {"code"} or not isinstance(data["code"], str):
        return False
    source = data["code"]
    if len(source) > 8192:
        return False
    tree = ast.parse(source)
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        return False
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal,
                 ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Try)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            return False
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return False
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith("_")):
            return False
        if isinstance(node, ast.Attribute) and node.attr != "append":
            return False
    if sys.platform != "linux":
        raise RuntimeError("Code oracle requires Linux address-space limits; run on the model host")
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--child"],
        input=source, text=True, capture_output=True, timeout=3,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "PASS"


def child():
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    safe = {"len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "list": list, "tuple": tuple, "int": int, "bool": bool,
            "dict": dict, "sum": sum, "min": min, "max": max}
    namespace = {"__builtins__": safe}
    exec(compile(sys.stdin.read(), "synthetic-model-answer", "exec"), namespace)
    fn = namespace["compress_runs"]
    for values, expected in (
        ([], []), ([1], [[1, 1]]),
        ([3, 3, 1, 1, 1, 3], [[3, 2], [1, 3], [3, 1]]),
        ([-2, -2, 0, 1, 1], [[-2, 2], [0, 1], [1, 2]]),
        ([0, 0, 0, 0], [[0, 4]]), ([1, 2, 1], [[1, 1], [2, 1], [1, 1]]),
    ):
        before = list(values)
        if json.loads(json.dumps(fn(values))) != expected or values != before:
            raise SystemExit(2)
    print("PASS")


if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    child()
