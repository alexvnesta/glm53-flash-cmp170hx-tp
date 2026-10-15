#!/usr/bin/env python3
"""Copy a qualified source tree into a separate experimental tree, then hook it.

Never edits the input tree or production pin tables. Not a deploy command.
"""
import argparse
import hashlib
from pathlib import Path
import shutil

TRANSFORMER_SHA256 = "6719c3b65952bb1fca2341c1671253d4048d9b497fe9c9d3c26574429f7a7d4a"
HERE = Path(__file__).resolve().parent
IMPORT = "from ..util import profile_opt\n"
RETURN = "        return to2(x, out_dtype, self.out_dtype)\n"
INITIALIZE = """
# Optional GLP experiment; each spawned TP worker initializes its own reader.
# Disabled leaves the model tensor operations and captures unchanged.
import os
_weightless_projector = None
if os.environ.get("GLM_WEIGHTLESS_VECTOR"):
    from ..util.weightless import make_projector
    _weightless_projector = make_projector()
"""
HOOK = """        if _weightless_projector is not None:
            x = _weightless_projector(self, x, params)
"""


def prepare(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("destination must be a new tree outside the source")
    file = source / "exllamav3/modules/transformer.py"
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != TRANSFORMER_SHA256:
        raise ValueError("engine transformer differs from the qualified source")
    text = data.decode()
    if text.count(IMPORT) != 1 or text.count(RETURN) != 2:
        raise ValueError("unexpected TransformerBlock hook sites")
    patched = text.replace(IMPORT, IMPORT + INITIALIZE).replace(RETURN, HOOK + RETURN, 1)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    (destination / "exllamav3/modules/transformer.py").write_text(patched)
    util = destination / "exllamav3/util"
    shutil.copy2(HERE / "glm_weightless.py", util / "weightless.py")
    shutil.copy2(HERE / "weightless_kernel.py", util / "weightless_kernel.py")
    return hashlib.sha256(patched.encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print("experimental transformer SHA256:", prepare(args.source, args.destination))
