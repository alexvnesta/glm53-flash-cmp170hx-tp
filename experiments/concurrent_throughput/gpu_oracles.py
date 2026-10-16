"""Run only in an exclusive GPU window, before starting a model server.

Reuses the qualified upstream cooperative-MoE numerical oracle, enlarging its
test output/counter buffers beyond eight rows. The arithmetic reference and
tolerances are unchanged. Also checks GLM widths, clipping and expert masking.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import ast

from prepare_row16 import verify


def expand_oracle(source):
    node, = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == "run_case"]
    lines = source.splitlines(keepends=True)
    function = "".join(lines[node.lineno - 1:node.end_lineno])
    replacements = {
        'out = torch.full((8, H_out),': 'out = torch.full((max(8, bsz), H_out),',
        'smax * (I // 128) + 8 * (Ho // 128)':
            'smax * (I // 128) + max(8, bsz) * (Ho // 128)',
    }
    for old, new in replacements.items():
        if function.count(old) != 1:
            raise RuntimeError("Oracle function anchor changed")
        function = function.replace(old, new)
    return "".join(lines[:node.lineno - 1]) + function + "".join(lines[node.end_lineno:])


def oracle_code(source, filename):
    tree = ast.parse(expand_oracle(source), filename=filename)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Import) and
                         any(alias.name == "pytest" for alias in node.names))
                 and not (isinstance(node, ast.FunctionDef) and node.name != "run_case")]
    return compile(ast.fix_missing_locations(tree), filename, "exec")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--decode-max-rows', type=int, choices=(16,32,64), default=16)
    parser.add_argument("--exclusive-window", action="store_true", required=True)
    args = parser.parse_args()
    verify(args.engine, args.engine.parent / "engine.sha256")
    build = json.loads((args.native.parent / "build-receipt.json").read_text())
    binary, = args.native.glob("exllamav3_ext*.so")
    if hashlib.sha256(binary.read_bytes()).hexdigest() != build["native_sha256"]:
        raise RuntimeError("Candidate native SHA changed")
    owners = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
    ], text=True).strip()
    if owners:
        raise RuntimeError("GPU owner still present; refusing to compete with a model service")
    sys.path[:0] = [str(args.native.resolve()), str(args.engine.resolve()),
                   str((args.engine / "tests").resolve())]
    import torch
    import exllamav3_ext as ext

    if ext.decode_max_rows != args.decode_max_rows:
        raise RuntimeError("Native decode capacity differs from the requested oracle")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Expected exclusive access to two CMP GPUs")
    path = args.engine / "tests/kernels/moe/test_moe_coop.py"
    namespace = {"__name__": "row16_upstream_oracle", "__file__": str(path)}
    exec(oracle_code(path.read_text(), str(path)), namespace)
    run_case = namespace["run_case"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"passed": False, "source_oracle_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "reference": "Upstream reconstruct/dense oracle; unchanged 3e-3 relative-to-max tolerance",
               "cases": []}
    try:
        for index in (0, 1):
            device = torch.device(f"cuda:{index}")
            if torch.cuda.get_device_capability(device) != (8, 0):
                raise RuntimeError("Expected SM80")
            extra=(17,24,31,32) if args.decode_max_rows>=32 else ()
            extra+=((33,48,63,64) if args.decode_max_rows==64 else ())
            for rows in (1, 2, 4, 7, 8, 9, 16) + extra:
                run_case(device, 20261016 + rows, 3, 3, 3, 2,
                         bsz=rows, limit=10.0, gu_f32=True)
                receipt["cases"].append({"device": index, "rows": rows, "shape": [256, 256], "passed": True})
            for rows in (9, 16) + ((17,32) if args.decode_max_rows>=32 else ()) + ((33,48,63,64) if args.decode_max_rows==64 else ()):
                run_case(device, 20261016 + rows, 3, 3, 3, 2,
                         bsz=rows, limit=10.0, gu_f32=True,
                         H=4096, Hi=4096, I=2048, Ho=4096, H_out=4096,
                         E=8, topk=8, min_e=1, max_e=9, empty_rows=True)
                receipt["cases"].append({"device": index, "rows": rows, "shape": [4096, 2048],
                                         "expert_range_masking": True, "passed": True})
            if args.decode_max_rows>=32:
                for rows in (16,32)+((64,) if args.decode_max_rows==64 else ()):
                    run_case(device, 20261016+rows, 5, 5, 5, 2,
                             bsz=rows, limit=10.0, gu_f32=True)
                    receipt['cases'].append({'device':index,'rows':rows,'shape':[256,256],
                                             'bits':5,'passed':True})
        receipt["passed"] = True
    finally:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": receipt["passed"], "cases": len(receipt["cases"])}))


if __name__ == "__main__":
    main()
