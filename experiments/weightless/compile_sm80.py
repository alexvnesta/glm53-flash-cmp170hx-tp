#!/usr/bin/env python3
"""Compile the projector for SM80 without exposing GPUs or launching a kernel.

Run with an existing Linux Triton installation; outputs are local build artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRITON_CACHE_DIR"] = str(root / "cache")
    import triton
    from triton.compiler import ASTSource, compile
    from triton.backends.compiler import GPUTarget
    from weightless_kernel import _project
    source = ASTSource(fn=_project, signature={"X": "*fp32", "D": "*fp32"},
                       constexprs={"alpha": 1.0, "WIDTH": 16384})
    kernel = compile(source, target=GPUTarget("cuda", 80, 32), options={"num_warps": 16})
    (root / "project.ptx").write_text(kernel.asm["ptx"])
    (root / "project.cubin").write_bytes(kernel.asm["cubin"])
    receipt = {"target": "sm80", "triton_version": triton.__version__,
               "gpu_visible": False, "kernel_launched": False,
               "num_warps": kernel.metadata.num_warps, "shared_bytes": kernel.metadata.shared,
               "ptx_bytes": len(kernel.asm["ptx"]), "cubin_bytes": len(kernel.asm["cubin"]),
               "kernel_source_sha256": hashlib.sha256(
                   Path(__file__).with_name("weightless_kernel.py").read_bytes()).hexdigest()}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
