"""Filesystem locations shared by the GLM serving modules.

The repo layout is::

    <repo root>/
        glm/        serving code (this package)
        dflash2/    DFlash2 BF16 -> EXL3 6bpw conversion scripts
        scripts/    setup / serve launchers

P16 removed kernels/ (k_hcfuse: inert under tensor parallelism, slower when forced on; P15d) and its
GLM53_K_HCFUSE / GLM53_K_HCFUSE_CHECK switches.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DFLASH2_DIR = REPO_ROOT / "dflash2"
