"""One fused FP32 rank-one projection launch per selected target layer.

No full-sized projection temporary or host synchronization in the steady path.
Requires separate numerical/performance qualification on CMP170HX.
"""
import triton
import triton.language as tl


@triton.jit
def _project(X, D, alpha: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    value = tl.load(X + row * WIDTH + col)
    direction = tl.load(D + col)
    coefficient = tl.sum(value * direction, axis=0)
    tl.store(X + row * WIDTH + col, value - alpha * coefficient * direction)


def project_in_place(x, direction, alpha):
    width = 16384
    rows = x.numel() // width
    if rows:
        _project[(rows,)](x, direction, alpha, WIDTH=width, num_warps=16)
