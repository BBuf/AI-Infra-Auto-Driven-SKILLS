"""Microbench skeleton for an AKO4ALL kernel iteration.

Copy this file into `AKO4ALL/<your-task>/bench/bench_<kernel>.py` and adapt:

1. Replace `kernel_baseline` and `kernel_candidate` with imports from
   `input/<kernel>.py` and `solution/<kernel>.py`.
2. Replace `reference_impl` with the trusted reference (PyTorch eager,
   previous operator, etc.).
3. Fill `SHAPES` with one or more representative production shapes plus tail
   shapes; never tune on a single shape only.
4. Set `DEVICE`, `DTYPE`, `WARMUP`, `ITERS` to match production.

The script prints a single table at the end so that both `bench_kernel.py
--baseline` and `--candidate` outputs can be diffed by eye and pasted into
`ITERATIONS.md`.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch


@dataclass
class ShapeSpec:
    tag: str  # "hot", "tail", ...
    args: tuple  # whatever your kernel takes


# TODO: fill with production-representative shapes.
SHAPES: list[ShapeSpec] = [
    # ShapeSpec("hot",  (1, 32, 4096, 4096, 128)),
    # ShapeSpec("tail", (1, 32,   16, 4096, 128)),
]

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = 20
ITERS = 100
ATOL = 1e-2
RTOL = 1e-2


# TODO: replace with real implementations.
def reference_impl(*args):
    raise NotImplementedError("plug in trusted reference")


def kernel_baseline(*args):
    raise NotImplementedError("plug in baseline (input/<kernel>.py)")


def kernel_candidate(*args):
    raise NotImplementedError("plug in candidate (solution/<kernel>.py)")


def make_inputs(spec: ShapeSpec):
    """Return a tuple of tensors matching kernel signature.

    Override per kernel. Use deterministic seeds so the bench is reproducible.
    """
    torch.manual_seed(0)
    raise NotImplementedError("build inputs from spec.args")


def correctness(spec: ShapeSpec, fn) -> tuple[float, float]:
    """Return (max_abs_err, max_rel_err) of ``fn`` vs ``reference_impl``."""
    inputs = make_inputs(spec)
    ref = reference_impl(*inputs)
    out = fn(*inputs)
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs().clamp_min(1e-9))
    return diff.max().item(), rel.max().item()


def time_one(spec: ShapeSpec, fn) -> dict:
    """Median + p95 over ITERS, after WARMUP. Uses CUDA events for accuracy."""
    inputs = make_inputs(spec)

    for _ in range(WARMUP):
        fn(*inputs)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]

    for i in range(ITERS):
        starts[i].record()
        fn(*inputs)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "median_us": statistics.median(times_ms) * 1e3,
        "p95_us": times_ms[int(0.95 * ITERS) - 1] * 1e3,
        "min_us": times_ms[0] * 1e3,
    }


def run(target: str) -> None:
    fn = {"baseline": kernel_baseline, "candidate": kernel_candidate}[target]

    rows: list[tuple] = []
    for spec in SHAPES:
        max_abs, max_rel = correctness(spec, fn)
        if max_abs > ATOL or max_rel > RTOL:
            print(f"[FAIL] {spec.tag}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
            return
        timing = time_one(spec, fn)
        rows.append((spec.tag, max_abs, max_rel, timing))

    fmt = "{:<8} {:>12} {:>12} {:>12} {:>10} {:>10}"
    print(fmt.format("shape", "median_us", "p95_us", "min_us", "max_abs", "max_rel"))
    for tag, max_abs, max_rel, t in rows:
        print(
            fmt.format(
                tag,
                f"{t['median_us']:.2f}",
                f"{t['p95_us']:.2f}",
                f"{t['min_us']:.2f}",
                f"{max_abs:.2e}",
                f"{max_rel:.2e}",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", choices=["baseline", "candidate"], default="candidate"
    )
    args = parser.parse_args()
    run(args.target)
