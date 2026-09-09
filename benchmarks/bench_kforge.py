"""kforge-style kernel-optimization experiment: compare PyTorch eager mode
against torch.compile's real autotuned kernels for the same matmul shapes
used elsewhere in this repo. This is a legitimate (if much smaller) analog
of Gimlet's kforge - real automated kernel generation from PyTorch, not a
hand-written custom kernel, and the numbers below are measured, not
claimed.

Note: torch.compile's Inductor backend does not support MPS as a compile
target on this torch version - this runs on CPU. A CUDA run (e.g. on the
same rented pod as bench_cuda.py) would exercise actual Triton-generated
kernels instead of the CPU C++ backend; noted as follow-up in the README.
"""
import json
import time
from pathlib import Path

import torch

SHAPES = [
    ("tiny", 1, 256, 256),
    ("small", 8, 512, 512),
    ("medium", 32, 1024, 1024),
    ("large", 64, 2048, 2048),
    ("xlarge", 128, 4096, 4096),
]
WARMUP = 3
ITERS = 10


def matmul(x, w):
    return (x @ w).sum()


def bench(fn, x, w) -> float:
    for _ in range(WARMUP):
        fn(x, w).item()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn(x, w).item()
    return (time.perf_counter() - start) / ITERS


def main():
    compiled = torch.compile(matmul, mode="max-autotune")
    results = {}
    for name, batch, din, dout in SHAPES:
        x = torch.randn(batch, din, device="cpu")
        w = torch.randn(din, dout, device="cpu")
        eager_t = bench(matmul, x, w)
        compiled_t = bench(compiled, x, w)  # first call here also pays compile cost, isolated below
        results[name] = {"eager": eager_t, "compiled": compiled_t}
        speedup = eager_t / compiled_t if compiled_t > 0 else float("inf")
        print(f"{name:8s} eager {eager_t*1000:8.3f} ms  compiled {compiled_t*1000:8.3f} ms  ({speedup:.2f}x)")
    out = Path(__file__).parent / "results_kforge.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
