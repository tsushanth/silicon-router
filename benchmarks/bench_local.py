"""Real, measured matmul latency on every local backend (CPU, MPS).

No simulated numbers - this is the ground truth the router's routing table
is built from. Run standalone: python3 benchmarks/bench_local.py
"""
import json
import time
from pathlib import Path

import torch

# Representative of transformer linear-layer shapes at a few batch/seq sizes,
# not a single arbitrary matmul - the routing decision needs to vary with
# workload size, since small ops rarely justify MPS/GPU dispatch overhead.
SHAPES = [
    ("tiny", 1, 256, 256),
    ("small", 8, 512, 512),
    ("medium", 32, 1024, 1024),
    ("large", 64, 2048, 2048),
    ("xlarge", 128, 4096, 4096),
    ("xxlarge", 256, 4096, 4096),
    ("huge", 256, 8192, 8192),
]
BACKENDS = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
WARMUP = 3
ITERS = 10


def bench(device: str, batch: int, dim_in: int, dim_out: int) -> float:
    x = torch.randn(batch, dim_in, device=device)
    w = torch.randn(dim_in, dim_out, device=device)
    for _ in range(WARMUP):
        (x @ w).sum().item()  # force sync
    start = time.perf_counter()
    for _ in range(ITERS):
        (x @ w).sum().item()
    return (time.perf_counter() - start) / ITERS


def main():
    results = {}
    for name, batch, din, dout in SHAPES:
        results[name] = {}
        for backend in BACKENDS:
            t = bench(backend, batch, din, dout)
            results[name][backend] = t
            print(f"{name:8s} {backend:5s} {t*1000:8.3f} ms")
    out = Path(__file__).parent / "results_local.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
