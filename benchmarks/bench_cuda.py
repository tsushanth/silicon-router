"""Same matmul benchmark as bench_local.py, run on a CUDA GPU (RunPod).
Kept as a near-identical twin rather than importing bench_local so this
file has zero dependency on the calling machine having torch/mps -
it's meant to be copied to and run entirely on the remote pod.
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
    ("xxlarge", 256, 4096, 4096),
    ("huge", 256, 8192, 8192),
]
WARMUP = 3
ITERS = 10


def bench(batch: int, dim_in: int, dim_out: int) -> float:
    x = torch.randn(batch, dim_in, device="cuda")
    w = torch.randn(dim_in, dim_out, device="cuda")
    for _ in range(WARMUP):
        (x @ w).sum().item()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        (x @ w).sum().item()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / ITERS


def main():
    assert torch.cuda.is_available(), "no CUDA device visible"
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}")
    results = {"_device": name}
    for label, batch, din, dout in SHAPES:
        t = bench(batch, din, dout)
        results[label] = {"cuda": t}
        print(f"{label:8s} cuda {t*1000:8.3f} ms")
    out = Path(__file__).parent / "results_cuda.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
