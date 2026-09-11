"""Local backend: CPU or MPS (Apple Silicon GPU), whatever torch reports
available on this machine. Same math as benchmarks/bench_local.py, now
behind the Backend interface so the router can call it directly instead
of reading it back out of a results_*.json file.
"""
import time

import torch

from backends.base import Backend


class LocalBackend(Backend):
    def __init__(self, device: str):
        assert device in ("cpu", "mps"), f"unsupported local device: {device}"
        if device == "mps":
            assert torch.backends.mps.is_available(), "MPS requested but not available"
        self.device = device
        self.name = f"local:{device}"

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    def benchmark_op(self, dim: int) -> float:
        wall, _ = self.run_batch(dim, 1)
        return wall

    def run_batch(self, dim: int, count: int):
        x = torch.randn(dim, dim, device=self.device)
        w = torch.randn(dim, dim, device=self.device)
        self._sync()
        start = time.perf_counter()
        result = None
        for _ in range(count):
            result = x @ w
            result.sum().item()
        self._sync()
        return time.perf_counter() - start, result
