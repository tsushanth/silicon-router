"""Local backend: CPU, MPS (Apple Silicon GPU), or CUDA - whatever torch
reports available on THIS machine, no network hop. Same math as
benchmarks/bench_local.py, now behind the Backend interface so the
router can call it directly instead of reading it back out of a
results_*.json file.

"cuda" here means a local CUDA device, not a remote one - this is what
makes Jetson's Orin iGPU usable through the same class as Mac's MPS,
since both are "compute attached to the machine actually running this
code," unlike backends/remote_http.py's genuinely remote GPU over a
network round-trip.
"""
import time

import torch

from backends.base import Backend
from models.tiny_transformer import TinyTransformerBlock, make_input


class LocalBackend(Backend):
    def __init__(self, device: str):
        assert device in ("cpu", "mps", "cuda"), f"unsupported local device: {device}"
        if device == "mps":
            assert torch.backends.mps.is_available(), "MPS requested but not available"
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA requested but not available"
        self.device = device
        self.name = f"local:{device}"
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = TinyTransformerBlock().to(self.device).eval()
        return self._model

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

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

    def run_model_batch(self, batch: int, seq_len: int):
        model = self._get_model()
        x = make_input(batch, seq_len, self.device)
        self._sync()
        start = time.perf_counter()
        with torch.no_grad():
            result = model(x)
            result.sum().item()
        self._sync()
        return time.perf_counter() - start, result
