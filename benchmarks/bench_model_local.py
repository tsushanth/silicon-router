"""Milestone 1: routing decisions for a real transformer block's forward
pass, not a single matmul. Same WARMUP/ITERS discipline as
bench_local.py - the first ad-hoc test of this (see commit history)
mixed one-time model-init cost into the measurement and produced
nonsensical numbers (CPU getting FASTER as seq_len grew); this script
exists specifically to not repeat that mistake.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, for `backends`/`models` imports

import torch

from backends.local import LocalBackend

BATCH = 4
SEQ_LENS = [16, 64, 256, 1024]
WARMUP = 3
ITERS = 5
BACKENDS = [LocalBackend("cpu")] + ([LocalBackend("mps")] if torch.backends.mps.is_available() else [])


def bench(backend: LocalBackend, seq_len: int) -> float:
    for _ in range(WARMUP):
        backend.run_model_batch(BATCH, seq_len)
    total = 0.0
    for _ in range(ITERS):
        wall, _ = backend.run_model_batch(BATCH, seq_len)
        total += wall
    return total / ITERS


def main():
    results = {}
    for seq_len in SEQ_LENS:
        results[str(seq_len)] = {}
        for backend in BACKENDS:
            t = bench(backend, seq_len)
            results[str(seq_len)][backend.name] = t
            print(f"seq_len={seq_len:5d}  {backend.name:12s} {t*1000:9.3f} ms")
    out = Path(__file__).parent / "results_model_local.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
