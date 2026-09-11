"""Milestone 1, remote half: the same real transformer-block forward
pass as bench_model_local.py, run against a live workers/batch_server.py
over real HTTP. Same WARMUP/ITERS discipline - a first ad-hoc test of
this without enough warmup produced a wildly inflated seq_len=16 number
(cuDNN/cuBLAS algorithm search on the server's first real call), which
is exactly the mistake this script's warmup loop exists to avoid.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.remote_http import RemoteHTTPBackend

BATCH = 4
SEQ_LENS = [16, 64, 256, 1024]
WARMUP = 5
ITERS = 5


def bench(backend: RemoteHTTPBackend, seq_len: int) -> float:
    for _ in range(WARMUP):
        backend.run_model_batch(BATCH, seq_len)
    total = 0.0
    for _ in range(ITERS):
        wall, _ = backend.run_model_batch(BATCH, seq_len)
        total += wall
    return total / ITERS


def main():
    if len(sys.argv) != 2:
        print("usage: bench_model_remote.py <remote_url e.g. https://host-8080.proxy.runpod.net/batch>")
        sys.exit(1)
    backend = RemoteHTTPBackend(sys.argv[1])
    results = {}
    for seq_len in SEQ_LENS:
        t = bench(backend, seq_len)
        results[str(seq_len)] = t
        print(f"seq_len={seq_len:5d}  remote  {t*1000:9.2f} ms")
    out = Path(__file__).parent / "results_model_remote.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
