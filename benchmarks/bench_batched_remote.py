"""Milestone 2: does batching ops into one remote call change Phase 2's
verdict ("a single op never justifies remote dispatch")?

For increasing batch sizes, measures:
  - remote_wall_s: real end-to-end time for ONE HTTP round trip that runs
    `batch_size` matmuls on the remote GPU (workers/batch_server.py) - this
    is genuine network + HTTP overhead, not an SSH exec.
  - local_wall_s: real wall-clock time to run the identical `batch_size`
    matmuls locally, one HTTP round trip's worth of overhead paid zero
    times (the whole point of "local" - no network at all).

Finds the batch size (if any) where remote crosses over to winning.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

DIM = 4096  # matches the "xlarge"-ish regime from earlier phases
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WARMUP_REMOTE_CALLS = 2


def local_batch(batch_size: int, device: str) -> float:
    x = torch.randn(DIM, DIM, device=device)
    w = torch.randn(DIM, DIM, device=device)
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(batch_size):
        (x @ w).sum().item()
    if device == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - start


def remote_batch(url: str, batch_size: int) -> float:
    payload = json.dumps({"batch_size": batch_size, "dim": DIM}).encode()
    # RunPod's HTTP proxy 403s requests with no/unusual User-Agent (curl's
    # default UA passes, urllib's doesn't) - match curl's rather than debug
    # their proxy's filtering further.
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    wall_s = time.perf_counter() - start
    return wall_s, body["server_compute_s"]


def main():
    if len(sys.argv) != 2:
        print("usage: bench_batched_remote.py <remote_url e.g. http://1.2.3.4:8080/batch>")
        sys.exit(1)
    url = sys.argv[1]
    local_device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"warming up remote endpoint ({WARMUP_REMOTE_CALLS} calls)...")
    for _ in range(WARMUP_REMOTE_CALLS):
        remote_batch(url, 1)

    results = {}
    for bs in BATCH_SIZES:
        remote_wall, server_compute = remote_batch(url, bs)
        local_wall = local_batch(bs, local_device)
        network_overhead = remote_wall - server_compute
        winner = "remote" if remote_wall < local_wall else "local"
        results[bs] = {
            "remote_wall_s": remote_wall,
            "server_compute_s": server_compute,
            "network_overhead_s": network_overhead,
            "local_wall_s": local_wall,
            "winner": winner,
        }
        print(
            f"batch={bs:4d}  remote_wall={remote_wall*1000:9.2f}ms "
            f"(compute={server_compute*1000:7.2f}ms, network={network_overhead*1000:7.2f}ms)  "
            f"local_wall={local_wall*1000:9.2f}ms  -> {winner}"
        )

    out = Path(__file__).parent / "results_batched_remote.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
