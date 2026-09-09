"""Routes a matmul-shaped workload to whichever backend measured fastest
for that size class - local (CPU/MPS) or a remote CUDA GPU - using the
benchmarks/results_*.json files as the routing table. No heuristics, no
guessed crossover point - the table is built from real measured timings.
"""
import json
from pathlib import Path

LOCAL_RESULTS_PATH = Path(__file__).parent.parent / "benchmarks" / "results_local.json"
CUDA_RESULTS_PATH = Path(__file__).parent.parent / "benchmarks" / "results_cuda.json"

# (name, batch, dim_in, dim_out) must match benchmarks/bench_local.py's SHAPES
SIZE_CLASSES = [
    ("tiny", 1 * 256 * 256),
    ("small", 8 * 512 * 512),
    ("medium", 32 * 1024 * 1024),
    ("large", 64 * 2048 * 2048),
    ("xlarge", 128 * 4096 * 4096),
    ("xxlarge", 256 * 4096 * 4096),
    ("huge", 256 * 8192 * 8192),
]


def load_routing_table() -> dict:
    """Merges local and remote-CUDA timings into one table, keyed by size
    class, each holding {backend: seconds}. The remote CUDA compute time
    alone (e.g. 0.04ms) is not comparable to local timings without adding
    real network round-trip latency - see route()'s network_overhead_ms.
    """
    if not LOCAL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{LOCAL_RESULTS_PATH} missing - run `python3 benchmarks/bench_local.py` first. "
            "This router refuses to route on guessed numbers."
        )
    table = json.loads(LOCAL_RESULTS_PATH.read_text())
    if CUDA_RESULTS_PATH.exists():
        cuda = json.loads(CUDA_RESULTS_PATH.read_text())
        for size_class, timings in table.items():
            if size_class in cuda:
                timings["cuda"] = cuda[size_class]["cuda"]
    return table


def nearest_size_class(batch: int, dim_in: int, dim_out: int) -> str:
    elems = batch * dim_in * dim_out
    best_name, best_dist = None, float("inf")
    for name, ref_elems in SIZE_CLASSES:
        dist = abs(ref_elems - elems)
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


def route(batch: int, dim_in: int, dim_out: int, network_overhead_ms: float | None = None) -> dict:
    """network_overhead_ms: measured round-trip latency to the remote GPU
    (e.g. via `ping` or a real request timing to that host). Left as None
    by default, which EXCLUDES the cuda backend from consideration rather
    than comparing raw remote compute time against local wall-clock time -
    that comparison is meaningless without accounting for the network hop
    a real remote call would pay. Pass a real measured value to include it.
    """
    table = load_routing_table()
    size_class = nearest_size_class(batch, dim_in, dim_out)
    timings = dict(table[size_class])
    if "cuda" in timings:
        if network_overhead_ms is None:
            timings.pop("cuda")
        else:
            timings["cuda"] = timings["cuda"] + network_overhead_ms / 1000
    backend = min(timings, key=timings.get)
    return {
        "size_class": size_class,
        "backend": backend,
        "measured_ms": timings[backend] * 1000,
        "all_measured_ms": {k: v * 1000 for k, v in timings.items()},
        "cuda_excluded_no_network_overhead_given": "cuda" not in timings and "cuda" in table[size_class],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (4, 5):
        print("usage: router.py <batch> <dim_in> <dim_out> [network_overhead_ms]")
        sys.exit(1)
    b, di, do = map(int, sys.argv[1:4])
    overhead = float(sys.argv[4]) if len(sys.argv) == 5 else None
    decision = route(b, di, do, overhead)
    print(json.dumps(decision, indent=2))
