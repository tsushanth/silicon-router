"""Routes a matmul-shaped workload to whichever local backend measured
fastest for that size class, using benchmarks/results_local.json as the
routing table. No heuristics, no guessed crossover point - the table is
built from bench_local.py's actual timings.
"""
import json
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "benchmarks" / "results_local.json"

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
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} missing - run `python3 benchmarks/bench_local.py` first. "
            "This router refuses to route on guessed numbers."
        )
    return json.loads(RESULTS_PATH.read_text())


def nearest_size_class(batch: int, dim_in: int, dim_out: int) -> str:
    elems = batch * dim_in * dim_out
    best_name, best_dist = None, float("inf")
    for name, ref_elems in SIZE_CLASSES:
        dist = abs(ref_elems - elems)
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


def route(batch: int, dim_in: int, dim_out: int) -> dict:
    table = load_routing_table()
    size_class = nearest_size_class(batch, dim_in, dim_out)
    timings = table[size_class]
    backend = min(timings, key=timings.get)
    return {
        "size_class": size_class,
        "backend": backend,
        "measured_ms": timings[backend] * 1000,
        "all_measured_ms": {k: v * 1000 for k, v in timings.items()},
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("usage: router.py <batch> <dim_in> <dim_out>")
        sys.exit(1)
    b, di, do = map(int, sys.argv[1:4])
    decision = route(b, di, do)
    print(json.dumps(decision, indent=2))
