"""Reproduces the three-way DispatchRouter result in README's Phase 4
update: local:cpu, local:mps, and a Jetson (or any LAN/remote host
running workers/batch_server.py) registered together in one live
router, calibrated and routed for real.

This script exists because the original run was done as an ad-hoc
interactive command, not a committed script - a real gap against this
repo's own reproducibility bar (see ROADMAP.md Milestone 4, CONTRIBUTING.md).
Fixed here rather than left as README prose nobody else can rerun.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.local import LocalBackend
from backends.remote_http import RemoteHTTPBackend
from router.dispatch_router import DispatchRouter

DIM = 4096
COUNTS = [1, 8, 32, 128]


def main():
    if len(sys.argv) != 2:
        print("usage: bench_dispatch_three_way.py <third_backend_url e.g. http://10.0.0.239:8080/batch>")
        sys.exit(1)
    third_url = sys.argv[1]

    cpu = LocalBackend("cpu")
    mps = LocalBackend("mps")
    third = RemoteHTTPBackend(third_url)
    router = DispatchRouter([cpu, mps, third])

    router.calibrate(dim=DIM, counts=COUNTS)
    print()
    for count in COUNTS:
        decision = router.route(DIM, count)
        estimates = {k: round(v * 1000, 1) for k, v in decision["all_estimates_s"].items()}
        print(f"count={count:4d} -> {decision['backend']:12s} ({decision['estimated_s']*1000:.1f}ms)  all={estimates}")

    print()
    print("dispatch() sanity check - actually executes, not just decides:")
    for count in (COUNTS[0], COUNTS[-1]):
        decision, wall, _ = router.dispatch(DIM, count)
        print(f"  count={count:4d} -> {decision['backend']:12s}  real wall={wall*1000:.1f}ms")


if __name__ == "__main__":
    main()
