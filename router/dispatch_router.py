"""The real routing engine (Milestone 3): picks a Backend and actually
executes on it, instead of router.py's older role of just reporting
which backend measured fastest from a static results_*.json table.

Wires in Phase 5's finding directly: routing depends on how many ops are
being dispatched together, not just op size, so this calibrates per
(dim, count) - not just per dim - and the local/remote crossover falls
out of real measurement rather than being hardcoded.
"""
import json
import time
from pathlib import Path

from backends.base import Backend

CALIBRATION_PATH = Path(__file__).parent.parent / "backends" / "calibration.json"


class DispatchRouter:
    def __init__(self, backends: list[Backend]):
        assert backends, "DispatchRouter needs at least one backend"
        self.backends = {b.name: b for b in backends}
        self._table: dict[str, dict[str, dict[str, float]]] = self._load()

    def _load(self) -> dict:
        if CALIBRATION_PATH.exists():
            return json.loads(CALIBRATION_PATH.read_text())
        return {}

    def _save(self):
        CALIBRATION_PATH.write_text(json.dumps(self._table, indent=2))

    def calibrate(self, dim: int, counts: list[int]):
        """Actually runs `count` real ops on every backend for every count
        in `counts`, at this `dim`, and records the real wall time. This
        is the only source of truth this router uses - no interpolation
        across dims, no guessed crossover points.
        """
        dim_key = str(dim)
        self._table.setdefault(dim_key, {})
        for name, backend in self.backends.items():
            self._table[dim_key].setdefault(name, {})
            for count in counts:
                wall, _ = backend.run_batch(dim, count)
                self._table[dim_key][name][str(count)] = wall
                print(f"calibrated {name:20s} dim={dim} count={count:4d} -> {wall*1000:9.2f}ms")
        self._save()

    def _nearest_calibrated_count(self, dim_key: str, backend_name: str, count: int) -> int | None:
        counts = self._table.get(dim_key, {}).get(backend_name, {})
        if not counts:
            return None
        return min((int(c) for c in counts), key=lambda c: abs(c - count))

    def route(self, dim: int, count: int) -> dict:
        """Which backend would be picked for this (dim, count), and why -
        based on the nearest calibrated count per backend. Does not
        execute anything.
        """
        dim_key = str(dim)
        if dim_key not in self._table:
            raise RuntimeError(
                f"no calibration for dim={dim} - call calibrate(dim, counts) first. "
                "This router refuses to route on an uncalibrated dimension."
            )
        estimates = {}
        for name in self.backends:
            nearest = self._nearest_calibrated_count(dim_key, name, count)
            if nearest is None:
                continue
            estimates[name] = self._table[dim_key][name][str(nearest)]
        if not estimates:
            raise RuntimeError(f"no backend has calibration data for dim={dim}")
        winner = min(estimates, key=estimates.get)
        return {
            "dim": dim,
            "count": count,
            "backend": winner,
            "estimated_s": estimates[winner],
            "all_estimates_s": estimates,
        }

    def dispatch(self, dim: int, count: int):
        """Routes AND actually executes the real work on the winning
        backend. Returns (decision, wall_seconds, result) - the wall time
        here is the real time for THIS call (exact count), which may
        differ slightly from route()'s estimate (nearest calibrated
        count) since real execution isn't interpolated.
        """
        decision = self.route(dim, count)
        backend = self.backends[decision["backend"]]
        start = time.perf_counter()
        wall, result = backend.run_batch(dim, count)
        actual_wall = time.perf_counter() - start
        return decision, wall, result
