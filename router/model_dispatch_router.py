"""Milestone 1: the same calibrate/route/dispatch pattern as
DispatchRouter (router/dispatch_router.py), but keyed by (batch, seq_len)
against Backend.run_model_batch instead of (dim, count) against
run_batch - a real transformer-block forward pass, not a matmul.

Kept as a separate class rather than generalizing DispatchRouter's
signature, specifically to not disturb the just-shipped DispatchRouter
and its test suite (tests/test_dispatch_router.py) mid-flight - this is
the same design, applied to the workload Milestone 1 asks for.
"""
import json
import time
from pathlib import Path

from backends.base import Backend

CALIBRATION_PATH = Path(__file__).parent.parent / "backends" / "model_calibration.json"


class ModelDispatchRouter:
    def __init__(self, backends: list[Backend]):
        assert backends, "ModelDispatchRouter needs at least one backend"
        self.backends = {b.name: b for b in backends}
        self._table: dict[str, dict[str, float]] = self._load()

    def _load(self) -> dict:
        if CALIBRATION_PATH.exists():
            return json.loads(CALIBRATION_PATH.read_text())
        return {}

    def _save(self):
        CALIBRATION_PATH.write_text(json.dumps(self._table, indent=2))

    def calibrate(self, batch: int, seq_lens: list[int]):
        for seq_len in seq_lens:
            key = f"{batch}:{seq_len}"
            self._table.setdefault(key, {})
            for name, backend in self.backends.items():
                wall, _ = backend.run_model_batch(batch, seq_len)
                self._table[key][name] = wall
                print(f"calibrated {name:20s} batch={batch} seq_len={seq_len:5d} -> {wall*1000:9.2f}ms")
        self._save()

    def route(self, batch: int, seq_len: int) -> dict:
        key = f"{batch}:{seq_len}"
        if key not in self._table:
            raise RuntimeError(
                f"no calibration for batch={batch}, seq_len={seq_len} - call calibrate() first. "
                "This router refuses to route on an uncalibrated workload."
            )
        estimates = self._table[key]
        winner = min(estimates, key=estimates.get)
        return {
            "batch": batch,
            "seq_len": seq_len,
            "backend": winner,
            "estimated_s": estimates[winner],
            "all_estimates_s": dict(estimates),
        }

    def dispatch(self, batch: int, seq_len: int):
        decision = self.route(batch, seq_len)
        backend = self.backends[decision["backend"]]
        start = time.perf_counter()
        wall, result = backend.run_model_batch(batch, seq_len)
        actual_wall = time.perf_counter() - start
        return decision, wall, result
