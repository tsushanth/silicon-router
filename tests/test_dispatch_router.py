"""Unit tests for router/dispatch_router.py's DispatchRouter: calibrate(),
route(), and dispatch()'s decision logic. These use a FakeBackend (a
Backend subclass returning synthetic, deterministic timings instead of
running real GPU/CPU work) so the tests are fast, need no torch/GPU, and
never touch backends/local.py or backends/remote_http.py, which belong
to the real hardware execution path (out of scope for this task).

backends/base.py's Backend ABC is explicitly designed for this kind of
substitution - benchmark_op/run_batch just need to return
(wall_seconds, result); nothing in DispatchRouter cares how those numbers
were produced.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.base import Backend  # noqa: E402
from router import dispatch_router as dr  # noqa: E402


class FakeBackend(Backend):
    """Returns a synthetic wall time per (dim, count) from a lookup table
    supplied at construction, instead of doing any real compute. Lets
    tests express "this backend is fast/slow at this batch size" directly,
    the same way a real backend's calibrate() would discover it from
    measurement - just without needing real hardware.
    """

    def __init__(self, name: str, timings: dict):
        self.name = name
        # timings: {count: wall_seconds}
        self.timings = timings
        self.calls = []

    def benchmark_op(self, dim: int) -> float:
        wall, _ = self.run_batch(dim, 1)
        return wall

    def run_batch(self, dim: int, count: int):
        self.calls.append((dim, count))
        if count not in self.timings:
            raise KeyError(f"FakeBackend {self.name} has no fixture timing for count={count}")
        return self.timings[count], f"fake-result-{self.name}-{dim}-{count}"

    def run_model_batch(self, batch: int, seq_len: int):
        # Not exercised by these tests (Milestone 1's real-transformer
        # path, out of scope here) - implemented only so FakeBackend
        # satisfies the Backend ABC's full interface.
        raise NotImplementedError("FakeBackend.run_model_batch is not used by these tests")


class TestCalibrate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        cal_path = Path(self.tmpdir.name) / "calibration.json"
        patcher = mock.patch.object(dr, "CALIBRATION_PATH", cal_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cal_path = cal_path

    def test_calibrate_records_real_synthetic_timings_and_persists(self):
        local = FakeBackend("local", {1: 0.001, 32: 0.010, 64: 0.015})
        remote = FakeBackend("remote", {1: 0.030, 32: 0.032, 64: 0.033})
        router = dr.DispatchRouter([local, remote])
        router.calibrate(dim=1024, counts=[1, 32, 64])

        # In-memory table populated correctly.
        table = router._table["1024"]
        self.assertEqual(table["local"]["1"], 0.001)
        self.assertEqual(table["local"]["64"], 0.015)
        self.assertEqual(table["remote"]["1"], 0.030)

        # Every backend actually got called for every count - calibrate()
        # is measurement, not interpolation.
        self.assertEqual(local.calls, [(1024, 1), (1024, 32), (1024, 64)])
        self.assertEqual(remote.calls, [(1024, 1), (1024, 32), (1024, 64)])

        # Persisted to disk.
        self.assertTrue(self.cal_path.exists())
        on_disk = json.loads(self.cal_path.read_text())
        self.assertEqual(on_disk["1024"]["remote"]["32"], 0.032)

    def test_router_loads_existing_calibration_from_disk(self):
        self.cal_path.write_text(json.dumps({"512": {"local": {"1": 0.002}}}))
        local = FakeBackend("local", {})
        router = dr.DispatchRouter([local])
        self.assertEqual(router._table["512"]["local"]["1"], 0.002)


class TestRoute(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        cal_path = Path(self.tmpdir.name) / "calibration.json"
        patcher = mock.patch.object(dr, "CALIBRATION_PATH", cal_path)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Local: cheap at low batch, expensive-ish at high batch.
        # Remote: expensive at low batch (network overhead dominates),
        # cheaper at high batch - mirrors README's Phase 5 batching finding.
        self.local = FakeBackend("local", {1: 0.001, 16: 0.010, 32: 0.018, 256: 0.140})
        self.remote = FakeBackend("remote", {1: 0.030, 16: 0.031, 32: 0.032, 256: 0.040})
        self.router = dr.DispatchRouter([self.local, self.remote])
        self.router.calibrate(dim=1024, counts=[1, 16, 32, 256])

    def test_uncalibrated_dim_raises(self):
        with self.assertRaises(RuntimeError):
            self.router.route(dim=2048, count=16)

    def test_low_batch_routes_local(self):
        decision = self.router.route(dim=1024, count=1)
        self.assertEqual(decision["backend"], "local")
        self.assertEqual(decision["estimated_s"], 0.001)

    def test_high_batch_routes_remote(self):
        decision = self.router.route(dim=1024, count=256)
        self.assertEqual(decision["backend"], "remote")
        self.assertEqual(decision["estimated_s"], 0.040)

    def test_crossover_at_batch_32_favors_local(self):
        # local=0.018 vs remote=0.032 at count=32 -> local still wins here,
        # matching this fixture's deliberately-placed crossover above 32.
        decision = self.router.route(dim=1024, count=32)
        self.assertEqual(decision["backend"], "local")

    def test_uncalibrated_count_snaps_to_nearest(self):
        # count=20 isn't calibrated directly; should snap to nearest
        # calibrated count (16) per backend.
        decision = self.router.route(dim=1024, count=20)
        self.assertEqual(decision["all_estimates_s"]["local"], 0.010)  # nearest to 20 is 16
        self.assertEqual(decision["all_estimates_s"]["remote"], 0.031)

    def test_route_does_not_execute_backend(self):
        local = FakeBackend("local", {1: 0.001})
        router = dr.DispatchRouter([local])
        router.calibrate(dim=64, counts=[1])
        local.calls.clear()
        router.route(dim=64, count=1)
        self.assertEqual(local.calls, [], "route() must not call run_batch - it only estimates")


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        cal_path = Path(self.tmpdir.name) / "calibration.json"
        patcher = mock.patch.object(dr, "CALIBRATION_PATH", cal_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dispatch_executes_the_winning_backend_only(self):
        local = FakeBackend("local", {1: 0.001, 64: 0.015})
        remote = FakeBackend("remote", {1: 0.030, 64: 0.010})
        router = dr.DispatchRouter([local, remote])
        router.calibrate(dim=256, counts=[1, 64])
        local.calls.clear()
        remote.calls.clear()

        decision, wall, result = router.dispatch(dim=256, count=64)

        self.assertEqual(decision["backend"], "remote")
        # Only the winning backend should have actually executed.
        self.assertEqual(local.calls, [])
        self.assertEqual(remote.calls, [(256, 64)])
        self.assertEqual(wall, 0.010)
        self.assertEqual(result, "fake-result-remote-256-64")

    def test_dispatch_raises_for_uncalibrated_dim(self):
        local = FakeBackend("local", {1: 0.001})
        router = dr.DispatchRouter([local])
        with self.assertRaises(RuntimeError):
            router.dispatch(dim=999, count=1)


if __name__ == "__main__":
    unittest.main()
