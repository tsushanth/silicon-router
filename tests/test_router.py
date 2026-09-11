"""Unit tests for router/router.py's deterministic decision logic:
nearest_size_class() and route(). These don't run any real benchmark -
they exercise the table-lookup/comparison logic against small, synthetic
results_*.json fixtures written to a temp directory, so they're fast and
need no GPU/MPS/torch.

Plain stdlib unittest, matching this repo's zero-extra-dependency style
(see CONTRIBUTING.md / ROADMAP.md Milestone 4).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from router import router  # noqa: E402


class TestNearestSizeClass(unittest.TestCase):
    def test_exact_match(self):
        # "medium" == 32 * 1024 * 1024 elems exactly
        self.assertEqual(router.nearest_size_class(32, 1024, 1024), "medium")

    def test_exact_match_tiny(self):
        self.assertEqual(router.nearest_size_class(1, 256, 256), "tiny")

    def test_exact_match_huge(self):
        self.assertEqual(router.nearest_size_class(256, 8192, 8192), "huge")

    def test_nearest_below_smallest_still_tiny(self):
        # Anything smaller than "tiny" should still snap to "tiny" (nearest).
        self.assertEqual(router.nearest_size_class(1, 1, 1), "tiny")

    def test_nearest_above_largest_still_huge(self):
        # Anything bigger than "huge" should still snap to "huge" (nearest).
        self.assertEqual(router.nearest_size_class(1024, 8192, 8192), "huge")

    def test_picks_closer_of_two_neighbors(self):
        # elems here is between "small" (8*512*512=2_097_152) and "medium"
        # (32*1024*1024=33_554_432), but much closer to "small".
        cls = router.nearest_size_class(9, 512, 512)
        self.assertEqual(cls, "small")


class TestRoute(unittest.TestCase):
    """route() reads benchmarks/results_local.json and results_cuda.json
    via module-level Path constants. We point those constants at a temp
    dir with small, hand-written fixture tables instead of touching the
    real benchmark output files (which belong to the parallel Milestone 1
    work / real hardware runs, not this test)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        tmp_path = Path(self.tmpdir.name)

        self.local_path = tmp_path / "results_local.json"
        self.cuda_path = tmp_path / "results_cuda.json"

        local_table = {
            "tiny": {"cpu": 0.000005, "mps": 0.000490},
            "small": {"cpu": 0.000011, "mps": 0.000612},
            "huge": {"cpu": 0.036058, "mps": 0.007711},
        }
        self.local_path.write_text(json.dumps(local_table))

        cuda_table = {
            "tiny": {"cuda": 0.000043},
            "huge": {"cuda": 0.001464},
        }
        self.cuda_path.write_text(json.dumps(cuda_table))

        patcher_local = mock.patch.object(router, "LOCAL_RESULTS_PATH", self.local_path)
        patcher_cuda = mock.patch.object(router, "CUDA_RESULTS_PATH", self.cuda_path)
        patcher_local.start()
        patcher_cuda.start()
        self.addCleanup(patcher_local.stop)
        self.addCleanup(patcher_cuda.stop)

    def test_missing_local_results_raises(self):
        missing = Path(self.tmpdir.name) / "does_not_exist.json"
        with mock.patch.object(router, "LOCAL_RESULTS_PATH", missing):
            with self.assertRaises(FileNotFoundError):
                router.route(1, 256, 256)

    def test_small_shape_routes_local_cpu(self):
        # tiny: cpu=0.005ms beats mps=0.49ms and cuda is excluded (no
        # network_overhead_ms given) -> cpu should win.
        decision = router.route(1, 256, 256)
        self.assertEqual(decision["backend"], "cpu")
        self.assertEqual(decision["size_class"], "tiny")
        self.assertTrue(decision["cuda_excluded_no_network_overhead_given"])
        self.assertNotIn("cuda", decision["all_measured_ms"])

    def test_large_shape_routes_mps(self):
        # huge: cpu=36ms, mps=7.7ms -> mps wins when cuda excluded.
        decision = router.route(256, 8192, 8192)
        self.assertEqual(decision["backend"], "mps")
        self.assertEqual(decision["size_class"], "huge")

    def test_cuda_excluded_without_network_overhead(self):
        decision = router.route(256, 8192, 8192, network_overhead_ms=None)
        self.assertNotIn("cuda", decision["all_measured_ms"])
        self.assertTrue(decision["cuda_excluded_no_network_overhead_given"])

    def test_cuda_included_and_can_win_with_low_overhead(self):
        # huge cuda compute = 1.464ms; with 1ms overhead -> 2.464ms,
        # still beats mps's 7.711ms.
        decision = router.route(256, 8192, 8192, network_overhead_ms=1.0)
        self.assertFalse(decision["cuda_excluded_no_network_overhead_given"])
        self.assertIn("cuda", decision["all_measured_ms"])
        self.assertEqual(decision["backend"], "cuda")
        self.assertAlmostEqual(decision["measured_ms"], 1.464 + 1.0, places=3)

    def test_cuda_loses_with_large_network_overhead(self):
        # tiny cuda compute = 0.043ms; with 30ms overhead -> 30.043ms,
        # loses badly to cpu's 0.005ms - matches README's documented
        # finding that a single op never justifies a remote round trip.
        decision = router.route(1, 256, 256, network_overhead_ms=30.0)
        self.assertEqual(decision["backend"], "cpu")
        self.assertIn("cuda", decision["all_measured_ms"])  # considered, just lost

    def test_route_without_cuda_results_file(self):
        # If results_cuda.json doesn't exist at all, route() should still
        # work fine on local-only data (cuda never enters the table).
        missing_cuda = Path(self.tmpdir.name) / "no_such_cuda.json"
        with mock.patch.object(router, "CUDA_RESULTS_PATH", missing_cuda):
            decision = router.route(1, 256, 256)
            self.assertNotIn("cuda", decision["all_measured_ms"])
            self.assertFalse(decision["cuda_excluded_no_network_overhead_given"])


if __name__ == "__main__":
    unittest.main()
