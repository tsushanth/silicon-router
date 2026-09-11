"""The pluggable backend interface (Milestone 3).

Adding a new backend - a different GPU, a TPU, an edge device - means
implementing this class, not editing router internals. Two methods only,
both real: `benchmark_op` measures one op's real cost on this backend
(so the router never routes on guessed numbers), `run_batch` actually
executes `count` real ops and returns (wall_seconds, result) - the
router calls this to do the real work, not just to time it.
"""
from abc import ABC, abstractmethod


class Backend(ABC):
    name: str

    @abstractmethod
    def benchmark_op(self, dim: int) -> float:
        """Real measured wall-clock seconds for ONE dim x dim @ dim x dim
        matmul on this backend, including whatever per-call overhead a
        real dispatch to it would pay (network round-trip for a remote
        backend, dispatch overhead for a local GPU, etc). No estimates.
        """
        raise NotImplementedError

    @abstractmethod
    def run_batch(self, dim: int, count: int):
        """Actually executes `count` real dim x dim @ dim x dim matmuls
        on this backend, in ONE logical call (one remote round-trip for
        a remote backend, one local loop for a local one). Returns
        (wall_seconds, last_result_tensor_or_none) - real work done, not
        simulated.
        """
        raise NotImplementedError
