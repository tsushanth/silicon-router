"""Remote backend: talks to workers/batch_server.py over real HTTP. This
is what Phase 5 measured - a genuine network round trip, not an SSH exec.
Requires a batch_server.py already running and reachable at `url`; this
class doesn't provision or tear down the remote host - that's an
infrastructure decision the caller makes deliberately (see README's
GPU-cost-discipline notes), not something a backend should do silently.
"""
import json
import time
import urllib.request

from backends.base import Backend


class RemoteHTTPBackend(Backend):
    def __init__(self, url: str):
        assert url.endswith("/batch"), (
            "expected a url ending in /batch (e.g. http://host:8080/batch) - "
            "model_batch's endpoint is derived from it, not passed separately"
        )
        self.url = url
        self.name = f"remote:{url}"

    def _post(self, dim: int, count: int) -> float:
        payload = json.dumps({"batch_size": count, "dim": dim}).encode()
        # RunPod's HTTP proxy 403s requests with no/unusual User-Agent -
        # matching curl's rather than depending on proxy internals.
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            json.loads(resp.read())
        return time.perf_counter() - start

    def benchmark_op(self, dim: int) -> float:
        return self._post(dim, 1)

    def run_batch(self, dim: int, count: int):
        wall = self._post(dim, count)
        # The server executes real CUDA matmuls and returns compute time,
        # not the tensor itself (shipping a 4096x4096 float tensor back
        # over HTTP would make network cost dominate for reasons that have
        # nothing to do with the routing question this repo is testing).
        return wall, None

    def _post_model(self, batch: int, seq_len: int) -> float:
        payload = json.dumps({"batch": batch, "seq_len": seq_len}).encode()
        req = urllib.request.Request(
            self.url.replace("/batch", "/model_batch"), data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            json.loads(resp.read())
        return time.perf_counter() - start

    def run_model_batch(self, batch: int, seq_len: int):
        wall = self._post_model(batch, seq_len)
        return wall, None
