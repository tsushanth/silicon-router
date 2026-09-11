"""Minimal HTTP server for the batched-remote-dispatch experiment (Milestone 2).

Accepts POST /batch {"batch_size": N, "dim": D} and runs N real DxD @ DxD
matmuls on this machine's CUDA device in one request, returning the
server-side compute time. Deliberately stdlib-only (http.server) - no
framework needed for a single endpoint, and it keeps the remote image's
dependency surface to just torch.

This is the piece that makes the batching experiment honest: a real TCP
round trip over the network, not an SSH `exec` (which has its own,
unrepresentative connection-setup overhead).
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

assert torch.cuda.is_available(), "no CUDA device visible on this server"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep stdout clean for the benchmark's own prints

    def do_POST(self):
        if self.path != "/batch":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        batch_size = int(body["batch_size"])
        dim = int(body["dim"])

        x = torch.randn(dim, dim, device="cuda")
        w = torch.randn(dim, dim, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(batch_size):
            (x @ w).sum().item()
        torch.cuda.synchronize()
        server_compute_s = time.perf_counter() - start

        resp = json.dumps({"server_compute_s": server_compute_s}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"listening on 0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
