"""Minimal HTTP server for the batched-remote-dispatch experiments
(Milestone 2's /batch, Milestone 1's /model_batch).

POST /batch {"batch_size": N, "dim": D} runs N real DxD @ DxD matmuls.
POST /model_batch {"batch": B, "seq_len": S} runs a real transformer
block's forward pass (same TinyTransformerBlock as
models/tiny_transformer.py - duplicated inline, not imported, because
this file is deployed standalone via scp to the remote GPU host, not
the whole repo).

Deliberately stdlib-only (http.server) - no framework needed, and it
keeps the remote image's dependency surface to just torch. This is the
piece that makes both experiments honest: a real TCP round trip over
the network, not an SSH `exec` (which has its own, unrepresentative
connection-setup overhead).
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn

assert torch.cuda.is_available(), "no CUDA device visible on this server"

D_MODEL, N_HEAD, DIM_FEEDFORWARD = 512, 8, 2048
_model = nn.TransformerEncoderLayer(
    d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=DIM_FEEDFORWARD, batch_first=True,
).to("cuda").eval()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep stdout clean for the benchmark's own prints

    def _respond(self, server_compute_s: float):
        resp = json.dumps({"server_compute_s": server_compute_s}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_POST(self):
        if self.path not in ("/batch", "/model_batch"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))

        if self.path == "/batch":
            batch_size = int(body["batch_size"])
            dim = int(body["dim"])
            x = torch.randn(dim, dim, device="cuda")
            w = torch.randn(dim, dim, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(batch_size):
                (x @ w).sum().item()
            torch.cuda.synchronize()
            self._respond(time.perf_counter() - start)
        else:  # /model_batch
            batch = int(body["batch"])
            seq_len = int(body["seq_len"])
            x = torch.randn(batch, seq_len, D_MODEL, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                out = _model(x)
                out.sum().item()
            torch.cuda.synchronize()
            self._respond(time.perf_counter() - start)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"listening on 0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
