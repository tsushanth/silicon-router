# silicon-router

A small, honest proof-of-concept exploring the idea behind [Gimlet Labs](https://gimletlabs.ai/) —
routing pieces of an AI workload to whichever chip actually runs them fastest,
instead of pinning everything to one device.

This is a learning/exploration project, not a product — being open
sourced (Apache-2.0) because the findings are more useful public than
private, not for any commercial angle. Every number in this repo is
measured on real hardware, not simulated. See [ROADMAP.md](ROADMAP.md)
for what's actually being worked toward next, and
[CONTRIBUTING.md](CONTRIBUTING.md) if you want to add a backend or a
workload.

## Phase 1 (done): local CPU vs MPS routing

`benchmarks/bench_local.py` times a matmul (the core op in every linear
layer) at increasing sizes on this Mac's CPU and Apple Silicon GPU (MPS),
and writes the real timings to `benchmarks/results_local.json`.

`router/router.py` reads that table and routes a given workload shape to
whichever backend measured faster for its size class. No guessed crossover
point — it's read straight off the benchmark.

**Real measured crossover on an M2 Pro:**

| Size class | Shape (batch × in × out) | CPU (ms) | MPS (ms) | Winner |
|---|---|---|---|---|
| tiny | 1×256×256 | 0.005 | 0.490 | CPU |
| small | 8×512×512 | 0.011 | 0.612 | CPU |
| medium | 32×1024×1024 | 0.061 | 0.298 | CPU |
| large | 64×2048×2048 | 0.806 | 0.841 | CPU (barely) |
| xlarge | 128×4096×4096 | 7.909 | 2.051 | **MPS (3.9x)** |
| xxlarge | 256×4096×4096 | 7.822 | 2.662 | **MPS (2.9x)** |
| huge | 256×8192×8192 | 36.058 | 7.711 | **MPS (4.7x)** |

MPS dispatch overhead dominates below ~large; above it, GPU parallelism
wins by 3-5x. This crossover is the whole reason a router is useful at
all — a static "always use the GPU" policy would be actively slower for
small ops here.

## Phase 2 (done): remote CUDA GPU as a third backend

`benchmarks/bench_cuda.py` runs the identical benchmark on a rented
RunPod GPU (RTX 3090, spun up, benchmarked, and torn down immediately
after — no idle billing). Real numbers, copied back to
`benchmarks/results_cuda.json`:

| Size class | CUDA (RTX 3090, ms) |
|---|---|
| tiny | 0.043 |
| small | 0.048 |
| medium | 0.045 |
| large | 0.085 |
| xlarge | 0.264 |
| xxlarge | 0.457 |
| huge | 1.464 |

Raw CUDA compute crushes both local backends at every size — but that
number alone is misleading for a *remote* GPU, since a real call pays a
network round-trip on top. `router.route()` refuses to compare raw remote
compute time against local wall-clock time: the `cuda` backend is
**excluded from routing unless you pass a real measured
`network_overhead_ms`.**

With a realistic ~30ms round-trip added, the picture flips entirely:

```
huge, no overhead given  -> mps  (7.7ms)   [cuda excluded, no honest comparison possible]
huge, +30ms RTT          -> mps  (7.7ms  vs cuda's 31.5ms)
tiny, +30ms RTT          -> cpu  (0.005ms vs cuda's 30ms)
```

**The actual finding**: local MPS beats remote CUDA-over-network at
*every* size tested here, even "huge." A single matmul is never worth
a remote dispatch — the network hop dominates regardless of how much
faster the remote compute is. This is the real reason systems like
Gimlet route whole workload *segments* to remote hardware, not
individual ops: amortizing one network round-trip across many ops is
the only way remote GPU dispatch pays for itself. This router
intentionally won't let you pretend otherwise with a fabricated
overhead number.

## Phase 3 (done): kforge-style kernel autotuning — a negative result

`benchmarks/bench_kforge.py` compares PyTorch eager mode against
`torch.compile(mode="max-autotune")` on the same matmul shapes — real
automated kernel generation (Triton), not a hand-written custom kernel,
same spirit as Gimlet's kforge on a much smaller scale.

**CPU (M2 Pro)** — compiled is slower at every size:

| Size class | eager (ms) | compiled (ms) | speedup |
|---|---|---|---|
| tiny | 0.004 | 0.020 | 0.22x |
| small | 0.012 | 0.108 | 0.11x |
| medium | 0.065 | 0.177 | 0.37x |
| large | 0.532 | 0.638 | 0.84x |
| xlarge | 5.396 | 5.508 | 0.98x |

**CUDA (RTX 3090)** — same result, and it gets worse with scale. Inductor's
autotune search genuinely ran (real `AUTOTUNE mm(...)` logs, 18 Triton
kernel configs benchmarked, best one selected), so this isn't a
misconfiguration — it's real generated kernels losing to cuBLAS:

| Size class | eager (ms) | compiled (ms) | speedup |
|---|---|---|---|
| tiny | 0.038 | 0.104 | 0.36x |
| small | 0.046 | 0.126 | 0.36x |
| medium | 0.044 | 0.178 | 0.25x |
| large | 0.081 | 0.632 | 0.13x |
| xlarge | 0.261 | 3.578 | **0.07x (14x slower)** |

**Honest takeaway**: for plain matmul, cuBLAS/MKL's hand-tuned kernels are
extremely hard to beat, and torch.compile's autotuning overhead only gets
more expensive relative to the op as size grows. This isn't evidence that
kforge-style kernel generation doesn't work in general — it's evidence
that a single plain matmul is the wrong workload to test it on. Real
kernel-fusion wins (which is what tools like kforge actually target) show
up on *sequences* of ops — fusing an activation, a bias-add, and a matmul
into one kernel avoids the intermediate memory round-trips that eager
mode pays for each op separately. A single isolated matmul has nothing
to fuse. That's the natural next experiment, not attempted here.

## Phase 5 (done): batched remote dispatch — the real crossover

Phase 2 found that a *single* op never justifies a remote GPU call once
network round-trip is counted. This directly tests the natural follow-up:
does batching multiple ops into **one** remote call change that verdict?

`workers/batch_server.py` is a real HTTP server on the rented GPU (not
an SSH exec — genuine TCP/HTTP round-trip, through RunPod's actual
public proxy). `benchmarks/bench_batched_remote.py` sends it increasing
batch sizes of the same 4096×4096 matmul, and compares total wall-clock
time against running that many matmuls locally on MPS — no network at
all, so its cost scales with pure compute.

**Real measured result** (RTX 3090, MPS on M2 Pro, live over the public
internet):

| Batch size | Remote wall (ms) | — compute / network | Local wall (ms) | Winner |
|---|---|---|---|---|
| 1 | 283.6 | 5.7 / 277.9 | 186.0 | local |
| 2 | 321.5 | 11.1 / 310.4 | 61.6 | local |
| 4 | 304.7 | 22.2 / 282.5 | 123.9 | local |
| 8 | 366.1 | 44.8 / 321.4 | 243.5 | local |
| 16 | 564.3 | 93.2 / 471.1 | 464.4 | local (close) |
| **32** | **473.5** | 188.8 / 284.7 | **929.8** | **remote (2x)** |
| 64 | 692.1 | 385.7 / 306.4 | 1882.9 | remote (2.7x) |
| 128 | 1270.8 | 770.5 / 500.4 | 3685.2 | remote (2.9x) |
| 256 | 2121.1 | 1550.9 / 570.3 | 7524.0 | remote (3.5x) |

**The crossover is real and clean, between batch 16 and 32.** Network
overhead stays roughly flat (~280-570ms, noisy but bounded) regardless
of batch size, while both compute times scale roughly linearly with
batch size — so once amortized compute dominates the fixed network
cost, remote wins, and the margin widens with batch size (up to 3.5x at
256). This is exactly the mechanism Phase 2 predicted but didn't test:
**a single op never justifies remote dispatch; ~32+ batched ops does,
decisively, on this hardware pair.**

The GPU pod was created, benchmarked, and deleted within minutes for
this test — confirmed via a follow-up `list-pods` call, no idle billing.

## Phase 4 (deferred, not blocking)

- **Jetson Orin Nano backend**: the genuinely interesting fourth silicon
  type here — ARM CPU + Tegra iGPU with unified memory, architecturally
  distinct from everything benchmarked above (unlike another RunPod/Vast
  box, which would just be a different GPU tier on the same x86+discrete-
  GPU architecture already covered in Phase 2). Deferred rather than
  swapped for a same-architecture provider, because that would dilute
  the actual multi-*silicon* claim this repo is making. Two real,
  external blockers, neither fixable from code: the pod is WiFi-only
  with no VPN/Tailscale set up, so it's only reachable when a laptop is
  on the same LAN; and even when reachable, no NVIDIA Tegra-CUDA wheel
  is published yet for its JetPack R39 / CUDA 13.2 build, so the Orin
  GPU can't be exercised regardless of connectivity. Revisit once
  Tailscale is installed on the device and/or NVIDIA ships a matching
  wheel.
- **Fused-op kernel benchmark**: retry the kforge experiment on a small
  chain of ops (e.g. matmul → bias-add → GELU) instead of one matmul,
  where kernel fusion actually has something to do.

## Running it

```bash
python3 benchmarks/bench_local.py                 # regenerate results_local.json on your own hardware
python3 router/router.py 128 4096 4096             # local-only routing
python3 router/router.py 128 4096 4096 30          # include remote CUDA w/ 30ms measured RTT

# batched-remote crossover (needs a running batch_server.py on a GPU host):
python3 workers/batch_server.py 8080               # on the remote GPU
python3 benchmarks/bench_batched_remote.py http://<host>:8080/batch   # from your client
```
