# silicon-router

A small, honest proof-of-concept exploring the idea behind [Gimlet Labs](https://gimletlabs.ai/) —
routing pieces of an AI workload to whichever chip actually runs them fastest,
instead of pinning everything to one device.

This is a learning/exploration project, not a product. Every number in this
repo is measured on real hardware, not simulated.

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

## Phase 4 (not started)

- **Jetson Orin Nano backend**: third architecture (ARM + CUDA), currently
  unreachable on the network — add once it's back online.
- **Batched remote dispatch**: route a whole *sequence* of ops to the
  remote GPU per round-trip instead of one op at a time, and measure
  whether that's actually where remote wins.
- **Fused-op kernel benchmark**: retry the kforge experiment on a small
  chain of ops (e.g. matmul → bias-add → GELU) instead of one matmul,
  where kernel fusion actually has something to do.

## Running it

```bash
python3 benchmarks/bench_local.py                 # regenerate results_local.json on your own hardware
python3 router/router.py 128 4096 4096             # local-only routing
python3 router/router.py 128 4096 4096 30          # include remote CUDA w/ 30ms measured RTT
```
