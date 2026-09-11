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

## Phase 6 (done): a real pluggable router (Milestone 3)

Everything above was a benchmark script that prints a table. This phase
turns it into something that actually routes: `backends/base.py` defines
a two-method interface (`benchmark_op`, `run_batch`) that any backend —
local CPU/MPS (`backends/local.py`), or a remote GPU over real HTTP
(`backends/remote_http.py`) — implements identically, and
`router/dispatch_router.py`'s `DispatchRouter` calibrates all of them at
once, picks the fastest for a given `(dim, count)`, and **actually
executes the real work** on it via `dispatch()` — not just reports a
decision.

Ran a full live integration test — CPU, MPS, and a real rented RTX 3090
over its public HTTP proxy, all three registered with one router:

```
count=   1 -> local:cpu    (84.3ms)
count=   8 -> local:mps    (229.1ms)
count=  16 -> local:mps    (467.4ms)
count=  32 -> local:mps    (915.4ms)
count=  64 -> remote       (832.1ms)   [local:mps was 1859.4ms]
count= 128 -> remote       (1176.3ms)  [local:mps was 3829.6ms]
```

`dispatch(4096, 128)` genuinely ran on the remote backend (1.7s real
wall time, no tensor shipped back — see `remote_http.py`'s note on why);
`dispatch(4096, 1)` genuinely ran locally and returned a real computed
4096×4096 tensor. Both confirmed by executing them, not by reading the
routing decision.

**Honest note on the crossover point**: this run's local/remote
crossover landed at count=64, not count=32 like Phase 5's number. Same
router, same code, different rented instance (a weaker, noisier
community-cloud box this time). That's not a bug — it's confirmation of
the actual thesis: **the crossover is real, but it's a property of the
specific hardware pair you measure, not a constant this project can
hardcode.** `DispatchRouter.calibrate()` exists specifically so nobody
has to trust last week's number for today's hardware.

Pod created, integration-tested, and deleted within minutes — confirmed
via `list-pods`.

## Phase 7 (done): a real model forward pass, not a matmul (Milestone 1)

Every phase above routes one isolated matmul. This one routes a real
transformer encoder block — `models/tiny_transformer.py` wraps PyTorch's
own `nn.TransformerEncoderLayer` at the original "Transformer base"
config (Vaswani et al. 2017: d_model=512, 8 heads, dim_feedforward=2048)
— a real multi-op forward pass (self-attention + MLP + layernorms +
residuals), not a stand-in shape.

**Local (M2 Pro), batch=4, properly warmed up (`bench_model_local.py`):**

| seq_len | CPU (ms) | MPS (ms) | Winner |
|---|---|---|---|
| 16 | 1.15 | 1.21 | CPU (close) |
| 64 | 2.29 | 1.93 | MPS |
| 256 | 7.05 | 2.83 | **MPS (2.5x)** |
| 1024 | 47.38 | 13.30 | **MPS (3.6x)** |

A real crossover again, this time on an actual model layer instead of a
raw matmul — small sequences favor CPU, MPS pulls ahead and widens as
sequence length grows, the same qualitative shape as Phase 1 but on
different, more realistic ops.

**Remote (RTX 3090, live public internet), same workload
(`bench_model_remote.py`):**

| seq_len | Remote wall (ms) |
|---|---|
| 16 | 817.2 |
| 64 | 925.0 |
| 256 | 442.3 |
| 1024 | 444.6 |

**Local wins at every size, by 20-600x.** Unlike the big-matmul batching
experiment (Phase 5), this single model layer's actual compute is tiny
(tens of milliseconds even at seq_len=1024) next to the network
round-trip (Phase 5 measured ~280-570ms of pure network overhead on
this same kind of connection) — so network noise dominates completely
and doesn't even resolve a clean seq_len trend. This extends Phase 2's
finding (a single op never justifies remote dispatch) to a real model:
**a single real-model forward pass doesn't either, for the same
reason** — and Phase 5/6 already showed what changes that: batching
many such calls into one remote round-trip, not sending them one at a
time.

`router/model_dispatch_router.py`'s `ModelDispatchRouter` — same
calibrate/route/dispatch pattern as `DispatchRouter`, applied to
`run_model_batch` instead of `run_batch` — correctly picked local at
every size when tested live against all three backends, matching the
table above. One honest wrinkle: `calibrate()` takes a single real
measurement per backend (matching `DispatchRouter`'s existing design),
so it can catch first-call overhead (lazy model init, cuDNN algorithm
search) that the separately-averaged benchmark scripts warm past — a
real, worth-knowing difference between "the router's own live
calibration" and "a clean benchmark script's numbers," not a bug in
either.

Pod created, tested against both the raw-matmul and model-forward
endpoints, and deleted within minutes — confirmed via `list-pods`.

## Phase 4 (resolved): Jetson Orin Nano — the real fourth silicon type

This was deferred for two reasons, both eventually resolved rather than
worked around: the device is WiFi-only with no VPN/Tailscale, so it's
only reachable when on the same LAN; and no NVIDIA Tegra-CUDA wheel was
published for its JetPack R39 / CUDA 13.2 build *at the time of Phase
6* — checked directly against NVIDIA's Jetson package index, not
assumed. That assumption turned out to be **wrong** by the time this was
revisited: a generic `pip3 install torch` (2.14.0+cu130) now pulls real
CUDA 13 packages that work on the Orin's iGPU. Worth stating plainly —
the earlier "deferred, no wheel available" note was correct when
written and stale by the time it mattered; checked again rather than
trusted.

**Correctness verified before trusting any speed number**: PyTorch
itself warns this build has no kernels compiled for Orin's compute
capability (8.7) and falls back to PTX JIT compilation. That's a real
"did it silently produce garbage" risk, not paranoia — checked directly
by comparing a CUDA matmul against the identical op on CPU: max
absolute difference **5.0e-5**, consistent with ordinary float32
accumulation-order variance between backends, not a correctness bug.

**Real measured result — GPU wins at every size, on both workloads**,
unlike Mac's MPS which lost badly at small sizes:

Matmul (`backends/local.py`'s `LocalBackend("cuda")`, now genuinely
usable for a local Tegra device, not just a discrete GPU):

| Size class | CPU (ms) | CUDA/Orin (ms) | Speedup |
|---|---|---|---|
| tiny | 0.7 | 0.2 | 3.4x |
| small | 24.2 | 2.8 | 8.5x |
| medium | 527.4 | 60.7 | 8.7x |
| large | 7562.6 | 861.8 | 8.8x |
| xlarge | 115930.4 | 13268.3 | 8.7x |

Real transformer block (same `TinyTransformerBlock` as Phase 7):

| seq_len | CPU (ms) | CUDA/Orin (ms) | Speedup |
|---|---|---|---|
| 16 | 9.8 | 1.1 | 9.3x |
| 64 | 28.3 | 2.0 | 13.9x |
| 256 | 70.1 | 7.1 | 9.9x |
| 1024 | 358.4 | 32.5 | 11.0x |

**Why GPU wins even at tiny sizes here, when MPS didn't on Mac**: Orin's
CPU and GPU share the same physical memory (true unified memory, not
just a unified *address space* like Apple Silicon) — there's no
PCIe-style transfer or the dispatch overhead that made MPS lose to CPU
below Phase 1's crossover point. A genuinely different architecture
produced a genuinely different routing story, which is the entire
reason this project wanted a real fourth silicon type instead of
another x86 GPU tier.

Also worth being honest about: the CPU numbers here are much slower per
element than the Mac's CPU numbers in Phase 1 — this is a 6-core ARM
CPU on generic OpenMP/oneDNN, not x86 with MKL, and that's a real,
expected difference in BLAS optimization maturity between platforms,
not a bug or a rigged comparison.

**Update — done**: registered as a real third `Backend` in one live
`DispatchRouter` session alongside `local:cpu` and `local:mps`, via
`RemoteHTTPBackend` pointed at `workers/batch_server.py` running on the
Jetson over the LAN (the same generic HTTP mechanism as the RunPod
backend — a same-network host is just another remote backend, no
special-casing needed):

| count | local:cpu | local:mps | Jetson (LAN HTTP) | Winner |
|---|---|---|---|---|
| 1 | 236.8ms | 319.8ms | 309.5ms | local:cpu |
| 8 | 589.1ms | 264.0ms | 844.0ms | local:mps |
| 32 | 2104.2ms | 977.0ms | 3332.6ms | local:mps |
| 128 | 8551.9ms | 3678.9ms | 13253.9ms | local:mps |

**Jetson loses at every size here — a genuinely important, non-obvious
result.** Its LAN-measured time (13253.9ms at count=128) is nearly
identical to its own standalone number from earlier in this section
(13268.3ms), confirming the network adds negligible overhead on a LAN —
this is a pure compute-power gap, not a connectivity artifact. Orin's
GPU beating its own CPU by 8-9x (above) does **not** mean it beats a
desktop-class GPU in absolute terms — those are different questions
entirely, and it's a real trap to conflate "beats its own weaker
sibling by a wide margin" with "wins the race." The Orin Nano is a
small edge chip; M2 Pro's MPS is not. `dispatch()` confirmed this live:
routed to `local:cpu` at count=1 and `local:mps` at count=128, matching
the table, both executed for real (89.2ms and 3767.0ms measured).

This is the honest payoff of building a real interface instead of
hardcoding a "GPU wins" assumption anywhere: adding a real fourth
architecture didn't just slot in as "another fast option" — it revealed
that per-workload absolute measurement matters more than which chip
*sounds* more capable.

## Phase 4b (not attempted)

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

# the real router - calibrates for real, then actually dispatches:
python3 - <<'PY'
from backends.local import LocalBackend
from backends.remote_http import RemoteHTTPBackend
from router.dispatch_router import DispatchRouter

r = DispatchRouter([LocalBackend("cpu"), LocalBackend("mps"), RemoteHTTPBackend("http://<host>:8080/batch")])
r.calibrate(dim=4096, counts=[1, 8, 32, 128])
decision, wall, result = r.dispatch(4096, 128)
print(decision["backend"], wall)
PY
```

## Testing and reproducing (Milestone 4)

```bash
python3 -m unittest discover tests/ -v   # deterministic router/dispatch logic, no GPU needed
```

`tests/` covers the non-benchmark logic in `router/router.py` and
`router/dispatch_router.py` (routing decisions, calibration
bookkeeping) against fixed fixtures and a fake `Backend`, so it's fast
and needs no hardware. `.github/workflows/ci.yml` runs these plus
`benchmarks/bench_local.py` on every push/PR — CPU-only, since
GitHub-hosted runners have no GPU. See
[REPRODUCING.md](REPRODUCING.md) for exactly what "documented
tolerance" means per phase before assuming a rerun that doesn't match
the tables above is a bug — most of them aren't meant to reproduce
exact millisecond figures, only the qualitative winner.
