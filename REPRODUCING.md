# Reproducing this repo's numbers

This project's whole credibility rests on "every number is real and
measured" (see README, CONTRIBUTING.md). That claim only means anything
if it's specific about what "reproducing" a number actually requires —
otherwise it's just a slogan. This doc says, phase by phase, what will
hold up if you rerun the scripts yourself, and what won't, and why.

The short version: **qualitative findings (which backend wins, and
roughly by how much) are the reproducible claim. Exact millisecond
figures are this-machine-specific and were never meant to be universal
constants.** If you rerun a script and get different numbers than the
README, that's expected, not a bug — as long as the *winner* and
rough *margin* match.

## What CI does and doesn't check

`.github/workflows/ci.yml` runs on every push/PR, on a standard
GitHub-hosted `ubuntu-latest` runner — CPU only, no MPS, no CUDA. It:

- runs `tests/` (stdlib unittest, no GPU needed — see below),
- confirms `torch` imports,
- runs `benchmarks/bench_local.py` end-to-end on the runner's CPU.

It **cannot** run anything that needs a GPU: `bench_cuda.py`,
`bench_kforge.py`'s CUDA half, `bench_batched_remote.py`, or anything
that talks to `workers/batch_server.py`. Those stay manually-reproduced
only — this repo is not claiming CI-verified GPU numbers, and the
workflow's own comments say so rather than silently skipping and
implying coverage that isn't there.

## Phase 1 — local CPU vs MPS (`benchmarks/bench_local.py`)

- **What should reproduce**: the qualitative crossover — small ops
  favor CPU (MPS dispatch overhead dominates), large ops favor MPS
  (parallelism wins). CI verifies the script *runs*, on CPU only.
- **What won't reproduce**: the exact ms figures and the exact size
  class where the crossover happens. The README's table is one M2 Pro.
  A different Apple Silicon chip (M1 vs M3, different core counts,
  thermal state, other load on the machine) will shift both the
  absolute numbers and plausibly the size class at which MPS starts
  winning. An Intel Mac has no MPS at all — `bench_local.py` already
  handles that (`torch.backends.mps.is_available()` gates whether MPS
  is even benchmarked), and so does the CI runner (no MPS device by
  definition).
- **Tolerance**: no numeric tolerance is claimed. The reproducible
  claim is directional: CPU wins below some crossover, MPS wins above
  it, by a factor of a few x once it does. Treat any single ms value
  in the README as "what one specific machine measured on one specific
  day," not a constant.

## Phase 2 — remote CUDA as a third backend (`benchmarks/bench_cuda.py`)

- **What should reproduce**: raw remote CUDA compute beats local
  compute at every size (RTX 3090 vs M2 Pro CPU/MPS) — but that's not
  the actual finding. The real claim is that once a real network
  round-trip is added, local MPS beats remote-over-network at every
  size tested, because the network hop dominates. That qualitative
  result — "a single op is never worth a remote dispatch" — is the
  reproducible part.
- **What won't reproduce**: the exact compute numbers (different GPU
  tier, different RunPod host), and definitely not the network overhead
  number. `router.py` deliberately requires the caller to pass a real
  measured `network_overhead_ms` rather than hardcoding one — see its
  docstring. Whatever RTT you measure to whatever community-cloud
  instance RunPod happens to assign you will differ from the README's
  ~30ms figure; that's expected, not a discrepancy to chase down.
- **Requires**: a rented GPU (RunPod or similar) — cannot run in CI.
  Not free to reproduce; budget for a few minutes of GPU rental and
  tear the pod down immediately after (see CONTRIBUTING.md's habit
  around this).

## Phase 3 — kforge-style autotuning (`benchmarks/bench_kforge.py`)

- **What should reproduce**: `torch.compile(mode="max-autotune")`
  loses to eager mode on a single plain matmul, on both CPU and CUDA,
  and the gap widens with size on CUDA. This is a negative result and
  the README already frames it as one — REPRODUCING.md's job is just
  to say it should still be a negative result if you rerun it, not to
  soften it.
- **What won't reproduce**: exact speedup ratios (they depend on the
  installed torch/Triton/Inductor version — autotune kernel selection
  is a moving target across releases — and on the specific GPU/CPU).
  If a future torch version closes or reverses this gap, that's a
  legitimately interesting update to the README, not a reproduction
  failure of the old numbers.
- **Requires**: CPU half runs anywhere (CI can technically run this
  script, though it isn't wired into ci.yml, since it's slow and not
  part of Milestone 4's core ask); CUDA half needs a rented GPU.

## Phase 5 — batched remote dispatch (`benchmarks/bench_batched_remote.py`)

- **What should reproduce**: a real crossover exists somewhere in the
  batch-size range tested — small batches lose to local, larger
  batches win remotely because compute increasingly amortizes the
  fixed network overhead. That the crossover *exists and moves the
  verdict* is the finding.
- **What won't reproduce**: the exact crossover batch size (README:
  "between batch 16 and 32" on one specific pod) or the exact speedup
  at any given batch size. Phase 6 already found this moves between
  runs on different rented instances (32 vs 64) — see below. Treat the
  crossover as "somewhere in the low tens of ops for this hardware
  pair," not a fixed number.
- **Requires**: a rented GPU running `workers/batch_server.py` over a
  real public HTTP path — cannot run in CI, and isn't a quick local
  rerun either (needs provisioning + teardown discipline).

## Phase 6 — the real dispatch router (`router/dispatch_router.py`,
`backends/`)

- **What should reproduce**: `DispatchRouter.calibrate()` produces
  *some* crossover between local and remote as batch count grows, and
  `dispatch()`'s actual executed wall time is close to what
  `calibrate()` predicted for that backend (same code path, same run).
- **What won't reproduce, by design, not by accident**: the exact
  crossover count. The README documents this directly — Phase 5's pod
  crossed over at batch 32, Phase 6's pod (a different rented instance,
  same code) crossed over at batch 64. This is the single clearest
  example in the repo of "don't expect exact numbers to reproduce" —
  it's why `calibrate()` is a method you call, not a constant baked
  into the router. If you rerun this against your own rented GPU and
  get yet another crossover count, that's consistent with the project's
  own finding, not a contradiction of it.
- **Requires**: real hardware for the remote side. The local-only
  logic (which backend wins given already-calibrated numbers,
  nearest-count snapping, that `route()` doesn't execute anything and
  `dispatch()` only executes the winner) is covered by `tests/` and
  needs no GPU — see below.

## What `tests/` actually covers (no tolerance issues at all)

`tests/test_router.py` and `tests/test_dispatch_router.py` are
deliberately **not** benchmark reproductions. They test the
deterministic decision logic — `nearest_size_class()`, `route()`'s
comparison/exclusion logic, `DispatchRouter.calibrate()`/`route()`/
`dispatch()`'s bookkeeping — against small hand-written fixture tables
or a fake `Backend` subclass with synthetic, fixed timings (see
`backends/base.py`'s interface, which is exactly shaped for this kind
of substitution). These have **zero** hardware dependency and should
produce byte-identical results on any machine, forever — that's the
whole point of separating "logic that's deterministic" from "numbers
that come from real hardware and vary by machine." Run them with:

```
python3 -m unittest discover tests/ -v
```

## Summary table

| Phase | Reproducible claim | Not reproducible | Needs GPU? | In CI? |
|---|---|---|---|---|
| 1 (local CPU/MPS) | qualitative crossover exists | exact ms, exact crossover size class | no (MPS optional) | yes (CPU only) |
| 2 (remote CUDA) | network RTT beats raw compute advantage | exact ms, exact RTT | yes | no |
| 3 (kforge) | autotune loses to eager here | exact speedup ratios | CPU: no: CUDA: yes | no |
| 5 (batched remote) | a batch-size crossover exists | exact crossover batch, exact speedup | yes | no |
| 6 (dispatch router) | calibrate()/dispatch() are consistent | exact crossover count (documented to move: 32 vs 64) | yes | no |
| router/dispatch_router decision logic (`tests/`) | exact, deterministic | n/a — no hardware numbers involved | no | yes |
