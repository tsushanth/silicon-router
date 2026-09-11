# Roadmap

This started as a 3-phase weekend experiment testing whether the idea
behind [Gimlet Labs](https://gimletlabs.ai/) — routing pieces of an AI
workload to whichever chip runs them fastest — holds up under real
measurement. It did, partially: two of three intuitive assumptions
about hardware routing turned out false when actually timed (see
README). This roadmap is what turning that into a real open-source
project — not a product, not a Gimlet competitor — actually requires.

**Explicit non-goal**: this is not trying to be a commercial or
production alternative to Gimlet's infrastructure. It's an open,
honestly-benchmarked exploration of the same idea, released so the
methodology and findings are reusable by anyone.

## Why open source, why now

No revenue or funding angle here — the value is in the findings being
real, reproducible, and public, plus whatever a wider set of
contributors could add (more hardware, more workloads, more honest
negative results). Apache-2.0 was picked over MIT specifically because
this sits in ML-infra territory where a patent grant is the ecosystem
norm (vLLM, Ray, etc.), not because of any concrete dispute — it's the
standard choice for this category of tool.

## Milestone 1: Real workloads, not synthetic matmuls — DONE

**Problem**: every result so far was one isolated matmul. That's enough
to find a genuine crossover point, but it's not representative of what
an actual AI workload looks like, and nobody should trust routing
decisions built only on it.

**Done** (see Phase 7 in the README): routed a real
`nn.TransformerEncoderLayer` forward pass (Transformer-base config, the
same building block real models stack) instead of a matmul. Local
CPU/MPS found a real, qualitatively-similar crossover to Phase 1's
(small sequences favor CPU, MPS pulls ahead up to 3.6x by seq_len=1024).
Remote lost at every size tested, by 20-600x — extending Phase 2's
single-op finding to a real model: the network round-trip dominates a
single small model layer's actual compute just as badly as it dominated
a single matmul. `router/model_dispatch_router.py`'s
`ModelDispatchRouter` correctly routed local at every tested size in a
live integration test against a real rented GPU.

**Still open, deliberately not attempted here**: this is one
transformer *layer*, not a full model with a real prefill/decode split
(a real LLM's actual routing-relevant structure). A real prefill-vs-
decode experiment — where the two stages have genuinely different
compute/memory profiles, unlike one repeated layer — is the honest next
step if this milestone gets revisited, not something this phase claims
to have already covered.

## Milestone 2: Answer the batching/segment-routing question — DONE

**Problem**: Phase 2 found that a single op is never worth a remote
GPU dispatch once you count network round-trip. That's a real finding,
but it leaves the interesting question unanswered — does routing a
*sequence* of ops per round-trip change that math? This is the actual
technical core of whether "route to remote hardware" is ever a good
idea for real workloads, and right now it's still a guess.

**Answered** (see Phase 5 in the README): yes, and cleanly. Local wins
through batch 16, remote wins from batch 32 onward — 2x at the
crossover, widening to 3.5x by batch 256 — measured with a real HTTP
server on the remote GPU, not an SSH exec. This is now the actual
thesis of the project: routing decisions are batch-size-dependent, not
a fixed "local vs remote" answer, and the crossover point is something
you measure per hardware pair, not assume.

**Still open, not yet done**: the router itself (`router/router.py`)
doesn't use this yet — it still only knows about single-op local
routing plus an opt-in manual network-overhead override. Wiring the
Phase 5 batch-crossover data into an actual routing decision (pick
local vs. "queue N ops for one remote call") is the next real step,
not a new milestone — it's what Milestone 3's dispatch abstraction
should be built around.

## Milestone 3: A real dispatch abstraction — DONE

**Problem**: `router.py` was a JSON lookup table. Honest and
appropriately simple for what had been tested at the time, but not
something a third party could plug their own backend into, and it
didn't use Milestone 2's batching finding at all.

**Done** (see Phase 6 in the README): `backends/base.py` defines a
two-method `Backend` interface (`benchmark_op`, `run_batch`);
`backends/local.py` and `backends/remote_http.py` implement it for
CPU/MPS and a real remote GPU over HTTP; `router/dispatch_router.py`'s
`DispatchRouter` calibrates every registered backend for a given
`(dim, count)` and **actually executes** the winning one via
`dispatch()`, not just reports a decision. Verified with a live
integration test against a real rented GPU — router correctly crossed
over to remote at high batch counts, correctly stayed local at low
ones, and `dispatch()`'s real wall-clock time matched calibration.

Also surfaced something worth keeping in mind: the crossover point
moved between Phase 5's pod and Phase 6's pod (32 vs 64) — different
community-cloud instances, same code. That's not noise to average
away; it's why `calibrate()` is a first-class method instead of a
one-time constant.

**Update**: Jetson became reachable and is now that real third
architecturally-distinct backend — see README's Phase 4. The `Backend`
interface needed exactly one generalization (`LocalBackend` accepting
`"cuda"` as a local device, not just `"cpu"`/`"mps"`) to support it, no
new abstraction required. Also corrected an earlier wrong assumption in
the process: the "no Tegra-CUDA wheel published" blocker from Phase 6
turned out to be stale by the time this was revisited, not permanent —
worth remembering that "checked once, documented as blocked" isn't the
same as "still true later."

**Done**: Jetson registered as a real third `Backend` in one live
`DispatchRouter` session (see README's Phase 4 update). Result was
genuinely non-obvious: Jetson lost at every size tested despite beating
its own CPU by 8-9x — a small edge GPU beating its own weak ARM CPU by
a wide margin doesn't mean it beats a desktop-class GPU in absolute
terms. Confirms the whole point of the `Backend` abstraction: adding a
real architecture doesn't just add "another fast option," it can
change what the router actually decides, for reasons only real
measurement surfaces.

## Milestone 4: Reproducibility and docs good enough for a stranger to trust

**Problem**: right now the credibility of this repo rests on me having
personally run everything and written it up honestly. That doesn't
scale to "someone found this on GitHub."

**Done when**:
- Every benchmark script runs unattended and reproduces its own numbers
  within a documented tolerance, on hardware someone else owns.
- The README's honest-negative-results framing (Phase 2's network
  reality check, Phase 3's kforge loss) is preserved as the project
  grows — that track record is the actual asset here, not something to
  smooth over once this is public-facing.
- A CONTRIBUTING.md exists so someone adding a GPU/backend knows the
  bar: real measured numbers or it doesn't get merged, positive or
  negative results both welcome.

**Progress so far**:
- `tests/` (plain stdlib `unittest`, zero extra deps): unit tests for
  `router/router.py`'s deterministic decision logic
  (`nearest_size_class`, `route()`) and `router/dispatch_router.py`'s
  `DispatchRouter` (`calibrate()`/`route()`/`dispatch()` bookkeeping),
  using a fake `Backend` subclass with synthetic timings instead of
  real hardware — `backends/base.py`'s interface is exactly designed
  for that substitution. 23 tests, no GPU/MPS/torch-device dependency,
  verified passing locally.
- `.github/workflows/ci.yml`: runs the above tests, a `torch` import
  sanity check, and `benchmarks/bench_local.py` itself on the CI
  runner's CPU, on every push/PR. Explicit in its own comments about
  what it can't do — no GPU on standard GitHub-hosted runners, so
  `bench_cuda.py`, the CUDA half of `bench_kforge.py`,
  `bench_batched_remote.py`, and anything touching
  `workers/batch_server.py` stay manually-reproduced only, not
  CI-verified.
- `REPRODUCING.md`: states, phase by phase, what "documented tolerance"
  actually means — qualitative crossovers/winners are the reproducible
  claim, exact millisecond figures and exact crossover points are
  this-machine-and-this-rented-instance-specific (Phase 6's README
  already documents the crossover moving from batch 32 to batch 64
  between two different rented pods — REPRODUCING.md treats that as
  the norm to expect, not an exception).

**Still open**: CONTRIBUTING.md's "real measured numbers or it doesn't
merge" bar is stated but not yet mechanically enforced (e.g. a PR
template checklist or CI check that a PR touching `benchmarks/`
actually includes/updates a `results_*.json`) — that's the remaining
piece before this milestone is fully done.

## Explicitly not planned

- Any production-readiness work (auth, multi-tenancy, SLAs) — out of
  scope for what this project is.
- Competing with Gimlet on features. If Milestone 2's answer is "remote
  dispatch basically never wins for workloads this small," that's a
  fine, honest place for this project to land.
