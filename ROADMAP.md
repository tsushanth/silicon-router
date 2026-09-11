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

## Milestone 1: Real workloads, not synthetic matmuls

**Problem**: every result so far is one isolated matmul. That's enough
to find a genuine crossover point, but it's not representative of what
an actual AI workload looks like, and nobody should trust routing
decisions built only on it.

**Done when**: the router makes decisions for an actual small model's
forward pass (e.g. a distilled LLM's prefill vs. decode stages, or a
multi-stage speech pipeline), not just a single op — with the same
measure-first discipline as Phase 1-3 (no simulated numbers, no
retrofitted narrative).

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

## Milestone 3: A real dispatch abstraction

**Problem**: `router.py` is a JSON lookup table today. That's honest
and appropriately simple for what's been tested, but it's not something
a third party could plug their own backend (a different GPU, a TPU, a
different edge device) into.

**Done when**: adding a new backend means implementing a small,
documented interface (something like `benchmark()` + `dispatch()`), not
editing router internals — and Jetson becomes the first real test of
that interface once it's reliably reachable (see README's Phase 4 note
on why it's deferred, not faked).

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

## Explicitly not planned

- Any production-readiness work (auth, multi-tenancy, SLAs) — out of
  scope for what this project is.
- Competing with Gimlet on features. If Milestone 2's answer is "remote
  dispatch basically never wins for workloads this small," that's a
  fine, honest place for this project to land.
