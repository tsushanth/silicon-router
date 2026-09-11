# Contributing

This project's only real rule: **every claim is backed by a number you
can reproduce, or it doesn't go in.** Positive and negative results are
equally welcome — Phase 2 and Phase 3 in the README are both "the
obvious approach lost," and that's treated as a finding, not a failure
to hide.

## Before opening a PR

- If you're adding a backend (a new GPU, a different edge device, a new
  provider), it should implement the same shape as the existing
  benchmark scripts: real measured timings written to a
  `benchmarks/results_*.json` file, nothing simulated or estimated.
- If you're adding a routing decision, show the crossover point you
  measured, not just the backend you'd expect to win — see how Phase 1
  finds the real CPU/MPS crossover instead of assuming "GPU always
  wins."
- If a rented resource (a cloud GPU, etc.) was used to produce your
  numbers, note in the PR that it was torn down after — this project
  has a specific habit of not leaving billed resources running (see
  Phase 2/3 in the README) and PRs should keep that habit.

## What "done" looks like for a change

- The benchmark script that produced your numbers is in the PR, not
  just the numbers — anyone should be able to rerun it.
- The README/ROADMAP is updated in the same PR if your change answers
  (or changes the status of) one of the open milestones in ROADMAP.md.
- No claim in the PR description that isn't backed by a number
  somewhere in the diff.

## Scope

See ROADMAP.md for what's actually being worked toward and what's
explicitly out of scope (this isn't trying to become production
infrastructure or a Gimlet competitor).
