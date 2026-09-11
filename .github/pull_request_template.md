<!--
This template mirrors CONTRIBUTING.md's one real rule: every claim is
backed by a number you can reproduce, or it doesn't go in. Fill this in
honestly - "N/A, this PR only touches tests/docs" is a fine answer when
it's true.
-->

## What this PR does

## Numbers

- [ ] This PR adds/changes a benchmark claim, and the script that
      produced the number(s) is included in this diff (not just the
      number) — see CONTRIBUTING.md.
- [ ] If a rented resource (cloud GPU, etc.) was used, it was torn down
      after — say so below.
- [ ] N/A — this PR doesn't add or change any measured claim (e.g.
      docs, tests, CI, refactor with no new numbers).

## ROADMAP.md / README.md

- [ ] This PR answers or changes the status of an open ROADMAP.md
      milestone, and ROADMAP.md/README.md are updated in this same PR.
- [ ] N/A — no milestone status changed.

## CI

- [ ] `python3 -m unittest discover tests/ -v` passes locally.
- [ ] If this PR touches `router/` or `backends/` logic, tests in
      `tests/` were added or updated to cover it.
