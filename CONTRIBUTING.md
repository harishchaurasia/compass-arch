# Contributing to Compass

Thanks for your interest. Compass is an open, honestly-reported experiment in
training-free agent calibration - contributions and PRs are welcome, especially
around new models, new task suites, and the open questions in
[FINDINGS.md](FINDINGS.md).

## Setup

```bash
uv sync                 # install / refresh dependencies (editable install)
uv run pytest           # run the full suite (no API keys needed - tests use fakes)
uv run ruff check .     # lint
```

Python 3.11+. No build step; `compass` is a pure editable package.

## Before opening a PR

1. `uv run pytest` is green.
2. `uv run ruff check .` is clean.
3. New behaviour has a test. New results have a way to reproduce them.

## Ground rules that keep results honest

These come from [DESIGN.md](DESIGN.md) and [CLAUDE.md](CLAUDE.md) - they exist so
the numbers stay trustworthy:

- **The aggregator and its thresholds are locked.** `T_MED` / `T_HIGH` in
  `policy.py` and the rules in `calibration.py` are fixed on a 5-task dev split
  before evaluation. Do not tune them against the eval set. New calibration ideas
  belong behind a flag (see how `--calibration shrinkage` is stored under a
  separate `model="...-shrink"` label so it never mixes with the baseline).
- **Never commit `results/trials.db`.** It is gitignored. Raw trial data stays
  local; back it up before any mutation.
- **Report negative results.** If a change doesn't beat the baseline, that's a
  finding, not a failure - write it up in FINDINGS.md.

## Adding a model

`compass/models.py` documents the split between `EVALUATED_MODELS` (has a full
115-task A/B) and `DEFAULT_MODELS` (per-provider fallback ids). A model only
moves into the "evaluated" story once it has a full vanilla + compass run and the
figures/README are regenerated from real data.
