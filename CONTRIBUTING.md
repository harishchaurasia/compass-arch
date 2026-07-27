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

## Good first contributions

Ranked roughly by effort. The first two need only a machine that can run Ollama.

1. **Put a new model on the [leaderboard](LEADERBOARD.md).** Run any suite with
   `--provider ollama --model <id>`, regenerate with
   `uv run python scripts/leaderboard.py`, and PR the updated `LEADERBOARD.md`
   **and** its `results/leaderboard_data.json` snapshot (CI fails if they drift).
   Open an issue with the "Add a model to the leaderboard" template first if you
   want a hand. The raw `trials.db` stays local; only the aggregates are committed.
2. **Gate your own agent and report friction.** `examples/gate_your_agent.py` is the
   whole integration surface (`build_compass_agent` + `run`). If wrapping your tools
   was awkward, that is a docs/API bug worth an issue.
3. **Run the §7 ablation or §8 `T_HIGH` sweep on airline.** Retail and MCP have them;
   airline does not yet. `scripts/run_airline_eval.py` already accepts
   `--no-verification` and `--t-high`, so this is running and writing up, not coding.
4. **Attack the real open problem (§8): an earlier, higher-discrimination signal.**
   Pooled discrimination AUC is ≈ 0.53 - the gate works by abstaining broadly, not by
   ranking good actions above bad. Precondition checks in `trajectory.py` before a
   high-risk action are the sketch. This is the Phase 4 work that would make Compass
   more than a caution switch.

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
