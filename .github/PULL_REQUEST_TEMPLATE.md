<!-- Thanks for contributing to Compass. Keep results honest; that is the whole point. -->

## What this changes
A short description. Link the issue it closes (e.g. `Closes #12`).

## Type
- [ ] Bug fix
- [ ] New model / suite result (regenerated `LEADERBOARD.md`)
- [ ] New feature or research idea
- [ ] Docs

## Checklist
- [ ] `uv run pytest` is green
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] No em dashes, no committed `results/trials.db` (it is gitignored)
- [ ] If I changed the gate/aggregator, I did **not** tune thresholds on the eval set,
      and I updated FINDINGS/README if a reported number moved
- [ ] If I added leaderboard numbers, I regenerated `LEADERBOARD.md` with
      `uv run python scripts/leaderboard.py` (raw DB stays local)

## Numbers moved? (if applicable)
Which reported metrics changed, and where the new figures come from.
