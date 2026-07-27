---
name: Add a model to the leaderboard
about: You ran a model on one or more suites and want it on the board
title: "[leaderboard] add <model>"
labels: leaderboard, good first issue
---

**Model**
Provider + id (e.g. `ollama mistral-small`, `openai gpt-4o`).

**Suites you ran**
- [ ] τ-bench retail (115)
- [ ] τ-bench airline (50)
- [ ] MCP filesystem (31)

**Numbers (from your run)**
Paste the SUMMARY block each runner prints, or the relevant `LEADERBOARD.md` rows.

**How to turn this into a PR**
1. Run the suite(s) (see [README → Reproduce](../../README.md#reproduce)).
2. `uv run python scripts/leaderboard.py` to regenerate `LEADERBOARD.md`.
3. Open a PR with the updated `LEADERBOARD.md`. The raw `results/trials.db` stays
   on your machine (it is gitignored); only the distilled board is committed.

Interesting submissions: a model whose confidence actually *discriminates* (that is
the open problem in [FINDINGS.md](../../FINDINGS.md) §8), or a failure mode we have
not seen yet.
