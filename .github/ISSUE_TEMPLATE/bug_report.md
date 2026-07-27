---
name: Bug report
about: Something in the agent, eval harness, or tooling is broken
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the bug.

**To reproduce**
The exact command(s), e.g. `uv run python scripts/run_airline_eval.py --limit 2`.

**Expected vs actual**
What you expected, and what happened instead (paste the traceback if there is one).

**Environment**
- OS:
- Python / `uv` version:
- Provider + model (e.g. `ollama qwen2.5:7b`, `openai gpt-4o-mini`):

**Checklist**
- [ ] `uv run pytest` is green on a clean checkout
- [ ] I searched existing issues
