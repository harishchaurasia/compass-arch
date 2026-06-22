# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session start checklist

1. **Read `DESIGN.md` fully** — it is the canonical spec. Architecture, experimental design, and open questions all live there.
2. **Read `.claude/skills/engineering-coach/Skill.md`** — the coaching skill is active for this project. Use it before every substantive engineering decision.
3. **Check `results/trials.db` schema** (if it exists) before designing or running new experiments — it tells you what has already been run.

## Commands

```bash
uv sync                         # install / refresh dependencies
uv run pytest                   # run all tests
uv run pytest tests/path/to_test.py::test_name  # single test
uv run ruff check .             # lint
uv run ruff format .            # format
```

There is no build step — `compass` is a pure Python package installed in editable mode via `uv sync`.

## Architecture

Compass is a calibration layer that wraps a standard ReAct agent. There are two agent implementations that share tooling but diverge in the step loop:

**`compass/agent_vanilla.py`** — baseline ReAct with `VanillaState`. Thought → action → observation loop, hard step budget, no confidence or abstention.

**`compass/agent_compass.py`** — the calibrated agent. At every step the model must emit a `CompassStep` (Pydantic): `reasoning`, `action`, `confidence ∈ [0,1]`, `risk_level ∈ {low, medium, high}`. That struct feeds a three-stage pipeline:

```
CompassStep
  → trajectory.py   (extract TrajectoryFeatures from the running steps list)
  → calibration.py  (aggregate verbalized confidence + trajectory features → success_prob float)
  → policy.py       (decide: EXECUTE / SELF_VERIFY / ABSTAIN based on success_prob × risk_level)
```

`T_MED` and `T_HIGH` in `policy.py` are the tunable thresholds. They are locked on a 5-task dev split before eval — do not tune them mid-experiment.

**`compass/models.py`** — `get_model(provider, model_name)` wraps `init_chat_model` for cross-provider access. `SUPPORTED_MODELS` maps the four experimental models (Anthropic, OpenAI, Google, Ollama/Qwen).

**`eval/`** — two runners (`tau_bench_runner.py`, `mcp_runner.py`) write one row per trial to `results/trials.db` (SQLite). `metrics.py` computes ECE, Brier score, selective success rate, and compound failure rate from those rows.

**`tasks/`** — task definitions live here: `tau_bench/` for the 40-task academic subset, `custom_mcp/` for the ~12 cascading-failure MCP tasks (designed in Phase 3).

## Key design constraints to honour

- **No custom tracing tool** — roll JSON trace format from scratch. No LangSmith in eval paths (dev debugging only).
- **Aggregator is rule-based and locked** — do not switch to a learned probe without flagging it as a Phase 4 stretch.
- **`results/trials.db`** is gitignored. Never commit raw trial data.
- **Phase 1 goal is end-to-end function**, not clean architecture. Ship a working primitive first.
