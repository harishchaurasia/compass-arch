# Runbook — local GPU eval runs (RTX 5080)

Step-by-step for running the τ-bench suite against local models via Ollama.
The gpt-4o-mini data already lives in the main `results/trials.db`; these runs
add `qwen2.5:7b` and (optionally) `llama3.1:8b` rows on your PC.

**TL;DR — one command does everything below:**

```bash
./scripts/run_local_gpu.sh all     # smoke → full qwen → full llama, logs + exports
```

(`smoke`, `qwen`, or `llama` run the individual stages.) It appends to
`results/run_local.log`, verifies completeness after each run, and dumps each
model's rows to `results/export_<model>.json` to send back for analysis. The
manual steps below explain what it does and how to intervene.

## 1. One-time setup

**Windows note:** install `uv` from PowerShell first (it isn't preinstalled
anywhere):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

then **close and reopen the terminal** so `uv` lands on PATH. Run the `.sh`
script from **Git Bash** (installed with Git for Windows — right-click →
"Open Git Bash here"), not from PowerShell/cmd. Ollama itself installs
natively on Windows.

```bash
git clone https://github.com/harishchaurasia/compass-arch.git && cd compass-arch
uv sync                        # installs everything incl. langchain-ollama

# install Ollama (https://ollama.com/download), then:
ollama pull qwen2.5:7b         # ~4.7 GB
ollama pull llama3.1:8b        # optional second model, ~4.9 GB
```

No API keys needed for local runs. `results/trials.db` starts empty on a fresh
clone — that's fine, runs append to it and we merge later (step 5).

## 2. Smoke test (do this first — ~5 minutes)

```bash
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:7b --limit 3 2>&1 | tee -a results/run.log
```

**Healthy output** looks like the gpt-4o-mini runs: one line per trial,
`✓`/`✗`, `steps=N`, compass lines carry `conf=[...]`. **Unhealthy:** repeated
`ERROR: ...` on every compass trial usually means the model can't drive
structured function-calling — stop and ping Claude; the fallback is
llama3.1:8b (DESIGN.md names it) rather than debugging a 7B's tool syntax.

A few `ERROR:` lines on isolated tasks are normal-ish (the resume flags exist
exactly for re-running those — see step 4).

## 3. Full run (overnight)

```bash
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:7b 2>&1 | tee -a results/run.log
```

- Watch progress from another terminal: `tail -f results/run.log`
- 230 trials; local 7B is slower per token than the API — expect several
  hours. Disable sleep/hibernate before walking away.
- Electricity: a few hours at ~0.5 kW ≈ pennies.

Repeat with `--model llama3.1:8b` for the second model.

## 4. If the run dies partway

Find what's missing and resume only those cells (same pattern as the
gpt-4o-mini resume):

```bash
uv run python scripts/export_trials.py --model qwen2.5:7b   # prints per-condition counts
# then e.g.:
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:7b \
  --task-ids tau_retail_090 tau_retail_091 --conditions compass 2>&1 | tee -a results/run.log
```

## 5. Getting results back

`results/trials.db` is gitignored (by design — raw trial data never gets
committed). The driver script already verifies counts and writes
`results/export_<model>.json` after each run; to redo it manually:

```bash
uv run python scripts/export_trials.py --model qwen2.5:7b
```

Expect 115 distinct tasks per condition per model. Then send the export
JSONs (or the whole `.db` file) back to the Mac — Claude merges the
new-model rows into the main `results/trials.db` there and refreshes the
analysis + charts.
