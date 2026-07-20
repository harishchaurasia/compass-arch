# Runbook — local GPU eval runs (RTX 5080)

Step-by-step for running the τ-bench suite against local models via Ollama.
The gpt-4o-mini data already lives in the main `results/trials.db`; these runs
add `qwen2.5:7b` and (optionally) `llama3.1:8b` rows on your PC.

**TL;DR — one command does everything below:**

```bash
./scripts/run_local_gpu.sh all     # smoke → qwen2.5:7b → qwen2.5:14b → llama3.1:8b
```

(`smoke [model]`, `qwen`, `qwen14`, or `llama` run the individual stages;
`smoke` defaults to qwen2.5:7b but takes an optional model, e.g.
`smoke qwen2.5:14b`.) It appends to
`results/run_local.log`, verifies completeness after each run, and dumps each
model's rows to `results/export_<model>.json` to send back for analysis. The
manual steps below explain what it does and how to intervene.

## 1. One-time setup

**Windows:** one script does all of this section — open **Git Bash** in the
repo folder (right-click → "Open Git Bash here") and run:

```bash
./scripts/windows_setup.sh
```

It installs uv and Ollama if missing, creates the venv, pulls both models,
and finishes with the smoke test. Everything below is the manual/macOS/Linux
equivalent. (All later commands on Windows also go in Git Bash, not
PowerShell/cmd.)

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
`✓`/`✗`, `steps=N`, compass lines carry `conf=[...]`. Ollama models are driven
via `json_schema` structured output with a content-salvage fallback, and tool
errors (bad args, unknown/None tool) are fed back to the model instead of
aborting — so `ERROR:` lines should now be **rare**. If you still see repeated
`ERROR:` on every compass trial, the structured-output plumbing has regressed —
ping Claude.

Expect lots of `✗` and `[ABSTAINED]`: small local models are weak and badly
**overconfident** (verbalized confidence sits near 1.0), so low task success is
the model ceiling, not a harness bug — it's exactly the miscalibration Compass
measures. A few `ERROR:` lines on isolated tasks are still normal-ish (the
resume flags exist for re-running those — see step 4).

## 3. Full run (overnight)

Prefer the driver, which pulls the model, runs the suite, and exports results:

```bash
./scripts/run_local_gpu.sh qwen14    # qwen2.5:14b — best-calibrated local model
```

`qwen2.5:14b` (~9 GB) fits a 16 GB GPU and is the recommended primary local
model; `qwen` (7b) and `llama` are the smaller alternatives. The raw command is
still available if you want it:

```bash
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:14b 2>&1 | tee -a results/run.log
```

- Watch progress from another terminal: `tail -f results/run_local.log`
- 230 trials; local models are slower per token than the API — expect several
  hours. Disable sleep/hibernate before walking away.
- Electricity: a few hours at ~0.5 kW ≈ pennies.

Repeat with `qwen` / `llama` (or `all` for every model) for more rows.

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

## 6. MCP suites (Phase 3)

The custom filesystem MCP suite grades on a purpose-built, resettable server so
results stay deterministic:

```bash
uv run python scripts/run_mcp_eval.py --provider ollama --model qwen2.5:14b
# gpt-4o-mini (most informative — mutates aggressively) needs an OpenAI key:
uv run python scripts/run_mcp_eval.py --provider openai --model gpt-4o-mini
```

The *same* Compass bridge (`compass.mcp.bridge.MCPToolServer`) also drives real,
off-the-shelf MCP servers — it just needs Node for `npx`:

```bash
# official filesystem MCP over a scratch dir (real stdio server)
uv run python scripts/mcp_real_servers.py
# add the GitHub MCP server (needs a token)
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...   # a throwaway/scoped token
uv run python scripts/mcp_real_servers.py --github
```

(Or drop the keys into a gitignored `.env` - see `.env.example` - and the scripts
load them automatically.) Note: `@modelcontextprotocol/server-github` is npm-deprecated
in favour of GitHub's official Go server; the demo still runs and lists 26 tools, but
that name may stop resolving eventually.

`mcp_real_servers.py` lists each server's tools, risk-classes them by name
(write / delete / move / edit / close / merge → high), and hands them to
`build_compass_agent` unchanged — so Compass gates destructive actions on real
servers the same way it does in the graded suites. The graded numbers stay on the
purpose-built server; the real servers are the integration proof.
