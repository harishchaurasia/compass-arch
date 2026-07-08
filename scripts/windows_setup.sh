#!/usr/bin/env bash
# One-shot Windows setup — run this from GIT BASH in the repo folder:
#
#   ./scripts/windows_setup.sh
#
# Installs uv and Ollama if missing, creates the venv (uv sync), pulls the
# local models, and runs the 3-task smoke test. When it finishes cleanly,
# start the real runs with:  ./scripts/run_local_gpu.sh all
set -u
cd "$(dirname "$0")/.."

say() { echo; echo "== [windows_setup] $*"; }

# ── uv ────────────────────────────────────────────────────────────────────────
# uv installs to %USERPROFILE%\.local\bin — put that on PATH for this session
# too, so a terminal restart isn't needed.
UV_BIN="$(cygpath "${USERPROFILE:-$HOME}")/.local/bin"
export PATH="$UV_BIN:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    say "uv not found — installing via PowerShell ..."
    powershell.exe -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" || {
        echo "uv install failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
    export PATH="$UV_BIN:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv still not on PATH — close ALL terminals, reopen Git Bash, rerun this script."; exit 1; }
say "uv $(uv --version 2>/dev/null || echo '?') ready"

# ── Ollama ────────────────────────────────────────────────────────────────────
# Default install location on Windows; add for this session.
OLLAMA_BIN="$(cygpath "${LOCALAPPDATA:-$USERPROFILE/AppData/Local}")/Programs/Ollama"
export PATH="$OLLAMA_BIN:$PATH"

if ! command -v ollama >/dev/null 2>&1; then
    say "Ollama not found — installing via winget (accept any prompts) ..."
    if command -v winget.exe >/dev/null 2>&1; then
        winget.exe install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements || true
        export PATH="$OLLAMA_BIN:$PATH"
    fi
fi
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not installed / not on PATH. Install from https://ollama.com/download,"
    echo "launch it once from the Start menu, then rerun this script."
    exit 1
fi

# the Windows Ollama app runs a tray service; make sure it's up
if ! ollama list >/dev/null 2>&1; then
    say "starting Ollama server ..."
    (ollama serve >/dev/null 2>&1 &)
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 2
        ollama list >/dev/null 2>&1 && break
    done
fi
ollama list >/dev/null 2>&1 || { echo "Ollama server won't start — launch 'Ollama' from the Start menu, then rerun."; exit 1; }
say "Ollama ready"

# ── venv + dependencies ───────────────────────────────────────────────────────
say "creating venv + installing dependencies (uv sync) ..."
uv sync || exit 1

# ── models ────────────────────────────────────────────────────────────────────
for model in qwen2.5:7b llama3.1:8b; do
    if ! ollama list | awk '{print $1}' | grep -qx "$model"; then
        say "pulling $model (few GB — one-time) ..."
        ollama pull "$model" || exit 1
    fi
done

# ── smoke test ────────────────────────────────────────────────────────────────
say "setup complete — running the 3-task smoke test (~5-10 min on first load) ..."
./scripts/run_local_gpu.sh smoke

say "DONE. If the smoke lines look healthy (RUNBOOK.md §2), start the full runs:"
echo "    ./scripts/run_local_gpu.sh all"
