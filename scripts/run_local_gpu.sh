#!/usr/bin/env bash
# Driver for the local GPU eval runs (see RUNBOOK.md).
#
# Usage:
#   ./scripts/run_local_gpu.sh smoke          # 3-task sanity check on qwen (~5 min)
#   ./scripts/run_local_gpu.sh qwen           # full 115x2 on qwen2.5:7b
#   ./scripts/run_local_gpu.sh llama          # full 115x2 on llama3.1:8b
#   ./scripts/run_local_gpu.sh all            # smoke, then qwen, then llama
#
# Every run appends to results/run_local.log (tail -f it from another
# terminal). After each full run the script verifies completeness and dumps
# that model's rows to results/export_<model>.json for transfer back to the
# Mac. Exports are gitignored — raw trial data never gets committed.
set -u
cd "$(dirname "$0")/.."

# uv installs to %USERPROFILE%\.local\bin and Ollama to LOCALAPPDATA on Windows.
# A freshly opened Git Bash may not have picked those up on PATH yet, so add them
# for this session (matches windows_setup.sh). No-op if already present or on POSIX.
if command -v cygpath >/dev/null 2>&1; then
    export PATH="$(cygpath "${USERPROFILE:-$HOME}")/.local/bin:$PATH"
    export PATH="$(cygpath "${LOCALAPPDATA:-$USERPROFILE/AppData/Local}")/Programs/Ollama:$PATH"
fi

LOG=results/run_local.log
MODE="${1:-all}"

say() { echo "[run_local_gpu] $*" | tee -a "$LOG"; }

require_ollama() {
    if ! command -v ollama >/dev/null 2>&1; then
        say "ERROR: ollama not installed — https://ollama.com/download"
        exit 1
    fi
    if ! ollama list >/dev/null 2>&1; then
        say "ERROR: ollama server not reachable — start the Ollama app/service"
        exit 1
    fi
}

ensure_model() {
    local model="$1"
    if ! ollama list | awk '{print $1}' | grep -qx "$model"; then
        say "pulling $model ..."
        ollama pull "$model" 2>&1 | tee -a "$LOG"
    fi
}

verify_and_export() {
    local model="$1"
    say "verifying completeness for $model ..."
    uv run python scripts/export_trials.py --model "$model" 2>&1 | tee -a "$LOG"
    say "send the exported file (or results/trials.db) back for analysis"
}

run_suite() {
    local model="$1"; shift
    ensure_model "$model"
    say "=== FULL RUN $model $(date) ==="
    uv run python scripts/run_tau_eval.py --provider ollama --model "$model" "$@" 2>&1 | tee -a "$LOG"
    verify_and_export "$model"
}

require_ollama
mkdir -p results

case "$MODE" in
    smoke)
        ensure_model qwen2.5:7b
        say "=== SMOKE qwen2.5:7b $(date) ==="
        uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:7b --limit 3 2>&1 | tee -a "$LOG"
        say "smoke done — check the lines above against RUNBOOK.md §2 before launching the full run"
        ;;
    qwen)  run_suite qwen2.5:7b ;;
    llama) run_suite llama3.1:8b ;;
    all)
        "$0" smoke
        say "smoke finished; starting full qwen run in 30s (Ctrl-C now if smoke looked broken)"
        sleep 30
        run_suite qwen2.5:7b
        run_suite llama3.1:8b
        say "ALL DONE $(date) — send results/export_*.json (or results/trials.db) back to the Mac"
        ;;
    *)
        echo "usage: $0 [smoke|qwen|llama|all]"; exit 1 ;;
esac
