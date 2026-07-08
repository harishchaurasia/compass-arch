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
    sqlite3 results/trials.db \
        "SELECT condition || ': ' || COUNT(*) || ' rows, ' || COUNT(DISTINCT task_id) || ' distinct tasks'
         FROM trials WHERE model='$model' AND task_id LIKE 'tau_retail%' GROUP BY condition;" \
        | tee -a "$LOG"
    local export="results/export_$(echo "$model" | tr ':/.' '___').json"
    sqlite3 results/trials.db \
        "SELECT json_group_array(json_object(
            'task_id', task_id, 'condition', condition, 'model', model,
            'success', success, 'steps', steps, 'abstained', abstained,
            'confidence_scores', json(confidence_scores),
            'success_probs', json(success_probs),
            'mutated_order_ids', json(mutated_order_ids),
            'risk_levels', json(risk_levels),
            'final_message', final_message, 'trace', json(trace),
            'created_at', created_at))
         FROM trials WHERE model='$model' AND task_id LIKE 'tau_retail%';" \
        > "$export"
    say "exported → $export ($(du -h "$export" | cut -f1)) — send this (or trials.db) back for analysis"
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
