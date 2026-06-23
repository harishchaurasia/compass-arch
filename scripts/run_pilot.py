"""Phase 1 pilot: 5 tasks × 2 conditions × 1 model.

Usage:
    uv run python scripts/run_pilot.py

Writes results to results/trials.db and prints a summary table.
"""
import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic

from compass.agent_compass import build_compass_agent
from compass.agent_vanilla import build_vanilla_agent
from compass.tools.retail import (
    cancel_pending_order,
    find_user_id_by_email,
    find_user_id_by_name_zip,
    get_order_details,
    get_user_details,
    modify_pending_order_address,
    return_delivered_order_items,
)
from eval.tau_bench_runner import run_trial
from eval.trial_store import load_trials, save_trial

TASKS_FILE = Path(__file__).parent.parent / "tasks" / "tau_bench" / "tasks.json"
DB_PATH = Path(__file__).parent.parent / "results" / "trials.db"
MODEL_NAME = "claude-haiku-4-5-20251001"  # cheapest for pilot; swap to sonnet/opus for full eval

TOOLS = [
    find_user_id_by_name_zip,
    find_user_id_by_email,
    get_user_details,
    get_order_details,
    cancel_pending_order,
    return_delivered_order_items,
    modify_pending_order_address,
]


def main():
    tasks = json.loads(TASKS_FILE.read_text())
    model = ChatAnthropic(model=MODEL_NAME, temperature=0)

    vanilla = build_vanilla_agent(model, TOOLS)
    compass = build_compass_agent(model, TOOLS)

    print(f"\n{'─'*60}")
    print(f"Phase 1 Pilot  |  {len(tasks)} tasks × 2 conditions  |  {MODEL_NAME}")
    print(f"{'─'*60}\n")

    for task in tasks:
        for condition, agent in [("vanilla", vanilla), ("compass", compass)]:
            print(f"  Running {task['id']} / {condition} ... ", end="", flush=True)
            try:
                result = run_trial(task, agent, condition=condition, model=MODEL_NAME)
                save_trial(result, DB_PATH)
                status = "✓" if result.success else "✗"
                conf = (
                    f"  conf={[round(c,2) for c in result.confidence_scores]}"
                    if condition == "compass" else ""
                )
                abstain = "  [ABSTAINED]" if result.abstained else ""
                print(f"{status}  steps={result.steps}{conf}{abstain}")
            except Exception as e:
                print(f"ERROR: {e}")

    # ── summary ──────────────────────────────────────────────────────────────
    results = load_trials(DB_PATH)
    print(f"\n{'─'*60}")
    print("SUMMARY")
    print(f"{'─'*60}")

    for condition in ("vanilla", "compass"):
        subset = [r for r in results if r.condition == condition]
        successes = sum(r.success for r in subset)
        abstentions = sum(r.abstained for r in subset)
        avg_steps = sum(r.steps for r in subset) / max(len(subset), 1)

        print(f"\n{condition.upper()}:")
        print(f"  Success rate : {successes}/{len(subset)}")
        print(f"  Avg steps    : {avg_steps:.1f}")
        if condition == "compass":
            all_conf = [c for r in subset for c in r.confidence_scores]
            if all_conf:
                print(f"  Confidence   : min={min(all_conf):.2f}  max={max(all_conf):.2f}  mean={sum(all_conf)/len(all_conf):.2f}")
                spread = max(all_conf) - min(all_conf)
                verdict = "VARIES (good signal)" if spread > 0.15 else "FLAT (weak signal — lean on trajectory features)"
                print(f"  Spread={spread:.2f}  → {verdict}")
            print(f"  Abstentions  : {abstentions}")

    print(f"\n{'─'*60}")
    print(f"Results saved to {DB_PATH}")


if __name__ == "__main__":
    main()
