"""Run the vendored τ-bench retail suite (single-shot variant).

Usage:
    uv run python scripts/run_tau_eval.py                # all 115 tasks × 2 conditions
    uv run python scripts/run_tau_eval.py --limit 3      # smoke: first 3 tasks
    uv run python scripts/run_tau_eval.py --model gpt-4o-mini

Writes one row per trial to results/trials.db (task ids are tau_retail_*,
distinguishing them from the homemade retail_* suite).
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import compass.tools.tau_retail.db as tau_db
from compass.agent_compass import build_compass_agent
from compass.agent_vanilla import build_vanilla_agent
from compass.tools.tau_retail import ALL_TOOLS, TOOL_RISK
from eval.metrics import compound_failure_rate, selective_success_rate
from eval.tau_bench_runner import run_trial
from eval.trial_store import load_trials, save_trial

ROOT = Path(__file__).parent.parent
TASKS_FILE = ROOT / "tasks" / "tau_bench" / "tasks_real.json"
WIKI_FILE = ROOT / "tasks" / "tau_bench" / "real_data" / "wiki.md"
DB_PATH = ROOT / "results" / "trials.db"

# τ-bench instructions are briefs for an interactive user simulator; our
# variant is single-shot. Without this framing both agents (correctly, per
# policy) stop to ask the customer for confirmation that can never come.
_SINGLE_SHOT_TEMPLATE = (
    "Customer request (single-shot; written from the customer's perspective):\n"
    "{instruction}\n\n"
    "The customer has already given explicit confirmation for the actions "
    "requested above. Complete everything the policy allows in this single "
    "interaction without asking follow-up questions. If something cannot be "
    "done under policy, say so or transfer to a human agent."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None, help="run first N tasks only")
    parser.add_argument("--task-ids", nargs="*", default=None, help="run only these task ids")
    parser.add_argument(
        "--conditions", nargs="*", default=["vanilla", "compass"],
        choices=["vanilla", "compass"],
    )
    args = parser.parse_args()

    load_dotenv()
    tasks = json.loads(TASKS_FILE.read_text())
    if args.task_ids:
        tasks = [t for t in tasks if t["id"] in set(args.task_ids)]
    if args.limit:
        tasks = tasks[: args.limit]
    policy = WIKI_FILE.read_text()

    model = ChatOpenAI(model=args.model, temperature=0)
    vanilla = build_vanilla_agent(model, ALL_TOOLS, policy=policy)
    compass = build_compass_agent(model, ALL_TOOLS, tool_risk=TOOL_RISK, policy=policy)

    print(f"\n{'─' * 60}")
    print(f"τ-bench retail (single-shot)  |  {len(tasks)} tasks × 2 conditions  |  {args.model}")
    print(f"{'─' * 60}\n")

    for task in tasks:
        task = {
            **task,
            "instruction": _SINGLE_SHOT_TEMPLATE.format(instruction=task["instruction"]),
        }
        agents = [("vanilla", vanilla), ("compass", compass)]
        for condition, agent in [(c, a) for c, a in agents if c in args.conditions]:
            print(f"  {task['id']} / {condition} ... ", end="", flush=True)
            try:
                tau_db.reset()
                result = run_trial(task, agent, condition=condition, model=args.model)
                save_trial(result, DB_PATH)
                status = "✓" if result.success else "✗"
                extra = ""
                if condition == "compass":
                    extra = f"  conf={[round(c, 2) for c in result.confidence_scores]}"
                if result.abstained:
                    extra += "  [ABSTAINED]"
                if not result.success and result.mutated_order_ids:
                    extra += f"  [MUTATED {result.mutated_order_ids}]"
                print(f"{status}  steps={result.steps}{extra}")
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")

    # ── summary over this suite (all models/rows with tau_retail_ prefix) ────
    rows = [
        r for r in load_trials(DB_PATH)
        if r.task_id.startswith("tau_retail_") and r.model == args.model
    ]
    print(f"\n{'─' * 60}\nSUMMARY ({args.model}, all tau_retail rows in db)\n{'─' * 60}")
    for condition in ("vanilla", "compass"):
        subset = [r for r in rows if r.condition == condition]
        if not subset:
            continue
        acc, abst = selective_success_rate(subset)
        print(f"\n{condition.upper()}: n={len(subset)}")
        print(f"  Selective success : {acc:.3f}")
        print(f"  Abstention rate   : {abst:.3f}")
        print(f"  Compound failures : {compound_failure_rate(subset):.3f}")


if __name__ == "__main__":
    main()
