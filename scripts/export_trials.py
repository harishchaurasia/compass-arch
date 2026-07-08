"""Verify completeness and export one model's trial rows to JSON.

Used by run_local_gpu.sh after each local run (cross-platform — avoids
needing the sqlite3 CLI, which Windows doesn't ship).

Usage: uv run python scripts/export_trials.py --model qwen2.5:7b
"""
import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "results" / "trials.db"

COLUMNS = [
    "task_id", "condition", "model", "success", "steps", "abstained",
    "confidence_scores", "success_probs", "mutated_order_ids",
    "risk_levels", "final_message", "trace", "created_at",
]
JSON_COLUMNS = {
    "confidence_scores", "success_probs", "mutated_order_ids",
    "risk_levels", "trace",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    db = sqlite3.connect(DB_PATH)
    for condition, n, distinct in db.execute(
        """SELECT condition, COUNT(*), COUNT(DISTINCT task_id) FROM trials
           WHERE model = ? AND task_id LIKE 'tau_retail%' GROUP BY condition""",
        (args.model,),
    ):
        print(f"  {condition}: {n} rows, {distinct} distinct tasks")

    rows = db.execute(
        f"SELECT {', '.join(COLUMNS)} FROM trials "
        "WHERE model = ? AND task_id LIKE 'tau_retail%'",
        (args.model,),
    ).fetchall()
    records = [
        {
            col: json.loads(val) if col in JSON_COLUMNS else val
            for col, val in zip(COLUMNS, row)
        }
        for row in rows
    ]

    safe_name = args.model.replace(":", "_").replace("/", "_").replace(".", "_")
    out = ROOT / "results" / f"export_{safe_name}.json"
    out.write_text(json.dumps(records))
    print(f"  exported {len(records)} trials -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
