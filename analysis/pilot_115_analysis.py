"""Analysis of the 115-task τ-bench pilot (gpt-4o-mini, 2026-07-07).

Produces:
  1. Per-task outcome categories (vanilla × compass success/abstain/mutation).
  2. The 40-task frontier subsample (stratified, seeded) →
     tasks/tau_bench/frontier_subsample.json
  3. T_HIGH sensitivity: offline replay of stored per-step success_probs
     against alternative thresholds (approximate — see caveat in output).

Run: uv run python analysis/pilot_115_analysis.py
"""
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

from compass.policy import max_risk
from compass.tools.tau_retail import TOOL_RISK

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "results" / "trials.db"
SUBSAMPLE_FILE = ROOT / "tasks" / "tau_bench" / "frontier_subsample.json"
MODEL = "gpt-4o-mini"
SEED = 42
SUBSAMPLE_SIZE = 40

# per-stratum quota: weighted toward mutation-prone / discriminative tasks
QUOTA = {
    "both_fail_v_mutates": 14,   # core compound-failure battleground
    "both_fail_both_mutate": 6,  # compass gate leaked
    "vanilla_only_abstained": 6, # over-abstention cost
    "vanilla_only_committed": 4, # compass committed and still failed
    "compass_only": 4,           # compass strictly better
    "both_succeed": 3,           # ceiling anchor
    "both_fail_clean": 3,        # hard, no destructive action
}


def load_tasks() -> dict:
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        """SELECT task_id, condition, success, abstained, mutated_order_ids,
                  success_probs, risk_levels, trace
           FROM trials WHERE task_id LIKE 'tau_retail%' AND model = ?""",
        (MODEL,),
    ).fetchall()
    tasks: dict = defaultdict(dict)
    for tid, cond, succ, abst, mut, probs, risks, trace in rows:
        tasks[tid][cond] = {
            "success": bool(succ),
            "abstained": bool(abst),
            "wrong_mutation": bool(json.loads(mut)) and not succ,
            "success_probs": json.loads(probs),
            "risk_levels": json.loads(risks),
            "tools": [s["tool"] for s in json.loads(trace).get("steps", [])],
        }
    return dict(tasks)


def categorize(v: dict, c: dict) -> str:
    if v["success"] and c["success"]:
        return "both_succeed"
    if v["success"]:
        return "vanilla_only_abstained" if c["abstained"] else "vanilla_only_committed"
    if c["success"]:
        return "compass_only"
    if v["wrong_mutation"] and c["wrong_mutation"]:
        return "both_fail_both_mutate"
    if v["wrong_mutation"]:
        return "both_fail_v_mutates"
    return "both_fail_clean"


def pick_subsample(by_cat: dict[str, list[str]]) -> list[str]:
    rng = random.Random(SEED)
    chosen: list[str] = []
    for cat, quota in QUOTA.items():
        pool = sorted(by_cat.get(cat, []))
        take = min(quota, len(pool))
        chosen += rng.sample(pool, take)
    # top up from the biggest pool if any stratum came up short
    if len(chosen) < SUBSAMPLE_SIZE:
        rest = sorted(set(t for pool in by_cat.values() for t in pool) - set(chosen))
        chosen += rng.sample(rest, SUBSAMPLE_SIZE - len(chosen))
    return sorted(chosen)


def _effective_risks(c: dict) -> list[str]:
    """Effective risk per step: max(verbalized, static tool class) when the
    trace (tool names) is available; verbalized-only otherwise. The Jul 1
    pilot rows predate trace storage (trace='{}'), so for them effective
    risk is UNDER-estimated wherever the model under-labelled a destructive
    tool — the replay validation below quantifies the impact."""
    if c["tools"]:
        return [
            max_risk(verb, TOOL_RISK.get(tool, "low")) if tool else verb
            for verb, tool in zip(c["risk_levels"], c["tools"])
        ]
    return list(c["risk_levels"])


def t_high_sensitivity(tasks: dict) -> None:
    """Replay compass per-step success_probs against alternative T_HIGH values.

    CAVEAT: approximate twice over. (1) We only observe the trajectory the
    shipped T_HIGH=0.8 produced; under another threshold it would diverge
    after the first differing decision. (2) Trials without stored traces
    use verbalized risk only (see _effective_risks). The @0.8 row's
    agreement with the actually-observed abstain flag calibrates how much
    to trust the rest of the curve.
    """
    traced = sum(1 for d in tasks.values() if d["compass"]["tools"])
    print(f"\nT_HIGH sensitivity (offline replay; {traced}/{len(tasks)} trials have traces):")

    # validation: does the replay at the shipped threshold reproduce reality?
    agree = 0
    for d in tasks.values():
        c = d["compass"]
        replayed = any(
            r == "high" and p < 0.8
            for p, r in zip(c["success_probs"], _effective_risks(c))
        )
        agree += replayed == c["abstained"]
    print(f"  replay@0.8 vs observed abstain flag: {agree}/{len(tasks)} agree")

    print(f"{'T_HIGH':>7} {'abstain%':>9} {'pre-mutation abstain% (traced only)':>36}")
    for t_high in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        abstain = pre_mut = n_mut = 0
        for d in tasks.values():
            c = d["compass"]
            risks = _effective_risks(c)
            first_abstain = next(
                (i for i, (p, r) in enumerate(zip(c["success_probs"], risks))
                 if r == "high" and p < t_high), None)
            if first_abstain is not None:
                abstain += 1
            if c["tools"]:  # mutation step index needs tool names
                first_mut = next(
                    (i for i, tool in enumerate(c["tools"])
                     if tool and TOOL_RISK.get(tool) == "high"
                     and tool != "transfer_to_human_agents"), None)
                if first_mut is not None:
                    n_mut += 1
                    if first_abstain is not None and first_abstain <= first_mut:
                        pre_mut += 1
        print(
            f"{t_high:>7} {abstain / len(tasks):>8.1%} "
            f"{pre_mut / n_mut if n_mut else 0:>35.1%}"
        )


def main() -> None:
    tasks = load_tasks()
    by_cat: dict[str, list[str]] = defaultdict(list)
    for tid, d in sorted(tasks.items()):
        by_cat[categorize(d["vanilla"], d["compass"])].append(tid)

    print(f"Pilot outcome categories (n={len(tasks)}, {MODEL}):")
    for cat, ids in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(ids):3d}  {cat}")

    subsample = pick_subsample(by_cat)
    assert len(subsample) == SUBSAMPLE_SIZE == len(set(subsample))
    SUBSAMPLE_FILE.write_text(json.dumps(
        {
            "description": "40-task frontier subsample, stratified over "
                           f"{MODEL} pilot outcome categories, seed={SEED}",
            "strata": {cat: sorted(set(by_cat.get(cat, [])) & set(subsample))
                       for cat in QUOTA},
            "task_ids": subsample,
        },
        indent=2,
    ) + "\n")
    print(f"\nWrote {SUBSAMPLE_SIZE}-task subsample → {SUBSAMPLE_FILE.relative_to(ROOT)}")
    for cat in QUOTA:
        n = len(set(by_cat.get(cat, [])) & set(subsample))
        print(f"  {n:3d}/{QUOTA[cat]:<3d} {cat}")

    t_high_sensitivity(tasks)


if __name__ == "__main__":
    main()
