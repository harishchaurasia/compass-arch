"""Tests for the leaderboard generator's ranking logic."""

import importlib.util
from pathlib import Path

from eval.trial_store import TrialResult

# scripts/ is not a package; load the module by path.
_spec = importlib.util.spec_from_file_location(
    "leaderboard", Path(__file__).parents[1] / "scripts" / "leaderboard.py"
)
leaderboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leaderboard)


def test_base_model_strips_variant_suffixes():
    assert leaderboard._base_model("qwen2.5:14b") == "qwen2.5:14b"
    assert leaderboard._base_model("qwen2.5:14b-shrink") == "qwen2.5:14b"
    assert leaderboard._base_model("qwen2.5:14b-noverify") == "qwen2.5:14b"
    assert leaderboard._base_model("gpt-4o-mini-thigh0.7") == "gpt-4o-mini"


def _trial(model, condition, success, abstained, mutated):
    return TrialResult(
        task_id="tau_airline_000",
        condition=condition,
        model=model,
        success=success,
        steps=1,
        abstained=abstained,
        confidence_scores=[],
        final_message="",
        mutated_order_ids=mutated,
    )


def test_ranking_prefers_low_compound_then_high_selective_success():
    # Two models both reach 0% compound; the one that keeps coverage should win.
    rows = []
    # model A: compass never mutates, never abstains, always succeeds -> 0% comp, high sel
    for i in range(4):
        rows.append(
            _trial("A", "vanilla", False, False, ["x"])
        )  # vanilla mutates (baseline)
        rows.append(_trial("A", "compass", True, False, []))
    # model B: compass reaches 0% only by abstaining on everything -> 0% comp, 0% sel
    for i in range(4):
        rows.append(_trial("B", "vanilla", False, False, ["x"]))
        rows.append(_trial("B", "compass", False, True, []))
    # tag their task_ids so the airline-prefixed filter keeps them
    table, count = leaderboard._suite_table(rows, "tau_airline_")
    assert count == 2
    # A (coverage kept) must rank above B (refuses everything)
    assert table.index("`A`") < table.index("`B`")
