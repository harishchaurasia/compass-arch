"""Tests for evaluation metrics: ECE, Brier, selective success, compound failure."""
import math

import pytest

from eval.metrics import brier_score, compound_failure_rate, ece, selective_success_rate
from eval.trial_store import TrialResult


def _trial(
    success: bool,
    abstained: bool = False,
    mutated_order_ids: list[str] | None = None,
) -> TrialResult:
    return TrialResult(
        task_id="retail_001",
        condition="compass",
        model="fake",
        success=success,
        steps=3,
        abstained=abstained,
        confidence_scores=[0.9],
        final_message="done",
        success_probs=[0.9],
        mutated_order_ids=mutated_order_ids or [],
    )


# ── brier (regression — already implemented) ─────────────────────────────────

def test_brier_score_known_value():
    # (0.9-1)^2 + (0.8-0)^2 = 0.01 + 0.64 → / 2 = 0.325
    assert brier_score([0.9, 0.8], [1, 0]) == pytest.approx(0.325)


# ── ece ───────────────────────────────────────────────────────────────────────

def test_ece_zero_when_perfectly_calibrated():
    # All in one bin: avg confidence 0.75, accuracy 3/4 = 0.75 → gap 0
    assert ece([0.75, 0.75, 0.75, 0.75], [1, 1, 1, 0]) == pytest.approx(0.0)

def test_ece_measures_overconfidence():
    # One bin: avg confidence 0.9, accuracy 0.25 → ECE = 0.65
    assert ece([0.9, 0.9, 0.9, 0.9], [1, 0, 0, 0]) == pytest.approx(0.65)


def test_ece_weights_bins_by_count():
    # bin [0.6, 0.7): conf 0.6, acc 0.5 → gap 0.1, weight 1/2
    # bin [0.9, 1.0]: conf 0.95, acc 1.0 → gap 0.05, weight 1/2
    result = ece([0.6, 0.6, 0.95, 0.95], [1, 0, 1, 1])
    assert result == pytest.approx(0.5 * 0.1 + 0.5 * 0.05)


def test_ece_handles_confidence_of_exactly_one():
    assert ece([1.0], [1]) == pytest.approx(0.0)


# ── selective success rate ────────────────────────────────────────────────────

def test_selective_success_rate_excludes_abstentions_from_accuracy():
    trials = [
        _trial(success=True),
        _trial(success=True),
        _trial(success=False),
        _trial(success=False, abstained=True),
    ]
    accuracy, abstention_rate = selective_success_rate(trials)
    assert accuracy == pytest.approx(2 / 3)
    assert abstention_rate == pytest.approx(1 / 4)


def test_selective_success_rate_all_abstained_is_nan_accuracy():
    trials = [_trial(success=False, abstained=True)]
    accuracy, abstention_rate = selective_success_rate(trials)
    assert math.isnan(accuracy)
    assert abstention_rate == pytest.approx(1.0)


# ── compound failure rate ─────────────────────────────────────────────────────

def test_compound_failure_rate_counts_wrong_and_destructive():
    trials = [
        # correct mutation — not a compound failure
        _trial(success=True, mutated_order_ids=["#W1111111"]),
        # wrong AND mutated the DB — the production nightmare
        _trial(success=False, mutated_order_ids=["#W2222222"]),
        # wrong but touched nothing — a loud failure, not compound
        _trial(success=False),
        # abstained without mutating — safe
        _trial(success=False, abstained=True),
    ]
    assert compound_failure_rate(trials) == pytest.approx(1 / 4)


def test_compound_failure_rate_counts_mutation_before_abstention():
    # Abstaining does not undo a destructive action already taken
    trials = [_trial(success=False, abstained=True, mutated_order_ids=["#W1111111"])]
    assert compound_failure_rate(trials) == pytest.approx(1.0)


def test_compound_failure_rate_empty_is_zero():
    assert compound_failure_rate([]) == 0.0
