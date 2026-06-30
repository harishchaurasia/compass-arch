"""Tests for the calibration aggregator.

calibrate(verbalized_confidence, features) → success_prob ∈ [0, 1]

Rules (locked on 5-task dev split, do not tune mid-experiment):
  - step_count > 10  → -0.20
  - step_count > 7   → -0.10
  - step_count > 5   → -0.05
  - is_oscillating   → cap at 0.50  (applied after step penalty)
  - steps_since_new_tool ≥ 2 → -0.10
  - result clamped to [0.0, 1.0]
"""
import pytest
from compass.calibration import calibrate
from compass.trajectory import TrajectoryFeatures


def _features(
    step_count: int = 3,
    is_oscillating: bool = False,
    unique_tools_used: int = 2,
    steps_since_new_tool: int = 0,
) -> TrajectoryFeatures:
    return TrajectoryFeatures(
        step_count=step_count,
        is_oscillating=is_oscillating,
        unique_tools_used=unique_tools_used,
        steps_since_new_tool=steps_since_new_tool,
    )


# ── clean run (no penalties) ──────────────────────────────────────────────────

def test_clean_run_passes_through_confidence():
    """Short run, no oscillation, fresh tools → prob ≈ verbalized confidence."""
    prob = calibrate(0.9, _features(step_count=3))
    assert prob == pytest.approx(0.9)


def test_clean_run_low_confidence_unchanged():
    prob = calibrate(0.5, _features(step_count=2))
    assert prob == pytest.approx(0.5)


# ── step count penalties ──────────────────────────────────────────────────────

def test_step_count_above_5_small_penalty():
    prob = calibrate(0.9, _features(step_count=6))
    assert prob == pytest.approx(0.85)  # 0.9 - 0.05


def test_step_count_above_7_medium_penalty():
    prob = calibrate(0.9, _features(step_count=8))
    assert prob == pytest.approx(0.80)  # 0.9 - 0.10


def test_step_count_above_10_large_penalty():
    """Task 003 scenario: 11 steps, model says 0.9 → pulls down to 0.70."""
    prob = calibrate(0.9, _features(step_count=11))
    assert prob == pytest.approx(0.70)  # 0.9 - 0.20


def test_step_count_penalties_are_mutually_exclusive():
    """Only the highest applicable penalty applies, not all of them stacked."""
    # step_count=11 triggers >10 rule only (-0.20), not also >7 and >5
    prob_11 = calibrate(0.9, _features(step_count=11))
    prob_8  = calibrate(0.9, _features(step_count=8))
    assert prob_11 < prob_8  # bigger penalty for more steps


# ── oscillation cap ───────────────────────────────────────────────────────────

def test_oscillation_caps_at_050():
    prob = calibrate(0.9, _features(is_oscillating=True))
    assert prob == pytest.approx(0.50)


def test_oscillation_cap_applied_after_step_penalty():
    """Step penalty first, then oscillation cap — worst case task 003."""
    # 0.9 - 0.20 (steps=11) = 0.70, then capped at 0.50
    prob = calibrate(0.9, _features(step_count=11, is_oscillating=True))
    assert prob == pytest.approx(0.50)


def test_oscillation_cap_does_not_raise_low_confidence():
    """If confidence is already below 0.50, oscillation cap doesn't raise it."""
    prob = calibrate(0.3, _features(is_oscillating=True))
    assert prob == pytest.approx(0.30)


# ── stuck-on-same-tool penalty ────────────────────────────────────────────────

def test_steps_since_new_tool_penalty():
    prob = calibrate(0.9, _features(steps_since_new_tool=2))
    assert prob == pytest.approx(0.80)  # 0.9 - 0.10


def test_steps_since_new_tool_below_threshold_no_penalty():
    prob = calibrate(0.9, _features(steps_since_new_tool=1))
    assert prob == pytest.approx(0.90)


# ── combined penalties ────────────────────────────────────────────────────────

def test_combined_step_and_stuck_penalties():
    # step_count=8 → -0.10, steps_since_new_tool=2 → -0.10 → 0.9-0.20=0.70
    prob = calibrate(0.9, _features(step_count=8, steps_since_new_tool=2))
    assert prob == pytest.approx(0.70)


def test_combined_all_penalties_worst_case():
    # step=11 (-0.20), oscillating (cap 0.50), stuck (-0.10)
    # 0.9 - 0.20 = 0.70, cap at 0.50, -0.10 = 0.40
    prob = calibrate(0.9, _features(step_count=11, is_oscillating=True, steps_since_new_tool=2))
    assert prob == pytest.approx(0.40)


# ── clamping ──────────────────────────────────────────────────────────────────

def test_result_never_below_zero():
    prob = calibrate(0.1, _features(step_count=11, is_oscillating=True, steps_since_new_tool=2))
    assert prob >= 0.0


def test_result_never_above_one():
    prob = calibrate(1.0, _features())
    assert prob <= 1.0
