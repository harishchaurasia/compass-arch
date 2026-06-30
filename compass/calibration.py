"""Aggregate verbalized confidence + trajectory features → success_prob.

Rule-based aggregator locked on the 5-task dev split.
Do NOT tune thresholds mid-experiment; changes belong in a new phase.
"""
from compass.trajectory import TrajectoryFeatures


def calibrate(verbalized_confidence: float, features: TrajectoryFeatures) -> float:
    """Return success_prob ∈ [0, 1], correcting for known model overconfidence."""
    prob = verbalized_confidence

    # Step count penalty — only the highest bracket applies
    if features.step_count > 10:
        prob -= 0.20
    elif features.step_count > 7:
        prob -= 0.10
    elif features.step_count > 5:
        prob -= 0.05

    # Oscillation cap — applied after step penalty so the cap can bind
    if features.is_oscillating:
        prob = min(prob, 0.50)

    # Stuck-on-same-tool penalty
    if features.steps_since_new_tool >= 2:
        prob -= 0.10

    return max(0.0, min(1.0, prob))
