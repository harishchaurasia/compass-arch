"""Aggregate verbalized confidence + trajectory features → success_prob.

Rule-based aggregator locked on the 5-task dev split.
Do NOT tune thresholds mid-experiment; changes belong in a new phase.
"""
from compass.trajectory import TrajectoryFeatures

# ── Phase 4 variant: confidence shrinkage ───────────────────────────────────────
# The baseline aggregator trusts verbalized confidence directly, then discounts
# it using trajectory features. But those features don't exist yet at the FIRST
# high-risk action, and on overconfident models verbalized confidence collapses
# to ~1.0 — so the first destructive action sails past T_HIGH ungated (see the
# qwen2.5:14b compound-failure finding). Shrinkage pulls verbalized confidence
# toward a base-rate prior BEFORE the trajectory penalties, so a bare "1.0" no
# longer clears the high-risk bar on its own. Params are fixed a priori (a
# max-entropy 0.5 prior, equal 0.5 trust in the model vs the prior) — NOT tuned
# on the eval set — to keep the comparison honest.
SHRINK_PRIOR: float = 0.5
SHRINK_WEIGHT: float = 0.5


def calibrate(
    verbalized_confidence: float,
    features: TrajectoryFeatures,
    shrink: bool = False,
) -> float:
    """Return success_prob ∈ [0, 1], correcting for known model overconfidence.

    shrink=False is the locked baseline. shrink=True applies the Phase 4
    base-rate shrinkage prior before the (unchanged) trajectory penalties.
    """
    if shrink:
        verbalized_confidence = (
            SHRINK_WEIGHT * verbalized_confidence + (1.0 - SHRINK_WEIGHT) * SHRINK_PRIOR
        )
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
