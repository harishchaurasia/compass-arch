"""Confidence-conditioned action policy (see DESIGN.md Table 1)."""
from enum import Enum

T_MED: float = 0.6   # tuned on dev split
T_HIGH: float = 0.8  # tuned on dev split


class PolicyDecision(Enum):
    EXECUTE = "execute"
    SELF_VERIFY = "self_verify"
    ABSTAIN = "abstain"


_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def max_risk(a: str, b: str) -> str:
    """Higher of two risk levels. Used to floor the model's verbalized
    risk_level with a tool's static risk class — the Phase 2 pilot showed
    models label destructive actions 'low', so verbalized risk alone
    cannot be trusted to gate anything."""
    for level in (a, b):
        if level not in _RISK_ORDER:
            raise ValueError(f"Unknown risk_level: {level!r}")
    return a if _RISK_ORDER[a] >= _RISK_ORDER[b] else b


def decide(success_prob: float, risk_level: str) -> PolicyDecision:
    if risk_level == "low":
        return PolicyDecision.EXECUTE
    if risk_level == "medium":
        return PolicyDecision.EXECUTE if success_prob >= T_MED else PolicyDecision.SELF_VERIFY
    if risk_level == "high":
        return PolicyDecision.EXECUTE if success_prob >= T_HIGH else PolicyDecision.ABSTAIN
    raise ValueError(f"Unknown risk_level: {risk_level!r}")
