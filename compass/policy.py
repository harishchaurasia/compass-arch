"""Confidence-conditioned action policy (see DESIGN.md Table 1)."""
from enum import Enum

T_MED: float = 0.6   # tuned on dev split
T_HIGH: float = 0.8  # tuned on dev split


class PolicyDecision(Enum):
    EXECUTE = "execute"
    SELF_VERIFY = "self_verify"
    ABSTAIN = "abstain"


def decide(success_prob: float, risk_level: str) -> PolicyDecision:
    raise NotImplementedError
