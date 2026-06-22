"""Tests for the confidence-conditioned action policy."""
from compass.policy import PolicyDecision, T_HIGH, T_MED, decide


# ── low risk ──────────────────────────────────────────────────────────────────

def test_low_risk_always_executes():
    assert decide(0.0, "low") == PolicyDecision.EXECUTE
    assert decide(0.5, "low") == PolicyDecision.EXECUTE
    assert decide(1.0, "low") == PolicyDecision.EXECUTE


# ── medium risk ───────────────────────────────────────────────────────────────

def test_medium_risk_high_confidence_executes():
    assert decide(T_MED, "medium") == PolicyDecision.EXECUTE
    assert decide(T_MED + 0.1, "medium") == PolicyDecision.EXECUTE

def test_medium_risk_low_confidence_self_verifies():
    assert decide(T_MED - 0.01, "medium") == PolicyDecision.SELF_VERIFY
    assert decide(0.0, "medium") == PolicyDecision.SELF_VERIFY


# ── high risk ─────────────────────────────────────────────────────────────────

def test_high_risk_high_confidence_executes():
    assert decide(T_HIGH, "high") == PolicyDecision.EXECUTE
    assert decide(T_HIGH + 0.1, "high") == PolicyDecision.EXECUTE

def test_high_risk_low_confidence_abstains():
    assert decide(T_HIGH - 0.01, "high") == PolicyDecision.ABSTAIN
    assert decide(0.0, "high") == PolicyDecision.ABSTAIN
