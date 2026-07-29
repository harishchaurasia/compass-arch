"""Tests for the semantic calibration probe (compass/probe.py) and its wiring
into build_compass_agent(calibration="probe").

Uses a fake bag-of-words embedder so nothing touches Ollama; the shipped weights
(compass/probe_weights.json) are exercised through it.
"""
import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from compass.agent_compass import CompassAction, CompassStep, build_compass_agent
from compass.probe import (
    WEIGHTS_PATH,
    ProbeUnavailable,
    SemanticProbe,
    extract_features,
    observations_from_messages,
    request_from_messages,
)

_VOCAB = ["cancel", "boston", "flight", "delete", "record", "water", "bottle", "res9", "r99"]


class FakeEmbedder:
    """Deterministic presence-vector embedder; records calls so a test can assert
    the probe path actually ran. model_id matches the shipped weights."""

    model_id = "nomic-embed-text"

    def __init__(self):
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        low = text.lower()
        vec = [float(w in low) for w in _VOCAB]
        return vec if any(vec) else [1.0] + [0.0] * (len(_VOCAB) - 1)


class RaisingEmbedder:
    model_id = "nomic-embed-text"

    def embed(self, text: str):
        raise ProbeUnavailable("backend down")


def _spec() -> dict:
    return json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))


def _probe(embedder=None) -> SemanticProbe:
    return SemanticProbe(_spec(), embedder or FakeEmbedder())


# ── feature extraction ──────────────────────────────────────────────────────────

def test_extract_features_order_and_passthrough():
    step = CompassStep(
        reasoning="cancel the boston flight",
        action=CompassAction(tool="cancel_reservation", args={"reservation_id": "res9"}),
        confidence=0.7,
        risk_level="high",
    )
    feats = extract_features(
        request="cancel boston flight",
        observations=["Tool 'get' returned: res9 boston flight"],
        step=step,
        n_prior_tool_calls=2,
        step_index=3,
        embedder=FakeEmbedder(),
    )
    assert len(feats) == 6
    assert feats[0] == 0.7   # confidence
    assert feats[1] == 3.0   # step_index
    assert feats[2] == 2.0   # n_prior_tool_calls
    for v in feats[3:]:      # three affinities are cosines in [0, 1] here
        assert 0.0 <= v <= 1.0


def test_success_prob_in_unit_interval_and_responsive():
    probe = _probe()
    step = CompassStep(
        reasoning="cancel boston flight",
        action=CompassAction(tool="cancel_reservation", args={"reservation_id": "res9"}),
        confidence=0.9,
        risk_level="high",
    )
    matched = probe.success_prob("cancel boston flight",
                                 ["Tool 'g' returned: res9 boston flight"], step, 1, 1)
    mismatched = probe.success_prob("cancel boston flight",
                                    ["Tool 'g' returned: res9 water bottle"], step, 1, 1)
    for p in (matched, mismatched):
        assert 0.0 <= p <= 1.0
    # the probe is a real function of the affinity features, not a constant
    assert matched != mismatched


def test_weights_mismatched_embedder_is_rejected():
    class WrongModel:
        model_id = "some-other-embedder"

        def embed(self, text):
            return [0.0]

    try:
        SemanticProbe(_spec(), WrongModel())
    except ValueError as e:
        assert "embed model" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched embedder model")


# ── agent wiring ─────────────────────────────────────────────────────────────────

class FakeCompassModel:
    def __init__(self, steps):
        self._iter = iter(steps)

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages) -> dict:
        return {"parsed": next(self._iter), "raw": None, "parsing_error": None}


@tool
def risky_cancel(reservation_id: str) -> str:
    """Cancel a reservation permanently."""
    return f"cancelled {reservation_id}"


def _init(content: str) -> dict:
    return {"messages": [HumanMessage(content=content)], "steps": [],
            "abstained": False, "self_verify_count": 0}


def test_probe_mode_scores_high_risk_via_probe():
    fake = FakeEmbedder()
    steps = [
        CompassStep(
            reasoning="cancel boston flight",
            action=CompassAction(tool="risky_cancel", args={"reservation_id": "res9"}),
            confidence=0.9,
            risk_level="high",
        ),
        # terminating step in case the probe scores high enough to execute and loop
        CompassStep(
            reasoning="done",
            action=CompassAction(final_answer="cancelled"),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(
        FakeCompassModel(steps), [risky_cancel],
        tool_risk={"risky_cancel": "high"},
        calibration="probe", probe=_probe(fake), verification=False,
    )
    agent.invoke(_init("cancel boston flight"))
    # the probe was consulted (embedder saw the request) rather than the rule score
    assert any("boston" in c for c in fake.calls)


def test_probe_falls_back_when_embedder_unavailable():
    # embedder raises -> _score falls back to the rule-based aggregator; a high-risk
    # action at confidence 0.5 is below T_HIGH, so it must still abstain (not crash).
    steps = [CompassStep(
        reasoning="cancel it",
        action=CompassAction(tool="risky_cancel", args={"reservation_id": "res9"}),
        confidence=0.5,
        risk_level="high",
    )]
    agent = build_compass_agent(
        FakeCompassModel(steps), [risky_cancel],
        tool_risk={"risky_cancel": "high"},
        calibration="probe", probe=_probe(RaisingEmbedder()), verification=False,
    )
    state = agent.invoke(_init("cancel it"))
    assert state["abstained"] is True


def test_calibration_shrink_synonym_and_invalid_mode():
    # calibration="shrink" and the legacy bool build the same way (no error)
    build_compass_agent(FakeCompassModel([]), [risky_cancel], calibration="shrink")
    build_compass_agent(FakeCompassModel([]), [risky_cancel], calibration_shrink=True)
    try:
        build_compass_agent(FakeCompassModel([]), [risky_cancel], calibration="bogus")
    except ValueError as e:
        assert "calibration" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for unknown calibration mode")


def test_message_helpers_extract_request_and_observations():
    msgs = [
        HumanMessage(content="cancel boston flight"),
        HumanMessage(content="Tool 'get_reservation' returned: res9 boston"),
        HumanMessage(content="You are about to take a HIGH-risk action: ..."),
    ]
    assert request_from_messages(msgs) == "cancel boston flight"
    obs = observations_from_messages(msgs)
    assert obs == ["Tool 'get_reservation' returned: res9 boston"]
