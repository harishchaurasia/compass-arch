"""One-call entry point for running a Compass agent on your own tools.

`build_compass_agent` returns a compiled LangGraph whose state has several
internal keys (`steps`, `self_verify_count`, `verified_action`, ...). This module
hides that setup so callers can wrap their own agent in a few lines:

    from compass import get_model, build_compass_agent, run

    agent = build_compass_agent(
        get_model("openai", "gpt-4o-mini"),
        tools=my_tools,
        tool_risk={"delete_file": "high", "list_files": "low"},
    )
    result = run(agent, "clean up the temp directory")
    if result.abstained:
        # Compass refused a high-risk action it was not confident enough to take.
        ...
"""

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from compass.calibration import calibrate
from compass.trajectory import extract_features


@dataclass
class GateResult:
    """Outcome of a single Compass run.

    `abstained` is the headline: True means the gate stopped the agent before a
    high-risk action it could not justify. `success_probs` is the calibrated
    probability the policy gated on at each step (baseline aggregator); compare
    it against the per-step `confidences` the model verbalized to see how much
    the gate discounted unearned certainty.
    """

    final_message: str
    abstained: bool
    steps: int
    confidences: list[float] = field(default_factory=list)
    success_probs: list[float] = field(default_factory=list)
    risk_levels: list[str] = field(default_factory=list)


def run(agent, instruction: str, *, calibration_shrink: bool = False) -> GateResult:
    """Run a Compass agent (from `build_compass_agent`) on one instruction.

    `calibration_shrink` must match the flag the agent graph was built with, so
    the replayed `success_probs` reflect the aggregator that actually gated the
    run. Returns a `GateResult`; nothing here touches the LangGraph state directly.

    Note: for a probe-mode agent (calibration="probe") the live gate uses the
    semantic probe, but the `success_probs` replayed here are the rule-based
    reference score (the probe needs an embedding backend and per-step context to
    recompute). The agent's decisions - abstained, steps - are authoritative; only
    this reported score is the rule-based stand-in in that mode.
    """
    final_state = agent.invoke(
        {
            "messages": [HumanMessage(content=instruction)],
            "steps": [],
            "abstained": False,
            "self_verify_count": 0,
            "verified_action": "",
        }
    )
    steps = final_state.get("steps", [])
    last = final_state["messages"][-1]
    confidences = [s.confidence for s in steps if hasattr(s, "confidence")]
    success_probs = [
        calibrate(
            step.confidence, extract_features(steps[: i + 1]), shrink=calibration_shrink
        )
        for i, step in enumerate(steps)
        if hasattr(step, "confidence")
    ]
    return GateResult(
        final_message=getattr(last, "content", str(last)),
        abstained=bool(final_state.get("abstained", False)),
        steps=len(steps),
        confidences=confidences,
        success_probs=success_probs,
        risk_levels=[s.risk_level for s in steps if hasattr(s, "risk_level")],
    )
