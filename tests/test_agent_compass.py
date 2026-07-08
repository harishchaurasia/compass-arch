"""Tests for the Compass calibrated agent.

FakeCompassModel supports with_structured_output() and returns CompassStep
objects in sequence — no real LLM calls.
"""
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from compass.agent_compass import CompassAction, CompassStep, build_compass_agent

# ── fake model ────────────────────────────────────────────────────────────────

class FakeCompassModel:
    """Yields CompassStep objects from with_structured_output().invoke().

    Returns include_raw=True style dict so it matches what build_compass_agent expects.
    """

    def __init__(self, steps: list[CompassStep]):
        self._iter = iter(steps)

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages) -> dict:
        step = next(self._iter)
        return {"parsed": step, "raw": None, "parsing_error": None}


# ── test tool ─────────────────────────────────────────────────────────────────

@tool
def delete_record(record_id: str) -> str:
    """Delete a record permanently."""
    return f"Record {record_id} deleted."


@tool
def read_record(record_id: str) -> str:
    """Read a record (read-only, safe)."""
    return f"Record {record_id}: some data."


# ── tests ─────────────────────────────────────────────────────────────────────

def _init_state(content: str) -> dict:
    return {
        "messages": [HumanMessage(content=content)],
        "steps": [],
        "abstained": False,
        "self_verify_count": 0,
    }


def test_low_risk_tool_executes():
    """Low risk action should always be executed regardless of confidence."""
    steps = [
        CompassStep(
            reasoning="Safe read operation.",
            action=CompassAction(tool="read_record", args={"record_id": "r1"}),
            confidence=0.3,  # low confidence, but low risk → still execute
            risk_level="low",
        ),
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="Read complete."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record, delete_record])
    state = agent.invoke(_init_state("Read r1"))

    assert state["abstained"] is False
    assert "Read complete." in state["messages"][-1].content


def test_high_risk_low_confidence_abstains():
    """High risk + low confidence → agent must abstain without calling the tool."""
    called = []

    @tool
    def risky_delete(record_id: str) -> str:
        """Delete permanently."""
        called.append(record_id)
        return "deleted"

    steps = [
        CompassStep(
            reasoning="About to delete something important.",
            action=CompassAction(tool="risky_delete", args={"record_id": "r99"}),
            confidence=0.5,  # below T_HIGH=0.8 → abstain
            risk_level="high",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [risky_delete])
    state = agent.invoke(_init_state("Delete r99"))

    assert state["abstained"] is True
    assert called == []  # tool was NEVER called


def test_medium_risk_low_confidence_self_verifies_then_executes():
    """Medium risk + low confidence → self-verify, then if confidence improves → execute."""
    steps = [
        # step 1: low confidence → self_verify
        CompassStep(
            reasoning="Not sure about this.",
            action=CompassAction(tool="read_record", args={"record_id": "r2"}),
            confidence=0.4,  # below T_MED=0.6 → self_verify
            risk_level="medium",
        ),
        # step 2: after self-verify prompt, confidence improves → execute
        CompassStep(
            reasoning="Re-read context, now confident.",
            action=CompassAction(tool="read_record", args={"record_id": "r2"}),
            confidence=0.75,  # above T_MED=0.6 → execute
            risk_level="medium",
        ),
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="Read r2 successfully."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record])
    state = agent.invoke(_init_state("Read r2"))

    assert state["abstained"] is False
    assert "Read r2 successfully." in state["messages"][-1].content
    all_content = " ".join(m.content for m in state["messages"] if hasattr(m, "content"))
    assert "verify" in all_content.lower() or "confidence" in all_content.lower()


def test_final_answer_ends_without_tool():
    """If first step is a final_answer, agent should return it without calling any tool."""
    steps = [
        CompassStep(
            reasoning="I already know the answer.",
            action=CompassAction(final_answer="The order status is pending."),
            confidence=0.95,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record])
    state = agent.invoke(_init_state("What is the order status?"))

    assert "pending" in state["messages"][-1].content
    assert state["abstained"] is False


def test_two_consecutive_self_verifies_escalate_to_abstain():
    """After MAX_SELF_VERIFY=2 consecutive SELF_VERIFYs with no EXECUTE, route to abstain."""
    called = []

    @tool
    def lookup(record_id: str) -> str:
        """Look up a record."""
        called.append(record_id)
        return "data"

    # Three identical low-confidence steps — the agent keeps not gaining confidence
    steps = [
        CompassStep(
            reasoning="Uncertain.",
            action=CompassAction(tool="lookup", args={"record_id": "r1"}),
            confidence=0.3,  # below T_MED → self_verify (self_verify_count → 1)
            risk_level="medium",
        ),
        CompassStep(
            reasoning="Still uncertain.",
            action=CompassAction(tool="lookup", args={"record_id": "r1"}),
            confidence=0.3,  # still low → self_verify_count = 1, still < 2 → self_verify again (→ 2)
            risk_level="medium",
        ),
        CompassStep(
            reasoning="Still stuck.",
            action=CompassAction(tool="lookup", args={"record_id": "r1"}),
            confidence=0.3,  # self_verify_count = 2 ≥ 2 → escalate to abstain
            risk_level="medium",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [lookup])
    state = agent.invoke(_init_state("Look up r1"))

    assert state["abstained"] is True
    assert called == []  # tool was never executed


def test_tool_risk_class_floors_model_risk_label():
    """A destructive tool statically classed 'high' must be gated as high even
    when the model labels the step 'low' — verbalized risk is unreliable."""
    called = []

    @tool
    def wipe_account(record_id: str) -> str:
        """Wipe an account permanently."""
        called.append(record_id)
        return "wiped"

    steps = [
        CompassStep(
            reasoning="Routine cleanup.",
            action=CompassAction(tool="wipe_account", args={"record_id": "r7"}),
            confidence=0.5,  # below T_HIGH=0.8; effective risk high → abstain
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(
        FakeCompassModel(steps),
        [wipe_account],
        tool_risk={"wipe_account": "high"},
    )
    state = agent.invoke(_init_state("Clean up account r7"))

    assert state["abstained"] is True
    assert called == []


def test_high_risk_high_confidence_requires_confirmation_before_execute():
    """DESIGN.md Table 1: High risk + conf ≥ T_high → execute WITH an explicit
    verification step. The tool runs only after the model re-affirms the action."""
    called = []

    @tool
    def cancel_subscription(sub_id: str) -> str:
        """Cancel a subscription (irreversible)."""
        called.append(sub_id)
        return "cancelled"

    plan_step = CompassStep(
        reasoning="User asked to cancel.",
        action=CompassAction(tool="cancel_subscription", args={"sub_id": "s1"}),
        confidence=0.9,  # above T_HIGH → execute path, but must confirm first
        risk_level="high",
    )
    steps = [
        plan_step,
        plan_step.model_copy(),  # re-affirms the same action after the confirm prompt
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="Subscription cancelled."),
            confidence=0.95,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [cancel_subscription])
    state = agent.invoke(_init_state("Cancel subscription s1"))

    assert called == ["s1"]  # executed exactly once, after re-affirmation
    all_content = " ".join(
        m.content for m in state["messages"] if hasattr(m, "content")
    ).lower()
    assert "high-risk" in all_content  # the confirmation prompt was injected


def test_confirmation_lets_model_back_out_of_wrong_action():
    """If re-reading the request makes the model realize the action is wrong,
    it can revise to a final answer and the tool is never called."""
    called = []

    @tool
    def cancel_subscription(sub_id: str) -> str:
        """Cancel a subscription (irreversible)."""
        called.append(sub_id)
        return "cancelled"

    steps = [
        CompassStep(
            reasoning="I'll cancel the subscription.",
            action=CompassAction(tool="cancel_subscription", args={"sub_id": "s1"}),
            confidence=0.9,
            risk_level="high",
        ),
        # after the confirm prompt: the user actually asked for a refund, not a cancel
        CompassStep(
            reasoning="Re-read the request — they asked about a refund, not cancellation.",
            action=CompassAction(final_answer="I can't do that; you asked about a refund."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [cancel_subscription])
    state = agent.invoke(_init_state("Refund subscription s1"))

    assert called == []
    assert "refund" in state["messages"][-1].content


def test_changed_course_high_risk_action_needs_its_own_confirmation():
    """Pilot finding (tau_retail_104): after a confirm prompt the model may
    change course to a DIFFERENT high-risk action. That new action must get
    its own confirm — the earlier confirmation must not unlock it."""
    called = []

    @tool
    def cancel_subscription(sub_id: str) -> str:
        """Cancel a subscription (irreversible)."""
        called.append(("cancel", sub_id))
        return "cancelled"

    @tool
    def delete_account(account_id: str) -> str:
        """Delete an account permanently."""
        called.append(("delete", account_id))
        return "deleted"

    steps = [
        CompassStep(
            reasoning="I'll cancel the subscription.",
            action=CompassAction(tool="cancel_subscription", args={"sub_id": "s1"}),
            confidence=0.9,
            risk_level="high",
        ),
        # after the confirm prompt: changes course to a different high-risk action
        CompassStep(
            reasoning="Actually the whole account should go.",
            action=CompassAction(tool="delete_account", args={"account_id": "a1"}),
            confidence=0.9,
            risk_level="high",
        ),
        # at delete_account's OWN confirm prompt, the model backs out entirely.
        # If the earlier confirm wrongly unlocked delete_account, it has
        # already executed and this step is never reached in time.
        CompassStep(
            reasoning="Re-read the request — no destructive action was asked for.",
            action=CompassAction(final_answer="No changes made."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(
        FakeCompassModel(steps), [cancel_subscription, delete_account]
    )
    state = agent.invoke(_init_state("Delete my account a1"))

    # nothing may execute: neither action ever received a re-affirmation
    assert called == []
    confirm_prompts = [
        m.content for m in state["messages"]
        if hasattr(m, "content") and "HIGH-risk action" in str(m.content)
    ]
    assert len(confirm_prompts) == 2  # one per distinct high-risk action
    assert "delete_account" in confirm_prompts[1]


def test_same_action_different_args_needs_new_confirmation():
    """A confirm for cancel(s1) must not unlock cancel(s2)."""
    called = []

    @tool
    def cancel_subscription(sub_id: str) -> str:
        """Cancel a subscription (irreversible)."""
        called.append(sub_id)
        return "cancelled"

    steps = [
        CompassStep(
            reasoning="Cancel s1.",
            action=CompassAction(tool="cancel_subscription", args={"sub_id": "s1"}),
            confidence=0.9,
            risk_level="high",
        ),
        # after confirm: same tool, different target — must be re-confirmed
        CompassStep(
            reasoning="It was s2, not s1.",
            action=CompassAction(tool="cancel_subscription", args={"sub_id": "s2"}),
            confidence=0.9,
            risk_level="high",
        ),
        # backs out at s2's own confirm prompt — nothing should have run
        CompassStep(
            reasoning="On reflection, no cancellation was requested.",
            action=CompassAction(final_answer="No changes made."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [cancel_subscription])
    agent.invoke(_init_state("Cancel subscription s2"))

    assert called == []


def test_execute_resets_self_verify_count():
    """A successful EXECUTE resets the counter so a later SELF_VERIFY gets a fresh start."""
    steps = [
        # self-verify #1 (count → 1)
        CompassStep(
            reasoning="Uncertain.",
            action=CompassAction(tool="read_record", args={"record_id": "r3"}),
            confidence=0.3,
            risk_level="medium",
        ),
        # confidence recovers → execute (count resets to 0)
        CompassStep(
            reasoning="Confident now.",
            action=CompassAction(tool="read_record", args={"record_id": "r3"}),
            confidence=0.8,
            risk_level="medium",
        ),
        # self-verify #1 again after reset (count → 1, NOT 2)
        CompassStep(
            reasoning="Uncertain again.",
            action=CompassAction(tool="read_record", args={"record_id": "r3"}),
            confidence=0.3,
            risk_level="medium",
        ),
        # confidence recovers → execute and finish
        CompassStep(
            reasoning="Confident.",
            action=CompassAction(tool="read_record", args={"record_id": "r3"}),
            confidence=0.8,
            risk_level="medium",
        ),
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="All done."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record])
    state = agent.invoke(_init_state("Read r3"))

    assert state["abstained"] is False
    assert "All done." in state["messages"][-1].content


def test_policy_included_in_system_prompt():
    """τ-bench tasks require the retail policy wiki in the agent's context."""
    seen = []

    class CapturingModel:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            seen.append(list(messages))
            step = CompassStep(
                reasoning="done",
                action=CompassAction(final_answer="ok"),
                confidence=0.9,
                risk_level="low",
            )
            return {"parsed": step, "raw": None, "parsing_error": None}

    agent = build_compass_agent(CapturingModel(), [], policy="POLICY: authenticate first.")
    agent.invoke(_init_state("hi"))

    assert "authenticate first" in seen[0][0].content
