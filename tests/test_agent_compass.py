"""Tests for the Compass calibrated agent.

FakeCompassModel supports with_structured_output() and returns CompassStep
objects in sequence — no real LLM calls.
"""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from compass.agent_compass import (
    CompassAction,
    CompassStep,
    build_compass_agent,
    salvage_step,
)

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


# ── salvage: recovering a step from unstructured local-model output ─────────────

def test_salvage_parses_compass_step_shape():
    """Model emitted the CompassStep shape directly as JSON in content."""
    content = (
        'Here is my step:\n```json\n'
        '{"reasoning": "look up user", '
        '"action": {"tool": "find_user", "args": {"email": "a@b.com"}}, '
        '"confidence": 0.7, "risk_level": "low"}\n```'
    )
    step = salvage_step(content)
    assert step is not None
    assert step.action.tool == "find_user"
    assert step.action.args == {"email": "a@b.com"}
    assert step.confidence == 0.7
    assert step.risk_level == "low"


def test_salvage_maps_function_call_envelope():
    """Real llama3.1:8b failure shape: a bare {name, parameters} tool envelope
    with no CompassStep fields — mapped onto an action with safe defaults."""
    content = (
        '{"name": "modify_pending_order_items", '
        '"parameters": {"order_id": "W2378156", "payment_method_id": "gift_card"}}'
    )
    step = salvage_step(content)
    assert step is not None
    assert step.action.tool == "modify_pending_order_items"
    assert step.action.args["order_id"] == "W2378156"
    assert 0.0 <= step.confidence <= 1.0
    assert step.risk_level in ("low", "medium", "high")


def test_salvage_grabs_first_object_amid_prose_and_extra_blobs():
    """llama sometimes free-writes several JSON blobs plus prose; take the first."""
    content = (
        '{"name": "list_all_product_types", "parameters": {}} \n\n'
        'The output of the tool call is:\n{"product_types": [{"name": "T-Shirt"}]}'
    )
    step = salvage_step(content)
    assert step is not None
    assert step.action.tool == "list_all_product_types"
    assert step.action.args == {}


def test_salvage_returns_none_on_no_json():
    assert salvage_step("I cannot help with that request.") is None


def test_plan_retries_with_nudge_when_output_is_unsalvageable():
    """First response is neither parseable nor salvageable (pure prose, no JSON);
    plan() must nudge and retry rather than crash, then accept the valid step."""

    class ProseThenValidModel:
        def __init__(self):
            self._calls = 0

        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            self._calls += 1
            if self._calls == 1:
                # unclosed/rambling content, nothing recoverable
                raw = AIMessage(content="I think the user wants to exchange, but //")
                return {"parsed": None, "raw": raw, "parsing_error": "2 validation errors"}
            step = CompassStep(
                reasoning="Retried cleanly.",
                action=CompassAction(final_answer="All set."),
                confidence=0.9,
                risk_level="low",
            )
            return {"parsed": step, "raw": None, "parsing_error": None}

    model = ProseThenValidModel()
    agent = build_compass_agent(model, [read_record])
    state = agent.invoke(_init_state("exchange my order"))
    assert model._calls == 2  # retried exactly once
    assert "All set." in state["messages"][-1].content


def test_plan_recovers_via_salvage_when_native_parse_is_none():
    """When with_structured_output returns parsed=None, plan() must fall back to
    salvaging the raw content instead of raising."""

    class UnparseableThenDoneModel:
        """First call: native parse fails but raw content holds an envelope.
        Second call: a clean final-answer step so the graph terminates."""

        def __init__(self):
            self._calls = 0

        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            self._calls += 1
            if self._calls == 1:
                raw = AIMessage(
                    content='{"name": "read_record", "parameters": {"record_id": "r1"}}'
                )
                return {"parsed": None, "raw": raw, "parsing_error": None}
            step = CompassStep(
                reasoning="done",
                action=CompassAction(final_answer="Read complete."),
                confidence=0.9,
                risk_level="low",
            )
            return {"parsed": step, "raw": None, "parsing_error": None}

    agent = build_compass_agent(UnparseableThenDoneModel(), [read_record])
    state = agent.invoke(_init_state("Read r1"))
    assert state["abstained"] is False
    assert "Read complete." in state["messages"][-1].content


def test_unknown_tool_feeds_back_instead_of_crashing():
    """A step that routes to EXECUTE but names a tool that doesn't exist (or
    None) must not raise KeyError — it feeds the error back so the model can
    recover on the next step."""
    steps = [
        # hallucinated tool name, high confidence + low risk → routes to execute
        CompassStep(
            reasoning="I'll use my analytics tool.",
            action=CompassAction(tool="predictive_analytics", args={"x": 1}),
            confidence=0.95,
            risk_level="low",
        ),
        # after the feed-back observation, model gives a valid final answer
        CompassStep(
            reasoning="No such tool; just answer.",
            action=CompassAction(final_answer="Here is your answer."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record])
    state = agent.invoke(_init_state("Do something"))

    assert state["abstained"] is False
    assert "Here is your answer." in state["messages"][-1].content
    all_content = " ".join(
        m.content for m in state["messages"] if hasattr(m, "content")
    ).lower()
    assert "not an available tool" in all_content


def test_tool_invocation_error_feeds_back_instead_of_crashing():
    """A real tool that raises on bad args (e.g. Pydantic validation) must feed
    the error back, not abort the trial."""

    @tool
    def strict_tool(count: int) -> str:
        """Needs an int count."""
        if not isinstance(count, int):
            raise ValueError("count must be an int")
        return f"counted {count}"

    steps = [
        # bad args → tool raises → feed back
        CompassStep(
            reasoning="Call it.",
            action=CompassAction(tool="strict_tool", args={"count": "not-an-int"}),
            confidence=0.95,
            risk_level="low",
        ),
        # model recovers with a final answer
        CompassStep(
            reasoning="Give up on the tool.",
            action=CompassAction(final_answer="Answered without the tool."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [strict_tool])
    state = agent.invoke(_init_state("count something"))

    assert state["abstained"] is False
    assert "Answered without the tool." in state["messages"][-1].content
    all_content = " ".join(
        m.content for m in state["messages"] if hasattr(m, "content")
    ).lower()
    assert "failed" in all_content


def test_none_tool_without_final_answer_feeds_back():
    """A no-op step (tool is None AND final_answer is None) must feed back, not
    crash on tool_map[None]."""
    steps = [
        CompassStep(
            reasoning="(empty action)",
            action=CompassAction(tool=None, args={}, final_answer=None),
            confidence=0.95,
            risk_level="low",
        ),
        CompassStep(
            reasoning="Now I'll answer.",
            action=CompassAction(final_answer="Done."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(FakeCompassModel(steps), [read_record])
    state = agent.invoke(_init_state("hi"))
    assert "Done." in state["messages"][-1].content


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


# ── verification ablation (verification=False) ────────────────────────────────

def test_ablation_skips_self_verify_and_executes_directly():
    """verification=False: a medium-risk step below T_MED would normally
    SELF_VERIFY. In the ablation it executes immediately instead."""
    called = []

    @tool
    def touch_record(record_id: str) -> str:
        """Update a record."""
        called.append(record_id)
        return "updated"

    steps = [
        CompassStep(
            reasoning="Not sure about this.",
            action=CompassAction(tool="touch_record", args={"record_id": "r2"}),
            confidence=0.4,  # below T_MED=0.6 → SELF_VERIFY when verification is on
            risk_level="medium",
        ),
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="Updated r2."),
            confidence=0.9,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(
        FakeCompassModel(steps), [touch_record], verification=False
    )
    state = agent.invoke(_init_state("Update r2"))

    assert called == ["r2"]  # executed without a verify detour
    assert state["abstained"] is False
    all_content = " ".join(
        m.content for m in state["messages"] if hasattr(m, "content")
    ).lower()
    assert "verify your plan" not in all_content


def test_ablation_skips_high_risk_confirm_pass():
    """verification=False: a high-risk step above T_HIGH executes on the first
    step, with no confirmation prompt injected."""
    called = []

    @tool
    def cancel_subscription(sub_id: str) -> str:
        """Cancel a subscription (irreversible)."""
        called.append(sub_id)
        return "cancelled"

    steps = [
        CompassStep(
            reasoning="User asked to cancel.",
            action=CompassAction(tool="cancel_subscription", args={"sub_id": "s1"}),
            confidence=0.9,  # above T_HIGH
            risk_level="high",
        ),
        CompassStep(
            reasoning="Done.",
            action=CompassAction(final_answer="Subscription cancelled."),
            confidence=0.95,
            risk_level="low",
        ),
    ]
    agent = build_compass_agent(
        FakeCompassModel(steps), [cancel_subscription], verification=False
    )
    state = agent.invoke(_init_state("Cancel subscription s1"))

    assert called == ["s1"]
    all_content = " ".join(
        m.content for m in state["messages"] if hasattr(m, "content")
    ).lower()
    assert "high-risk action" not in all_content  # no confirm prompt


def test_ablation_still_abstains_on_high_risk_low_confidence():
    """The ablation removes verification, not the gate: a high-risk step below
    T_HIGH must still abstain and never touch the tool."""
    called = []

    @tool
    def risky_delete(record_id: str) -> str:
        """Delete a record permanently."""
        called.append(record_id)
        return "deleted"

    steps = [
        CompassStep(
            reasoning="Probably the right record.",
            action=CompassAction(tool="risky_delete", args={"record_id": "r99"}),
            confidence=0.3,  # below T_HIGH → abstain
            risk_level="high",
        ),
    ]
    agent = build_compass_agent(
        FakeCompassModel(steps), [risky_delete], verification=False
    )
    state = agent.invoke(_init_state("Delete r99"))

    assert state["abstained"] is True
    assert called == []
