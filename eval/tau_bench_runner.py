"""τ-bench trial runner. Runs one agent on one task, returns a TrialResult."""
import copy

from langchain_core.messages import HumanMessage

import compass.tools.retail_db as db
from compass.calibration import calibrate
from compass.trajectory import extract_features
from eval.trial_store import TrialResult


def _serialize_trace(final_state: dict, condition: str) -> dict:
    """DESIGN.md's roll-your-own JSON trace: per-step decisions (Compass) and
    the message transcript. The DB row is the only artifact of a trial, so
    everything post-hoc analysis might need goes here."""
    messages = [
        {"role": getattr(m, "type", "unknown"), "content": str(getattr(m, "content", m))}
        for m in final_state.get("messages", [])
    ]
    steps = []
    if condition == "compass":
        for s in final_state.get("steps", []):
            steps.append({
                "reasoning": s.reasoning,
                "tool": s.action.tool,
                "args": s.action.args,
                "final_answer": s.action.final_answer,
                "confidence": s.confidence,
                "risk_level": s.risk_level,
            })
    return {"steps": steps, "messages": messages}


def _replay_success_probs(steps: list, shrink: bool = False) -> list[float]:
    """Recompute the calibrated success_prob the route edge saw at each step.

    `shrink` must match the aggregator variant the run actually gated on
    (agent_compass builds the graph with calibration_shrink=...); otherwise
    the stored success_probs diverge from the decisions they claim to record.

    Valid only while calibrate() is a pure function of the step history —
    if it ever grows state (e.g. a learned probe), success_prob must be
    captured inside the agent instead of replayed here.
    """
    return [
        calibrate(step.confidence, extract_features(steps[: i + 1]), shrink=shrink)
        for i, step in enumerate(steps)
    ]


def _changed_keys(before: dict, after: dict) -> list[str]:
    return sorted(
        key for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    )


def _grade_homemade(
    task: dict, expected: str, final_text: str, mutated_order_ids: list[str]
) -> bool:
    # Edge-case tasks (wrong order status) declare invariant_order_id: the
    # order must come out byte-for-byte identical. Message text is too easy
    # to paraphrase ("cannot be canceled" vs "cannot cancel") to grade
    # reliably, and it can't catch a wrong destructive action hiding behind
    # a reasonable-sounding reply.
    invariant_order_id = task.get("invariant_order_id")
    expected_order_state = task.get("expected_order_state")

    if invariant_order_id:
        return invariant_order_id not in mutated_order_ids
    if expected_order_state:
        # Mutating tasks (cancel/return/update): judge success on what the DB
        # actually shows afterward, not on whether the final message contains
        # the right word. A model that says "cancelled!" without calling the
        # tool must fail; a model that cancels it via unexpected wording must
        # pass.
        oid = expected_order_state["order_id"]
        after = db.ORDERS.get(oid, {})
        return all(
            after.get(key) == value
            for key, value in expected_order_state.items()
            if key != "order_id"
        )
    return expected.lower() in final_text.lower()


def run_trial(
    task: dict, agent, condition: str, model: str, calibration_shrink: bool = False
) -> TrialResult:
    """Run a single trial. Agent must be a compiled LangGraph graph.

    `calibration_shrink` must match the flag the agent graph was built with,
    so the replayed success_probs reflect the aggregator that actually gated
    the run.

    Two task flavours: homemade tasks grade on expected_order_state /
    invariant_order_id / substring; real τ-bench tasks (marked by a
    ground_truth_actions field) grade by replaying ground truth on the
    pristine dataset and comparing final orders+users state, plus checking
    expected_outputs in the final message — mirroring upstream τ-bench.
    """
    instruction = task["instruction"]
    expected = task.get("expected_outcome", "")  # homemade tasks only
    is_tau = "ground_truth_actions" in task

    # Snapshot the whole order DB so we can detect any destructive action the
    # trial took, not just changes to the graded order. Compound-failure rate
    # needs to distinguish "failed by doing nothing" from "failed by mutating".
    if is_tau:
        import compass.tools.tau_retail.db as tau_db
        orders_before = copy.deepcopy(tau_db.DATA["orders"])
        users_before = copy.deepcopy(tau_db.DATA["users"])
    else:
        orders_before = copy.deepcopy(db.ORDERS)

    init_state: dict = {
        "messages": [HumanMessage(content=instruction)],
        "steps": 0 if condition == "vanilla" else [],
        "abstained": False,
        "self_verify_count": 0,
        "verified_action": "",
    }

    final_state = agent.invoke(init_state)

    # last message is the agent's final output
    last_msg = final_state["messages"][-1]
    final_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if is_tau:
        from compass.tools.tau_retail import expected_state

        mutated_order_ids = _changed_keys(orders_before, tau_db.DATA["orders"]) + [
            f"user:{uid}" for uid in _changed_keys(users_before, tau_db.DATA["users"])
        ]
        want = expected_state(task["ground_truth_actions"])
        state_ok = (
            tau_db.DATA["orders"] == want["orders"]
            and tau_db.DATA["users"] == want["users"]
        )
        # Upstream τ-bench checks required outputs case-insensitively with
        # commas stripped (so "1,053.60" matches "1053.60").
        haystack = final_text.lower().replace(",", "")
        outputs_ok = all(
            out.lower() in haystack for out in task.get("expected_outputs", [])
        )
        success = state_ok and outputs_ok
    else:
        mutated_order_ids = _changed_keys(orders_before, db.ORDERS)
        success = _grade_homemade(task, expected, final_text, mutated_order_ids)

    # step count differs by agent type
    steps = (
        final_state["steps"]
        if condition == "vanilla"
        else len(final_state.get("steps", []))
    )

    # confidence scores only exist in Compass state
    if condition == "compass":
        compass_steps = final_state.get("steps", [])
        confidence_scores = [
            s.confidence for s in compass_steps if hasattr(s, "confidence")
        ]
        success_probs = _replay_success_probs(compass_steps, shrink=calibration_shrink)
        risk_levels = [
            s.risk_level for s in compass_steps if hasattr(s, "risk_level")
        ]
    else:
        confidence_scores = []
        success_probs = []
        risk_levels = []

    return TrialResult(
        task_id=task["id"],
        condition=condition,
        model=model,
        success=success,
        steps=steps,
        abstained=final_state.get("abstained", False),
        confidence_scores=confidence_scores,
        final_message=final_text,
        success_probs=success_probs,
        mutated_order_ids=mutated_order_ids,
        risk_levels=risk_levels,
        trace=_serialize_trace(final_state, condition),
    )
