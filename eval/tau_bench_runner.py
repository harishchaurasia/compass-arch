"""τ-bench trial runner. Runs one agent on one task, returns a TrialResult."""
import copy

from langchain_core.messages import HumanMessage

import compass.tools.retail_db as db
from compass.calibration import calibrate
from compass.trajectory import extract_features
from eval.trial_store import TrialResult


def _replay_success_probs(steps: list) -> list[float]:
    """Recompute the calibrated success_prob the route edge saw at each step.

    Valid only while calibrate() is a pure function of the step history —
    if it ever grows state (e.g. a learned probe), success_prob must be
    captured inside the agent instead of replayed here.
    """
    return [
        calibrate(step.confidence, extract_features(steps[: i + 1]))
        for i, step in enumerate(steps)
    ]


def run_trial(task: dict, agent, condition: str, model: str) -> TrialResult:
    """Run a single trial. Agent must be a compiled LangGraph graph."""
    instruction = task["instruction"]
    expected = task["expected_outcome"]

    # Snapshot the whole order DB so we can detect any destructive action the
    # trial took, not just changes to the graded order. Compound-failure rate
    # needs to distinguish "failed by doing nothing" from "failed by mutating".
    orders_before = copy.deepcopy(db.ORDERS)

    init_state: dict = {
        "messages": [HumanMessage(content=instruction)],
        "steps": 0 if condition == "vanilla" else [],
        "abstained": False,
        "self_verify_count": 0,
    }

    final_state = agent.invoke(init_state)

    # last message is the agent's final output
    last_msg = final_state["messages"][-1]
    final_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    all_order_ids = set(orders_before) | set(db.ORDERS)
    mutated_order_ids = sorted(
        oid for oid in all_order_ids
        if orders_before.get(oid) != db.ORDERS.get(oid)
    )

    # Edge-case tasks (wrong order status) declare invariant_order_id: the
    # order must come out byte-for-byte identical. Message text is too easy
    # to paraphrase ("cannot be canceled" vs "cannot cancel") to grade
    # reliably, and it can't catch a wrong destructive action hiding behind
    # a reasonable-sounding reply.
    invariant_order_id = task.get("invariant_order_id")
    expected_order_state = task.get("expected_order_state")

    if invariant_order_id:
        success = invariant_order_id not in mutated_order_ids
    elif expected_order_state:
        # Mutating tasks (cancel/return/update): judge success on what the DB
        # actually shows afterward, not on whether the final message contains
        # the right word. A model that says "cancelled!" without calling the
        # tool must fail; a model that cancels it via unexpected wording must
        # pass.
        oid = expected_order_state["order_id"]
        after = db.ORDERS.get(oid, {})
        success = all(
            after.get(key) == value
            for key, value in expected_order_state.items()
            if key != "order_id"
        )
    else:
        success = expected.lower() in final_text.lower()

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
        success_probs = _replay_success_probs(compass_steps)
    else:
        confidence_scores = []
        success_probs = []

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
    )
