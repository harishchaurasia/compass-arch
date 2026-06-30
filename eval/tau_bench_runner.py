"""τ-bench trial runner. Runs one agent on one task, returns a TrialResult."""
from langchain_core.messages import AIMessage, HumanMessage

from eval.trial_store import TrialResult


def run_trial(task: dict, agent, condition: str, model: str) -> TrialResult:
    """Run a single trial. Agent must be a compiled LangGraph graph."""
    instruction = task["instruction"]
    expected = task["expected_outcome"]

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

    success = expected.lower() in final_text.lower()

    # step count differs by agent type
    steps = (
        final_state["steps"]
        if condition == "vanilla"
        else len(final_state.get("steps", []))
    )

    # confidence scores only exist in Compass state
    confidence_scores = [
        s.confidence for s in final_state.get("steps", [])
        if hasattr(s, "confidence")
    ] if condition == "compass" else []

    return TrialResult(
        task_id=task["id"],
        condition=condition,
        model=model,
        success=success,
        steps=steps,
        abstained=final_state.get("abstained", False),
        confidence_scores=confidence_scores,
        final_message=final_text,
    )
