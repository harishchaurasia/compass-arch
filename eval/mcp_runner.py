"""Custom MCP task suite runner (Phase 3): filesystem cascading-failure tasks.

Runs one agent on one task against the purpose-built filesystem MCP server,
grading destructive-while-wrong failures deterministically (see
compass.mcp.grading). Mirrors eval.tau_bench_runner but the "database" is the
MCP server's world, reset per trial via the bridge's control surface.
"""
from pathlib import Path

from langchain_core.messages import HumanMessage

from compass.mcp.grading import grade
from eval.tau_bench_runner import _replay_success_probs, _serialize_trace
from eval.trial_store import TrialResult, save_trial

RESULTS_DB = Path(__file__).parent.parent / "results" / "trials.db"


def run_mcp_trial(
    task: dict,
    agent,
    condition: str,
    model: str,
    server,
    seed: dict[str, str],
    calibration_shrink: bool = False,
) -> TrialResult:
    """Run one trial. `server` is an MCPToolServer; `seed` is the world to reset
    to before the agent runs. Task ids are stored as ``mcp_fs_<id>``."""
    server.reset(seed)

    init_state: dict = {
        "messages": [HumanMessage(content=task["instruction"])],
        "steps": 0 if condition == "vanilla" else [],
        "abstained": False,
        "self_verify_count": 0,
        "verified_action": "",
    }
    final_state = agent.invoke(init_state)

    last_msg = final_state["messages"][-1]
    final_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    final_world = server.dump()
    success, mutated = grade(task, seed, final_world, final_text)

    steps = (
        final_state["steps"]
        if condition == "vanilla"
        else len(final_state.get("steps", []))
    )
    if condition == "compass":
        compass_steps = final_state.get("steps", [])
        confidence_scores = [s.confidence for s in compass_steps if hasattr(s, "confidence")]
        success_probs = _replay_success_probs(compass_steps, shrink=calibration_shrink)
        risk_levels = [s.risk_level for s in compass_steps if hasattr(s, "risk_level")]
    else:
        confidence_scores = success_probs = risk_levels = []

    return TrialResult(
        task_id=f"mcp_fs_{task['id']}",
        condition=condition,
        model=model,
        success=success,
        steps=steps,
        abstained=final_state.get("abstained", False),
        confidence_scores=confidence_scores,
        final_message=final_text,
        success_probs=success_probs,
        mutated_order_ids=mutated,  # reused column: filesystem paths mutated
        risk_levels=risk_levels,
        trace=_serialize_trace(final_state, condition),
    )


def run_mcp_suite(
    tasks: list[dict],
    agents: dict,
    server,
    seed: dict[str, str],
    model: str,
    db_path: Path = RESULTS_DB,
    calibration_shrink: bool = False,
) -> None:
    """Run every (task × condition) in `agents` ({condition: compiled graph}) and
    persist each TrialResult."""
    for task in tasks:
        for condition, agent in agents.items():
            result = run_mcp_trial(
                task, agent, condition, model, server, seed,
                calibration_shrink=calibration_shrink,
            )
            save_trial(result, db_path)
