"""Compass calibrated agent: structured output + trajectory features + action policy."""
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from compass.calibration import calibrate
from compass.policy import PolicyDecision, decide, max_risk
from compass.schemas import CompassAction, CompassStep  # noqa: F401 — re-exported for callers
from compass.trajectory import extract_features

MAX_SELF_VERIFY = 2  # consecutive SELF_VERIFYs before escalating to ABSTAIN


def _append(left: list, right: list) -> list:
    return left + right


class CompassState(TypedDict):
    messages: Annotated[list[BaseMessage], _append]
    steps: Annotated[list[CompassStep], _append]
    abstained: bool
    self_verify_count: int
    high_risk_verified: bool  # last high-risk action passed the confirm step


def _system_prompt(tools: list[BaseTool]) -> SystemMessage:
    tool_lines = []
    for t in tools:
        params = ", ".join(
            f"{name}: {info.get('type', 'string')}"
            for name, info in t.args.items()
        )
        tool_lines.append(f"  - {t.name}({params}): {t.description}")
    tool_str = "\n".join(tool_lines)
    return SystemMessage(content=(
        f"You are a calibrated retail customer service agent.\n\n"
        f"Available tools (use EXACT parameter names shown):\n{tool_str}\n\n"
        "For each step fill ALL fields at the TOP LEVEL — never nest confidence or risk_level inside action:\n"
        "  reasoning        — explain your thinking\n"
        "  action.tool      — tool name to call (leave null if giving a final answer)\n"
        "  action.args      — exact tool parameters as a dict (leave empty if giving a final answer)\n"
        "  action.final_answer — your response text (use instead of tool when task is complete)\n"
        "  confidence       — float 0.0–1.0 (TOP-LEVEL field, NOT inside action)\n"
        "  risk_level       — \"low\" | \"medium\" | \"high\" (TOP-LEVEL field, NOT inside action)\n\n"
        "IMPORTANT: confidence and risk_level are SIBLINGS of action, not children of it.\n"
        "IMPORTANT: Use the EXACT parameter names listed above. Do not invent or rename them."
    ))


def build_compass_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = 20,
    tool_risk: dict[str, str] | None = None,
) -> StateGraph:
    """tool_risk maps tool name → static risk class ('low'|'medium'|'high').
    Effective risk is max(model's verbalized risk_level, tool class), so an
    under-labelled destructive tool is still gated as high."""
    tool_risk = tool_risk or {}
    tool_map = {t.name: t for t in tools}
    structured_model = model.with_structured_output(
        CompassStep, method="function_calling", strict=False, include_raw=True
    )
    system = _system_prompt(tools)

    def plan(state: CompassState) -> dict:
        result = structured_model.invoke([system] + list(state["messages"]))
        if result["parsed"] is None:
            raise ValueError(
                f"Model returned unparseable output.\n"
                f"Raw response: {result['raw']}\n"
                f"Parse error: {result['parsing_error']}"
            )
        step: CompassStep = result["parsed"]
        return {"steps": [step]}

    def _effective_risk(step: CompassStep) -> str:
        return max_risk(step.risk_level, tool_risk.get(step.action.tool, "low"))

    def route(state: CompassState) -> str:
        if len(state["steps"]) >= max_steps:
            return "abstain"
        step = state["steps"][-1]
        if step.action.final_answer is not None:
            return "finish"
        features = extract_features(state["steps"])
        success_prob = calibrate(step.confidence, features)
        risk = _effective_risk(step)
        decision = decide(success_prob, risk)
        # Escalate to abstain if SELF_VERIFY keeps firing without any execution
        if decision == PolicyDecision.SELF_VERIFY and state["self_verify_count"] >= MAX_SELF_VERIFY:
            return "abstain"
        # High-risk executes need an explicit verification pass first
        # (DESIGN.md Table 1): the model must re-confirm intent before acting.
        if (
            decision == PolicyDecision.EXECUTE
            and risk == "high"
            and not state.get("high_risk_verified", False)
        ):
            return "confirm"
        return decision.value  # "execute" | "self_verify" | "abstain"

    def execute(state: CompassState) -> dict:
        step = state["steps"][-1]
        tool_name = step.action.tool
        tool_args = step.action.args
        result = tool_map[tool_name].invoke(tool_args)
        return {
            "messages": [HumanMessage(content=f"Tool '{tool_name}' returned: {result}")],
            "self_verify_count": 0,  # real progress — reset the streak
            "high_risk_verified": False,  # each high-risk action needs its own confirm
        }

    def confirm(state: CompassState) -> dict:
        step = state["steps"][-1]
        msg = HumanMessage(content=(
            f"You are about to take a HIGH-risk action: "
            f"{step.action.tool}({step.action.args}). "
            "Re-read the user's original request and confirm this action is "
            "exactly what they asked for. If it is, repeat the same action; "
            "if not, change course."
        ))
        return {"messages": [msg], "high_risk_verified": True}

    def self_verify(state: CompassState) -> dict:
        step = state["steps"][-1]
        msg = HumanMessage(content=(
            f"Low confidence ({step.confidence:.2f}) detected on a {step.risk_level}-risk action. "
            "Please re-read the context carefully and verify your plan before proceeding."
        ))
        return {"messages": [msg], "self_verify_count": state["self_verify_count"] + 1}

    def abstain(state: CompassState) -> dict:
        step = state["steps"][-1]
        risk = _effective_risk(step) if step.action.tool else step.risk_level
        msg = AIMessage(content=(
            f"ABSTAINING: confidence {step.confidence:.2f} is below threshold for "
            f"{risk}-risk action. Reasoning: {step.reasoning}"
        ))
        return {"messages": [msg], "abstained": True}

    def finish(state: CompassState) -> dict:
        step = state["steps"][-1]
        return {"messages": [AIMessage(content=step.action.final_answer)]}

    graph = StateGraph(CompassState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("confirm", confirm)
    graph.add_node("self_verify", self_verify)
    graph.add_node("abstain", abstain)
    graph.add_node("finish", finish)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route, {
        "execute": "execute",
        "confirm": "confirm",
        "self_verify": "self_verify",
        "abstain": "abstain",
        "finish": "finish",
    })
    graph.add_edge("execute", "plan")
    graph.add_edge("confirm", "plan")
    graph.add_edge("self_verify", "plan")
    graph.add_edge("abstain", END)
    graph.add_edge("finish", END)

    return graph.compile()
