"""Compass calibrated agent: structured output + trajectory features + action policy."""
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from compass.policy import PolicyDecision, decide


def _append(left: list, right: list) -> list:
    return left + right


class CompassStep(BaseModel):
    reasoning: str
    action: dict  # {"tool": "...", "args": {...}} OR {"final_answer": "..."}
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]


class CompassState(TypedDict):
    messages: Annotated[list[BaseMessage], _append]
    steps: Annotated[list[CompassStep], _append]
    abstained: bool


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
        f"You are a calibrated agent. Available tools with EXACT parameter names:\n{tool_str}\n\n"
        "At each step output JSON with these exact fields:\n"
        "  reasoning: your thinking (string)\n"
        "  action: {\"tool\": \"<name>\", \"args\": {<exact params as listed above>}} to call a tool,\n"
        "          OR {\"final_answer\": \"<text>\"} when done\n"
        "  confidence: float 0.0–1.0, your belief this action is correct\n"
        "  risk_level: \"low\" | \"medium\" | \"high\" — cost of being wrong\n\n"
        "IMPORTANT: Use the EXACT parameter names shown above. Do not rename them."
    ))


def build_compass_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = 20,
) -> StateGraph:
    tool_map = {t.name: t for t in tools}
    structured_model = model.with_structured_output(CompassStep, method="function_calling")
    system = _system_prompt(tools)

    def plan(state: CompassState) -> dict:
        step: CompassStep = structured_model.invoke([system] + list(state["messages"]))
        return {"steps": [step]}

    def route(state: CompassState) -> str:
        if len(state["steps"]) >= max_steps:
            return "abstain"
        step = state["steps"][-1]
        if "final_answer" in step.action:
            return "finish"
        decision = decide(step.confidence, step.risk_level)
        return decision.value  # "execute" | "self_verify" | "abstain"

    def execute(state: CompassState) -> dict:
        step = state["steps"][-1]
        tool_name = step.action["tool"]
        tool_args = step.action.get("args", {})
        result = tool_map[tool_name].invoke(tool_args)
        return {"messages": [HumanMessage(content=f"Tool '{tool_name}' returned: {result}")]}

    def self_verify(state: CompassState) -> dict:
        step = state["steps"][-1]
        msg = HumanMessage(content=(
            f"Low confidence ({step.confidence:.2f}) detected on a {step.risk_level}-risk action. "
            "Please re-read the context carefully and verify your plan before proceeding."
        ))
        return {"messages": [msg]}

    def abstain(state: CompassState) -> dict:
        step = state["steps"][-1]
        msg = AIMessage(content=(
            f"ABSTAINING: confidence {step.confidence:.2f} is below threshold for "
            f"{step.risk_level}-risk action. Reasoning: {step.reasoning}"
        ))
        return {"messages": [msg], "abstained": True}

    def finish(state: CompassState) -> dict:
        step = state["steps"][-1]
        return {"messages": [AIMessage(content=step.action["final_answer"])]}

    graph = StateGraph(CompassState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("self_verify", self_verify)
    graph.add_node("abstain", abstain)
    graph.add_node("finish", finish)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route, {
        "execute": "execute",
        "self_verify": "self_verify",
        "abstain": "abstain",
        "finish": "finish",
    })
    graph.add_edge("execute", "plan")
    graph.add_edge("self_verify", "plan")
    graph.add_edge("abstain", END)
    graph.add_edge("finish", END)

    return graph.compile()
