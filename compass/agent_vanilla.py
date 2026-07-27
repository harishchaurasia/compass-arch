"""Baseline ReAct agent — no calibration, no abstention, hard step budget."""
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


def _append(left: list, right: list) -> list:
    return left + right


class VanillaState(TypedDict):
    messages: Annotated[list[BaseMessage], _append]
    steps: int


def build_vanilla_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = 20,
    policy: str | None = None,
) -> StateGraph:
    """policy, when given, is prepended as a system message (e.g. the τ-bench
    retail policy wiki that graded behaviour depends on)."""
    model_with_tools = model.bind_tools(tools)
    # handle_tool_errors turns a tool that raises (e.g. a KeyError on a malformed
    # arg) into an error observation the model can recover from, instead of
    # aborting the whole trial. The Compass agent's execute node already does this;
    # this keeps the baseline from losing a data point to a single bad call.
    tool_node = ToolNode(tools, handle_tool_errors=True)
    system = [SystemMessage(content=policy)] if policy else []

    def call_model(state: VanillaState) -> dict:
        response = model_with_tools.invoke(system + list(state["messages"]))
        return {"messages": [response], "steps": state["steps"] + 1}

    def should_continue(state: VanillaState) -> str:
        last = state["messages"][-1]
        if state["steps"] >= max_steps:
            return END
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(VanillaState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
