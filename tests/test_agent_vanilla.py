"""Tests for the vanilla ReAct agent.

We use a FakeModel instead of a real LLM so tests never hit an API.
FakeModel returns preset AIMessage responses in order.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from compass.agent_vanilla import build_vanilla_agent


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


class FakeModel:
    """Returns preset AIMessage responses in sequence. No API calls."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = iter(responses)

    def bind_tools(self, tools):
        return self  # swallow tools, return self so chaining works

    def invoke(self, messages):
        return next(self._responses)


# ── tests ────────────────────────────────────────────────────────────────────

def test_build_returns_runnable():
    """build_vanilla_agent should return something we can call .invoke() on."""
    model = FakeModel([AIMessage(content="done")])
    agent = build_vanilla_agent(model, [get_weather])
    assert hasattr(agent, "invoke")


def test_agent_returns_final_answer_without_tools():
    """When the model gives a plain answer, the agent should stop immediately."""
    model = FakeModel([AIMessage(content="The answer is 42.")])
    agent = build_vanilla_agent(model, [get_weather])

    state = agent.invoke({"messages": [HumanMessage(content="What is 6x7?")], "steps": 0})

    assert state["messages"][-1].content == "The answer is 42."
    assert state["steps"] == 1


def test_agent_calls_tool_then_returns_answer():
    """When the model requests a tool, the agent should run it and continue."""
    tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "get_weather",
            "args": {"city": "Paris"},
            "id": "call_1",
            "type": "tool_call",
        }],
    )
    final = AIMessage(content="It is sunny in Paris.")
    model = FakeModel([tool_call, final])
    agent = build_vanilla_agent(model, [get_weather])

    state = agent.invoke({"messages": [HumanMessage(content="Weather in Paris?")], "steps": 0})

    # should have a ToolMessage in the history (proof the tool actually ran)
    assert any(isinstance(m, ToolMessage) for m in state["messages"])
    assert state["messages"][-1].content == "It is sunny in Paris."
    assert state["steps"] == 2


def test_agent_stops_at_max_steps():
    """Agent must not run forever — it stops when steps >= max_steps."""
    # model always wants to call a tool, so without a budget it would loop forever
    tool_calls = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_weather",
                "args": {"city": "Paris"},
                "id": f"call_{i}",
                "type": "tool_call",
            }],
        )
        for i in range(10)
    ]
    model = FakeModel(tool_calls)
    agent = build_vanilla_agent(model, [get_weather], max_steps=2)

    state = agent.invoke({"messages": [HumanMessage(content="Go forever")], "steps": 0})

    assert state["steps"] >= 2
