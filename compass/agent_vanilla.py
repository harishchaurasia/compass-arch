"""Baseline ReAct agent — no calibration, no abstention, hard step budget."""
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


class VanillaState(TypedDict):
    messages: Annotated[list[BaseMessage], ...]
    steps: int


def build_vanilla_agent(model: BaseChatModel, tools: list, max_steps: int = 20) -> StateGraph:
    raise NotImplementedError
