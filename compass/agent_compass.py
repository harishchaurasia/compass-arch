"""Compass calibrated agent: structured output + trajectory features + action policy."""
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field


class CompassStep(BaseModel):
    reasoning: str
    action: dict
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]


class CompassState(TypedDict):
    messages: Annotated[list[BaseMessage], ...]
    steps: list[CompassStep]
    abstained: bool


def build_compass_agent(model: BaseChatModel, tools: list, max_steps: int = 20) -> StateGraph:
    raise NotImplementedError
