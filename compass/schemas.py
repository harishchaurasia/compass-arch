"""Shared Pydantic schemas used across the Compass pipeline.

Lives here (not in agent_compass) so that trajectory.py and calibration.py
can import CompassStep without creating a circular dependency with agent_compass.
"""
from typing import Literal

from pydantic import BaseModel, Field


class CompassAction(BaseModel):
    tool: str | None = None
    args: dict = {}
    final_answer: str | None = None


class CompassStep(BaseModel):
    reasoning: str
    action: CompassAction
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
