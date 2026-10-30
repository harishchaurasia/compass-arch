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
    # Post-hoc annotation, never emitted by the model (default None in every
    # normal run). In confidence_source="logprob" mode the agent copies the
    # model's self-reported confidence here before overwriting `confidence` with
    # the logprob-derived value, so one run keeps BOTH signals for the paired
    # calibration comparison (FINDINGS §16). Optional so it changes nothing for
    # the verbalized path.
    verbalized_confidence: float | None = None
