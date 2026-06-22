"""Trajectory feature extraction from agent execution state."""
from pydantic import BaseModel, Field


class TrajectoryFeatures(BaseModel):
    tool_error_count: int = 0
    retry_count: int = 0
    plan_revision_count: int = 0
    semantic_consistency: float = Field(default=1.0, ge=0.0, le=1.0)
    trajectory_length_ratio: float = Field(default=1.0, gt=0.0)


def extract_features(steps: list) -> TrajectoryFeatures:
    raise NotImplementedError
