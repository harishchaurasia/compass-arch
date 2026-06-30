"""Trajectory feature extraction from agent step history.

Reads the list of CompassStep objects accumulated so far and extracts
behavioural signals that verbalized confidence alone cannot capture:
how long the agent has been running, whether it is looping, and how
long since it last tried something new.
"""
from dataclasses import dataclass

from compass.schemas import CompassStep

_OSCILLATION_WINDOW = 4  # look at last N tool steps to detect loops


@dataclass
class TrajectoryFeatures:
    step_count: int           # total steps including final_answer
    is_oscillating: bool      # last 4 tool steps use ≤ 2 unique tools
    unique_tools_used: int    # distinct tool names called so far
    steps_since_new_tool: int # trailing count of steps that repeated a seen tool


def extract_features(steps: list[CompassStep]) -> TrajectoryFeatures:
    step_count = len(steps)

    # Only tool-calling steps contribute to tool metrics
    tool_names = [s.action.tool for s in steps if s.action.tool is not None]

    unique_tools_used = len(set(tool_names))

    # Oscillation: need at least 4 tool steps; last 4 use ≤ 2 unique tools
    if len(tool_names) >= _OSCILLATION_WINDOW:
        last_window = tool_names[-_OSCILLATION_WINDOW:]
        is_oscillating = len(set(last_window)) <= 2
    else:
        is_oscillating = False

    # Steps since new tool: count trailing tool steps that called an already-seen tool
    if not tool_names:
        steps_since_new_tool = 0
    else:
        seen: set[str] = set()
        was_new: list[bool] = []
        for name in tool_names:
            was_new.append(name not in seen)
            seen.add(name)

        steps_since_new_tool = 0
        for flag in reversed(was_new):
            if not flag:
                steps_since_new_tool += 1
            else:
                break

    return TrajectoryFeatures(
        step_count=step_count,
        is_oscillating=is_oscillating,
        unique_tools_used=unique_tools_used,
        steps_since_new_tool=steps_since_new_tool,
    )
