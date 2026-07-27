"""Compass — calibrated agent architecture for production reliability.

Public API:

    from compass import get_model, build_compass_agent, run

    agent = build_compass_agent(get_model("openai", "gpt-4o-mini"),
                                tools=my_tools, tool_risk={"delete": "high"})
    result = run(agent, "clean up the temp directory")
    result.abstained  # True if the gate stopped a high-risk action
"""

from compass.agent_compass import build_compass_agent
from compass.models import get_model
from compass.quickstart import GateResult, run

__version__ = "0.1.0"

__all__ = [
    "build_compass_agent",
    "get_model",
    "run",
    "GateResult",
    "__version__",
]
