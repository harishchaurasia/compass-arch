"""Compass calibrated agent: structured output + trajectory features + action policy."""
import json
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from compass.calibration import calibrate
from compass.policy import PolicyDecision, decide, max_risk
from compass.schemas import CompassAction, CompassStep  # noqa: F401 — re-exported for callers
from compass.trajectory import extract_features

MAX_SELF_VERIFY = 2  # consecutive SELF_VERIFYs before escalating to ABSTAIN
MAX_PARSE_RETRIES = 2  # extra plan() attempts with a corrective nudge before giving up

_PARSE_RETRY_NUDGE = (
    "Your previous response was not a valid step. Reply with ONLY a single JSON "
    "object containing ALL of these fields: reasoning (string), action (object "
    'with tool, args, final_answer), confidence (number 0.0-1.0), and risk_level '
    '("low" | "medium" | "high"). No prose, no comments, no trailing text.'
)


def _append(left: list, right: list) -> list:
    return left + right


def _first_json_object(text: str) -> dict | None:
    """Return the first balanced top-level JSON object in `text`, or None.

    Local models (Ollama) frequently wrap their JSON in prose or markdown
    fences and sometimes emit several blobs; brace-matching grabs the first
    complete object and ignores the rest. String-aware so braces inside string
    literals don't throw off the depth count."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def salvage_step(content: str) -> CompassStep | None:
    """Best-effort CompassStep from free-form model output when native
    structured parsing returned None. Handles the two shapes local models
    emit: the CompassStep shape directly, and the OpenAI-style function
    envelope {"name": ..., "parameters"/"arguments": {...}}."""
    blob = _first_json_object(content)
    if blob is None:
        return None
    # Shape 1: already CompassStep-shaped.
    try:
        return CompassStep.model_validate(blob)
    except ValidationError:
        pass
    # Shape 2: a bare function-call envelope — map it onto an action.
    name = blob.get("name")
    if isinstance(name, str) and name:
        raw_args: Any = blob.get("parameters", blob.get("arguments", {}))
        args = raw_args if isinstance(raw_args, dict) else {}
        try:
            return CompassStep(
                reasoning=str(blob.get("reasoning", "(recovered from tool-call envelope)")),
                action=CompassAction(tool=name, args=args),
                confidence=float(blob.get("confidence", 0.5)),
                risk_level=blob.get("risk_level") if blob.get("risk_level") in ("low", "medium", "high") else "medium",
            )
        except (ValidationError, ValueError, TypeError):
            return None
    return None


class CompassState(TypedDict):
    messages: Annotated[list[BaseMessage], _append]
    steps: Annotated[list[CompassStep], _append]
    abstained: bool
    self_verify_count: int
    verified_action: str  # fingerprint of the action the confirm step covered ("" = none)


def _action_fingerprint(step: CompassStep) -> str:
    """Identity of a proposed tool call. A confirm only unlocks THIS exact
    action — a changed-course action (different tool OR different args, e.g.
    a different order id) must earn its own confirm."""
    return json.dumps({"tool": step.action.tool, "args": step.action.args}, sort_keys=True)


def _system_prompt(tools: list[BaseTool], policy: str | None = None) -> SystemMessage:
    tool_lines = []
    for t in tools:
        params = ", ".join(
            f"{name}: {info.get('type', 'string')}"
            for name, info in t.args.items()
        )
        tool_lines.append(f"  - {t.name}({params}): {t.description}")
    tool_str = "\n".join(tool_lines)
    policy_block = f"\n\nDomain policy you must follow:\n{policy}\n" if policy else ""
    return SystemMessage(content=(
        f"You are a calibrated retail customer service agent.{policy_block}\n\n"
        f"Available tools (use EXACT parameter names shown):\n{tool_str}\n\n"
        "For each step fill ALL fields at the TOP LEVEL — never nest confidence or risk_level inside action:\n"
        "  reasoning        — explain your thinking\n"
        "  action.tool      — tool name to call (leave null if giving a final answer)\n"
        "  action.args      — exact tool parameters as a dict (leave empty if giving a final answer)\n"
        "  action.final_answer — your response text (use instead of tool when task is complete)\n"
        "  confidence       — float 0.0–1.0 (TOP-LEVEL field, NOT inside action)\n"
        "  risk_level       — \"low\" | \"medium\" | \"high\" (TOP-LEVEL field, NOT inside action)\n\n"
        "IMPORTANT: confidence and risk_level are SIBLINGS of action, not children of it.\n"
        "IMPORTANT: Use the EXACT parameter names listed above. Do not invent or rename them."
    ))


def build_compass_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = 20,
    tool_risk: dict[str, str] | None = None,
    policy: str | None = None,
    calibration_shrink: bool = False,
) -> StateGraph:
    """tool_risk maps tool name → static risk class ('low'|'medium'|'high').
    Effective risk is max(model's verbalized risk_level, tool class), so an
    under-labelled destructive tool is still gated as high. policy, when
    given, is embedded in the system prompt (e.g. the τ-bench retail wiki).
    calibration_shrink=True selects the Phase 4 shrinkage aggregator (baseline
    stays the default)."""
    tool_risk = tool_risk or {}
    tool_map = {t.name: t for t in tools}
    # Local (Ollama) models don't reliably emit native tool_calls, so
    # method="function_calling" leaves `parsed` empty — they put schema-shaped
    # JSON in message content instead. json_schema uses Ollama's constrained
    # decoding to force valid JSON there. API models keep function_calling
    # (that path already has data and works). Either way, salvage_step below
    # recovers a step from raw content if native parsing still comes back None.
    if type(model).__name__ == "ChatOllama":
        structured_model = model.with_structured_output(
            CompassStep, method="json_schema", include_raw=True
        )
    else:
        structured_model = model.with_structured_output(
            CompassStep, method="function_calling", strict=False, include_raw=True
        )
    system = _system_prompt(tools, policy)

    def plan(state: CompassState) -> dict:
        # Parse the next step: native structured output → content salvage →
        # a bounded retry that nudges the model to re-emit a valid step. Weak
        # local models sometimes drop required fields or ramble prose into the
        # JSON; feeding the error back beats crashing the whole trial.
        messages = [system, *state["messages"]]
        result = None
        for _ in range(1 + MAX_PARSE_RETRIES):
            result = structured_model.invoke(messages)
            step: CompassStep | None = result["parsed"]
            if step is None:
                content = getattr(result.get("raw"), "content", "") or ""
                step = salvage_step(content)
            if step is not None:
                return {"steps": [step]}
            messages = [*messages, HumanMessage(content=_PARSE_RETRY_NUDGE)]
        raise ValueError(
            f"Model returned unparseable output after {MAX_PARSE_RETRIES} retries.\n"
            f"Raw response: {result['raw']}\n"
            f"Parse error: {result['parsing_error']}"
        )

    def _effective_risk(step: CompassStep) -> str:
        return max_risk(step.risk_level, tool_risk.get(step.action.tool, "low"))

    def route(state: CompassState) -> str:
        if len(state["steps"]) >= max_steps:
            return "abstain"
        step = state["steps"][-1]
        if step.action.final_answer is not None:
            return "finish"
        features = extract_features(state["steps"])
        success_prob = calibrate(step.confidence, features, shrink=calibration_shrink)
        risk = _effective_risk(step)
        decision = decide(success_prob, risk)
        # Escalate to abstain if SELF_VERIFY keeps firing without any execution
        if decision == PolicyDecision.SELF_VERIFY and state["self_verify_count"] >= MAX_SELF_VERIFY:
            return "abstain"
        # High-risk executes need an explicit verification pass first
        # (DESIGN.md Table 1): the model must re-confirm intent before acting.
        if (
            decision == PolicyDecision.EXECUTE
            and risk == "high"
            and state.get("verified_action", "") != _action_fingerprint(step)
        ):
            return "confirm"
        return decision.value  # "execute" | "self_verify" | "abstain"

    def execute(state: CompassState) -> dict:
        step = state["steps"][-1]
        tool_name = step.action.tool
        tool = tool_map.get(tool_name) if tool_name else None
        if tool is None:
            # The step routed to EXECUTE but names no real tool — either tool is
            # None (no action, no final answer) or a hallucinated name (e.g. a
            # local model inventing "Predictive Analytics"). Feed the error back
            # as an observation so the model can recover on the next step rather
            # than crashing the whole trial; max_steps still bounds the loop.
            available = ", ".join(sorted(tool_map)) or "(none available)"
            msg = (
                f"No valid tool call was made (action.tool={tool_name!r} is not an "
                f"available tool). Either call one of [{available}] with the exact "
                f"parameter names, or set action.final_answer to respond."
            )
            return {"messages": [HumanMessage(content=msg)], "verified_action": ""}
        try:
            result = tool.invoke(step.action.args)
        except Exception as e:
            # The tool exists but rejected the call (e.g. Pydantic arg validation,
            # a not-found id). Surface it as an observation so the model can fix
            # the call rather than aborting the whole trial. Not counted as
            # progress, so the self_verify streak keeps running toward abstain.
            msg = (
                f"Tool '{tool_name}' failed: {type(e).__name__}: {e}. "
                f"Check the parameter names and values against the tool signature "
                f"and try again, or set action.final_answer to respond."
            )
            return {"messages": [HumanMessage(content=msg)], "verified_action": ""}
        return {
            "messages": [HumanMessage(content=f"Tool '{tool_name}' returned: {result}")],
            "self_verify_count": 0,  # real progress — reset the streak
            "verified_action": "",  # each high-risk action needs its own confirm
        }

    def confirm(state: CompassState) -> dict:
        step = state["steps"][-1]
        msg = HumanMessage(content=(
            f"You are about to take a HIGH-risk action: "
            f"{step.action.tool}({step.action.args}). "
            "Re-read the user's original request and confirm this action is "
            "exactly what they asked for. If it is, repeat the same action; "
            "if not, change course."
        ))
        return {"messages": [msg], "verified_action": _action_fingerprint(step)}

    def self_verify(state: CompassState) -> dict:
        step = state["steps"][-1]
        msg = HumanMessage(content=(
            f"Low confidence ({step.confidence:.2f}) detected on a {step.risk_level}-risk action. "
            "Please re-read the context carefully and verify your plan before proceeding."
        ))
        return {"messages": [msg], "self_verify_count": state["self_verify_count"] + 1}

    def abstain(state: CompassState) -> dict:
        step = state["steps"][-1]
        risk = _effective_risk(step) if step.action.tool else step.risk_level
        success_prob = calibrate(
            step.confidence, extract_features(state["steps"]), shrink=calibration_shrink
        )
        msg = AIMessage(content=(
            f"ABSTAINING: calibrated success probability {success_prob:.2f} "
            f"(verbalized confidence {step.confidence:.2f}) is below threshold "
            f"for {risk}-risk action. Reasoning: {step.reasoning}"
        ))
        return {"messages": [msg], "abstained": True}

    def finish(state: CompassState) -> dict:
        step = state["steps"][-1]
        return {"messages": [AIMessage(content=step.action.final_answer)]}

    graph = StateGraph(CompassState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("confirm", confirm)
    graph.add_node("self_verify", self_verify)
    graph.add_node("abstain", abstain)
    graph.add_node("finish", finish)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route, {
        "execute": "execute",
        "confirm": "confirm",
        "self_verify": "self_verify",
        "abstain": "abstain",
        "finish": "finish",
    })
    graph.add_edge("execute", "plan")
    graph.add_edge("confirm", "plan")
    graph.add_edge("self_verify", "plan")
    graph.add_edge("abstain", END)
    graph.add_edge("finish", END)

    return graph.compile()
