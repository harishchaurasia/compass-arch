# Compass — Calibrated Agent for Production Reliability

> A training-free calibration architecture for LLM agents. Compass knows when to act, when to verify, and when to abstain.

---

## What this project is

Modern LLM agents fail in production not because they can't do tasks — they fail _confidently_. They take destructive actions while being wrong, report success on bad outputs, and compound errors across long-horizon trajectories. This silent-failure mode is the #1 blocker for production deployment of agents in 2026.

Compass is a calibration architecture that bolts onto a standard ReAct agent and gives it three capabilities the vanilla agent lacks: per-step confidence estimation, trajectory-level uncertainty signals, and a confidence-conditioned action policy that abstains or self-verifies under uncertainty.

The project tests whether this training-free architecture — runnable on top of any frontier LLM via API — measurably reduces compound failure rates on production-relevant agent tasks, without sacrificing more accuracy than is worth the safety gain.

---

## Why this matters (the research context)

Three 2026 papers anchor the direction:

- **Salesforce AUQ (Jan 2026)** — Agentic Uncertainty Quantification. Established that verbalized confidence in tool-using agents is a real signal but noisy, and that combining it with execution-trace features dramatically improves calibration.
- **MIT CSAIL RLCR (April 2026)** — Reinforcement Learning for Calibrated Reasoning. Showed 90% calibration error reduction with no accuracy loss via RL fine-tuning. We're testing the training-free analog.
- **Holistic Trajectory Calibration (Jan 2026)** — Introduced trajectory-level diagnostics (error counts, retry patterns, semantic consistency) as calibration features.

Compass takes the training-free subset of these ideas and tests them across frontier models in a realistic agent harness.

---

## The hypothesis

> An agent that integrates trajectory-level uncertainty signals — both verbalized confidence from the model and structural features from its own execution trace — and uses them to abstain, self-verify, or escalate, achieves measurably higher _expected-utility-adjusted_ task success than a vanilla agent on tasks with cascading-error potential.

The italicized phrase matters. We don't just measure "did it succeed." We measure "did it succeed, _and_ when it failed, did it fail loudly enough to be caught." A vanilla agent that's 70% right but confidently wrong on the other 30% is more dangerous in production than a calibrated agent that's 65% right and _knows_ when it might be in the wrong 35%.

This experiment is honest — if Compass doesn't beat the baseline, the negative result is also publishable. We're not committed to a marketing claim; we're committed to publishing what the experiment shows.

---

## Design — three layered components

### Component 1 — Structured per-step output

At every agent step, the model is required to output a structured JSON object:

```json
{
  "reasoning": "Brief justification for the chosen action.",
  "action": { "tool": "...", "args": { ... } },
  "confidence": 0.72,
  "risk_level": "high"
}
```

- `confidence` ∈ [0, 1] — the model's stated belief that the chosen action is correct
- `risk_level` ∈ {low, medium, high} — the cost of being wrong about this action

`risk_level` is critical because it forces the model to reason about _stakes_, not just probability. Reading data is low risk; deleting data is high risk. The model should be far more conservative about high-risk actions even at the same confidence.

Enforced via LangChain structured output / Pydantic schema.

**Amendment (Phase 2 pilot finding):** verbalized risk is unreliable — across 168 pilot steps (gpt-4o-mini) the model emitted zero "high" labels, including on 25 trials that mutated the database, so verbalized risk alone can never reach the abstain branch. Each tool therefore carries a **static risk class** (`tool_risk` map passed to `build_compass_agent`; DB-mutating retail tools are `high`, lookups `low`), and the policy gates on `effective_risk = max(verbalized risk_level, tool class)`. The verbalized label is kept as a signal that can only *raise* risk, never lower it.

### Component 2 — Trajectory feature aggregation

Per-step verbalized confidence alone is unreliable. RLHF-trained frontier models often emit confidence in a narrow band (0.8–0.95) regardless of actual correctness. So we don't rely on it alone — we also extract structural features from the running trajectory:

- `tool_error_count` — number of tool errors observed so far
- `retry_count` — number of retry / correction loops
- `plan_revision_count` — number of times the agent has changed its plan
- `semantic_consistency` — does the agent's reasoning at step N contradict step N-1?
- `trajectory_length_ratio` — actual steps elapsed vs. expected steps for task complexity

These features are computed in real time at each decision point. A rule-based aggregator combines them with verbalized confidence into a single **trajectory success probability**.

The aggregator is hand-tuned on a small dev split (~5 tasks) and locked before evaluation. The methodology is reported honestly in the writeup. A learned calibration probe (small classifier on these features) is the Phase 4 stretch goal.

### Component 3 — Confidence-conditioned action policy

This is where calibration becomes _behavior_. The trajectory success probability + risk level gates action execution:

| Risk   | Confidence | Action                                                                           |
| ------ | ---------- | -------------------------------------------------------------------------------- |
| Low    | any        | Execute                                                                          |
| Medium | ≥ T_med    | Execute                                                                          |
| Medium | < T_med    | **Self-verify**: pause, re-read context, restate plan, then execute or escalate  |
| High   | ≥ T_high   | Execute with explicit verification step (model re-confirms intent before acting) |
| High   | < T_high   | **Abstain**: return "uncertain, escalating" with full reasoning trace            |

T_med and T_high are tuned on the dev split. Risk in this table is the _effective_ risk (see Component 1 amendment). The "execute with explicit verification" row is implemented as a `confirm` graph node: the model is told what high-risk action it is about to take, must re-read the user's request, and either re-affirms the action (which then executes) or changes course.

The two thresholds matter because abstention is _expensive_ in terms of task success rate but _cheap_ in terms of compound failures. A well-tuned Compass abstains often enough to catch its mistakes but not so often that it becomes useless.

---

## The vanilla baseline (for fair comparison)

The baseline is a standard ReAct agent with:

- Same base model
- Same agent loop (thought → action → observation → loop)
- Same tool definitions
- Same task format
- **No** confidence elicitation, **no** trajectory features, **no** abstention, **no** self-verification
- Hard step budget (max N steps)

This represents the current production deployment baseline. Not a strawman — what most agents in production today actually look like.

---

## Experimental setup

### Models (4)

- **Claude Opus 4.7** (Anthropic API) — frontier reasoning
- **GPT-5.5** (OpenAI API) — frontier general
- **Gemini 3-tier** (Google API) — frontier alternative
- **Qwen 2.5 7B** (local on RTX 5080 via vLLM or Ollama) — open-source reference

### Task suite (~52 tasks total)

**τ-bench subset — 40 tasks.** Balanced retail + airline domains. Used as the academic anchor: results are directly comparable to AUQ and HTC paper evaluations. We subsample (don't run all 165) for budget and statistical sufficiency.

**Custom MCP suite — ~12 tasks.** Designed across 2–3 real public MCP servers (GitHub MCP, Linear MCP, Slack MCP, or similar). Tasks are _deliberately designed_ to have cascading-failure potential: an early wrong tool call poisons the rest of the trajectory. This is the production-grounding half of the eval and the source of the demo video.

The custom MCP task design happens _after_ Phase 1 pilot reveals what failure patterns we want to amplify.

### Conditions (2)

- **Vanilla** — baseline ReAct
- **Compass** — full calibration architecture

### Trial matrix

~52 tasks × 4 models × 2 conditions = **~416 trials**

API cost estimate: $150–$300 total across the project.

### Primary metrics

- **Expected Calibration Error (ECE) / Brier score** — does Compass's confidence track actual outcomes?
- **Selective task success rate** — accuracy on tasks where the agent committed (didn't abstain), plus abstention rate
- **Compound failure rate** — % of trials where the agent took a destructive action while wrong (the production nightmare)

### Secondary metrics

- Token and latency overhead of Compass vs. vanilla
- Failure-mode breakdown (overconfident-wrong, underconfident-right, well-calibrated-wrong, well-calibrated-right)
- Skill transfer: do Compass's trajectory features generalize across domains within τ-bench?

---

## Stack

- **Python 3.11+**
- **LangGraph** — agent loops, state graphs, checkpointing, `interrupt()` for abstention/escalation
- **LangChain** — model wrappers, tool abstraction (`init_chat_model` for cross-provider unified interface)
- **Anthropic MCP Python SDK** — for the custom MCP task suite
- **vLLM** or **Ollama** — local Qwen 2.5 7B serving on RTX 5080
- **SQLite** — trial results storage (one row per trial, JSON columns for trajectories)
- **Jupyter notebooks** — analysis and chart production
- **LangSmith** (optional, development only) — tracing during debugging

### What we deliberately are NOT using

- No custom tracing tool — we roll our own JSON trace format. Lighter, no external deps, and _understanding our own tracing_ is more useful interview material than "I plugged in LangSmith."
- No web UI / dashboard — the writeup is markdown + static charts from Jupyter
- No Docker — single laptop project, skip ops overhead
- No RL fine-tuning (except as Phase 4 stretch) — training-free is the primary contribution

---

## Repo structure

```
compass/
├── .claude/
│   └── skills/
│       └── engineering-coach/SKILL.md   # decision-narration skill, project-agnostic
├── compass/                              # main package
│   ├── __init__.py
│   ├── agent_vanilla.py                 # baseline ReAct LangGraph
│   ├── agent_compass.py                 # Compass LangGraph with calibration
│   ├── trajectory.py                    # trajectory feature extraction
│   ├── calibration.py                   # confidence aggregation logic
│   ├── policy.py                        # action policy rules
│   ├── models.py                        # LangChain model wrappers
│   └── tools/                           # tool implementations
├── eval/                                 # evaluation harness
│   ├── tau_bench_runner.py
│   ├── mcp_runner.py
│   └── metrics.py                       # ECE, Brier, selective accuracy, compound failure
├── tasks/                                # task definitions
│   ├── tau_bench/                       # subset of τ-bench
│   └── custom_mcp/                      # ~12 custom MCP tasks
├── results/
│   └── trials.db                        # SQLite of every trial
├── analysis/                             # Jupyter notebooks
├── pyproject.toml
├── README.md                             # short, user-facing
└── DESIGN.md                             # this document, the canonical spec
```

---

## Build approach

Agile, build-as-we-go. Four loose phases, ~20 hours each, ~80 hours total. Phases are guides not contracts — slippage between phases is fine; pivoting between projects is not.

### Phase 1 — Foundation

- LangGraph repo scaffold
- Vanilla ReAct agent working end-to-end on 2–3 τ-bench tasks
- Compass v1 with verbalized confidence + risk_level + basic action policy
- Pilot on 5 τ-bench tasks comparing vanilla vs Compass v1

**Critical pilot question:** does verbalized confidence vary meaningfully across tasks, or is it flat? This determines whether trajectory features are _helpful_ or _essential_ in Phase 2.

### Phase 2 — Full Compass + τ-bench at scale

- Trajectory feature extraction
- Aggregator combining verbalized confidence + trajectory features
- Threshold tuning on 5-task dev split
- Full 40-task τ-bench evaluation × 3 frontier models × 2 conditions = 240 trials
- Initial chart drafts

**Headline question:** does Compass meaningfully reduce compound-failure rate vs. vanilla?

### Phase 3 — Custom MCP + local model

- Design 12 custom MCP tasks across 2–3 real MCP servers with cascading-failure potential
- Set up Qwen 2.5 7B on vLLM or Ollama
- Run full eval on local model
- Run vanilla and Compass on custom MCP tasks

**Differentiation question:** do findings hold across frontier APIs _and_ a local open-source model?

### Phase 4 — Writeup + ship

- Final charts, error analysis, failure-mode taxonomy
- Blog post / DESIGN.md polished into shareable writeup
- Demo video (side-by-side: vanilla failing vs Compass abstaining)
- LinkedIn post + Twitter thread
- Repo README polished for cloning

**Stretch:** LoRA-fine-tune a small calibration probe on Qwen as a "what if learned aggregation works better than rule-based" ablation. Stretch only — don't burn primary milestones for this.

---

## Open questions to resolve during the build

- **Verbalized confidence flatness.** ~~If frontier models output near-constant confidence regardless of task difficulty, the design needs adjustment in Phase 2.~~ **Answered by the Phase 2 pilot (gpt-4o-mini):** confidence is flat (0.8–1.0, mode 0.9) and verbalized risk never reaches "high". Adjustments made: static tool risk classes + confirm step (see Component 1/3). Trajectory features also never fired on short 3–5 step trajectories — they need harder, longer tasks to matter.
- **Threshold tuning.** A 5-task dev split is small. May need cross-validation across τ-bench domains. Report sensitivity analysis in the writeup.
- **Custom MCP task design.** Should happen _after_ Phase 2 pilot reveals which failure modes are most informative to amplify.
- **Local model architecture.** Qwen 2.5 7B is the planning baseline. If instruction-following is too weak for tool-use agent tasks, swap to Llama 3.1 8B or similar.
- **Custom MCP server choice.** GitHub MCP, Linear MCP, Slack MCP, filesystem MCP — narrow to 2–3 in Phase 3 based on which exhibit the cleanest cascading-failure patterns.

---

## Working notes for Claude Code sessions

When starting a fresh Claude Code session in this repo:

1. **Read this `DESIGN.md` fully** — it is the canonical project spec.
2. **Read `.claude/skills/engineering-coach/SKILL.md`** — the coaching skill is active for this project. Substantive engineering decisions should be narrated using the coach block template.
3. **Check `results/trials.db`** schema (if it exists) to understand what experiments have already been run before designing new ones.
4. **Honor the coach skill aggressively.** Undertriggering defeats the point. Narrate decisions before executing them: library choice, architecture pattern, naming a key abstraction, completing a substantive task. Skip narration only for trivial actions (running tests, viewing files, simple renames).
5. **When in doubt, ship a working primitive over a polished design.** Phase 1's goal is "does any of this work end to end" — not "is the architecture beautiful." Polish comes in Phase 4.

---

## The story for interviews

> "Most production agents fail not because they can't do tasks but because they fail confidently — they take destructive actions while wrong and report success on bad outputs. I built Compass, a training-free calibration architecture that combines verbalized confidence with trajectory-level uncertainty signals to abstain, self-verify, or escalate before high-stakes actions. Across ~52 tasks (τ-bench + a custom MCP suite) and four models (3 frontier APIs + a local 7B), Compass reduced compound failure rate by [X]% at the cost of [Y]% absolute task success rate. In production deployments where silent failures matter more than slight accuracy drops, this is the right tradeoff."

The numbers go in when we have them.

---

## Sibling work

This project is a sibling, not a replacement, for **WebSecArena** (Harish Chaurasia, 2025). WebSecArena measured agent robustness under _adversarial inputs_. Compass measures agent reliability under _epistemic uncertainty_. Together they form a reliability stack: WebSecArena tests whether agents fail loudly when attacked; Compass tests whether agents fail loudly when uncertain.

Both projects share the multi-axis experimental rigor: multiple models × multiple architectures × multiple conditions × hundreds of trials, with honest reporting of negative results.
