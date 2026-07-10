# Compass 🧭

**A training-free calibration layer that teaches AI agents when they *don't* actually know.**

Compass wraps a standard ReAct agent and decides - before every action - whether to
**execute, self-verify, or abstain**, based on how trustworthy the agent's own confidence
really is. No fine-tuning, no extra model. It runs on top of any frontier or local LLM.

> See [DESIGN.md](DESIGN.md) for the full architecture and experimental design, and
> [FINDINGS.md](FINDINGS.md) for the detailed find → diagnose → intervene → measure writeup.

---

## The problem: agents can't tell "I know" from "I *feel* like I know"

Ask an LLM agent if it's sure, and it says *"100%."* It then takes an irreversible action -
cancels the wrong order, issues the wrong refund - or loops in circles insisting it's got it,
while being confidently, completely wrong. The confidence number is real to the model. It's
just **disconnected from reality.** In production, this silent-failure mode - acting decisively
while wrong - is more dangerous than an agent that simply gives up.

## The approach: confidence is *earned*, not announced

Compass never takes a self-reported "100%" at face value. Two ideas do the work:

1. **Discount unearned certainty.** A bare "I'm sure" is pulled toward a realistic base rate
   *before* the agent is allowed to act on a high-stakes step. Certainty has to be justified.
2. **Decay confidence as the agent flails.** The longer it loops without real progress
   (oscillation, repeated tools, ballooning step count), the more Compass shaves its confidence
   down - until it makes the agent stop, double-check, or hand off.

That estimate, combined with each action's **risk level**, gates behavior:

```
CompassStep (reasoning · action · confidence · risk)
  → trajectory features  (oscillation, step count, stuck-on-tool …)
  → calibration          (verbalized confidence + trajectory → success probability)
  → policy               (EXECUTE / SELF_VERIFY / ABSTAIN, conditioned on risk)
```

## Results (Qwen2.5 14B · 115 τ-bench retail tasks · single-shot)

Run against a deliberately overconfident local model, where verbalized confidence is a flat,
useless ≈1.0 - the hardest case for a calibration layer.

| Metric | Vanilla | Compass | Compass + shrinkage |
|---|---|---|---|
| Selective success | 6.1% | 10.6% | 7.6% |
| Abstention rate | 0.0% | 26.1% | 42.6% |
| **Destructive compound failures** | 6.1% | 18.3% | **0.0%** |
| Trials that mutated a real order | 7 | 24 | **0** |

A naive gate is blind to the *first* high-risk action (trajectory penalties haven't accumulated
yet, verbalized confidence carries no signal) - so it actually made destructive failures worse.
The **shrinkage** variant discounts that first unearned "100%" and eliminated every irreversible
mutation on this benchmark.

**Honest scope:** this is one model, one domain. "Zero" means zero *on these 115 tasks*, not a
proof of perfection. Safety also costs coverage here - the agent asks for help more often
(+16pp abstention, −3pp success). The open question ([FINDINGS.md](FINDINGS.md)) is recovering
that coverage with an *earlier* honest signal.

## Reproduce

```bash
uv sync
# baseline (locked aggregator)
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:14b
# shrinkage variant (Phase 4)
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:14b \
  --calibration shrinkage --conditions compass
```

Local-GPU / Windows runners and troubleshooting live in [RUNBOOK.md](RUNBOOK.md).

## Status & roadmap

Heading toward production, built in the open - not there yet.

- ✅ End-to-end calibrated agent + locked rule-based aggregator
- ✅ Full A/B on Qwen2.5 14B; shrinkage variant gates the first destructive action
- 🔜 Benchmark across more open-source local models - **Qwen2.5 7B**, **Llama 3.1 8B**
- 🔜 Recover the coverage that caution costs (an earlier, honest pre-action signal)

Contributions and PRs welcome - if agent reliability is your world, let's connect.

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
```
