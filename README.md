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

## Results (115 τ-bench retail tasks · single-shot)

Two models, same suite. Compound failure = the agent took a destructive, irreversible action
while wrong (mutated a real order it shouldn't have).

**gpt-4o-mini (frontier model).** Plain ReAct is dangerous here: it mutates the wrong order on
95 of 115 tasks. Compass cuts that by roughly two thirds, out of the box.

| Metric | Vanilla | Compass |
|---|---|---|
| Selective success | 33.0% | 15.6% |
| Abstention rate | 0.0% | 60.9% |
| **Destructive compound failures** | 54.8% | **18.3%** |
| Trials that mutated a real order | 95 | **24** |

**Qwen2.5 14B (weak, overconfident model).** Verbalized confidence is a flat, useless ~1.0 here,
the hardest case for a calibration layer. Baseline Compass is blind to the *first* high-risk
action (no trajectory signal has accumulated yet), so it actually makes things worse. The
shrinkage variant, which discounts an unearned "100%" before the agent acts, closes the gap.

| Metric | Vanilla | Compass | Compass + shrinkage |
|---|---|---|---|
| Selective success | 6.1% | 10.6% | 7.6% |
| Abstention rate | 0.0% | 26.1% | 42.6% |
| **Destructive compound failures** | 6.1% | 18.3% | **0.0%** |
| Trials that mutated a real order | 7 | 24 | **0** |

**The cross-model finding:** calibration works when the model's confidence carries signal
(gpt-4o-mini), and needs an extra base-rate prior when it doesn't (Qwen). Safety costs coverage
either way, the agent abstains and asks for help more often. "Zero" means zero *on these 115
tasks*, not a proof of perfection. The open question ([FINDINGS.md](FINDINGS.md)) is recovering
that lost coverage with an *earlier* honest signal.

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
- ✅ Full 115-task A/B on **gpt-4o-mini**: compound failures 54.8% -> 18.3%
- ✅ Full 115-task A/B on **Qwen2.5 14B**; shrinkage variant gates the first destructive action
- 🔜 Extend the benchmark to more open-source local models: **Qwen2.5 7B**, **Llama 3.1 8B**
- 🔜 Recover the coverage that caution costs (an earlier, honest pre-action signal)

Contributions and PRs welcome - if agent reliability is your world, let's connect.

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
```
