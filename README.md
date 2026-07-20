# Compass 🧭

**A training-free calibration layer that teaches AI agents when they *don't* actually know.**

[![CI](https://github.com/harishchaurasia/compass-arch/actions/workflows/ci.yml/badge.svg)](https://github.com/harishchaurasia/compass-arch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

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

**Four models, one suite.** Compound failure = the agent took a destructive, irreversible action
while wrong (mutated a real order it shouldn't have). *Shrinkage* is an opt-in variant that
discounts an unearned "100%" toward a base rate *before* the agent acts.

![Headline results](analysis/figures/headline_metrics.png)

**Destructive compound failures (lower is better), and the trials that mutated a real order:**

| Model | Vanilla | Compass | Compass + shrinkage |
|---|---|---|---|
| gpt-4o-mini *(frontier)* | 54.8% (95) | **18.3%** (24) | — |
| Qwen2.5 14B | 6.1% (7) | 18.3% (24) | **0.0%** (0) |
| Qwen2.5 7B | 12.2% (14) | 12.2% (16) | **0.0%** (0) |
| Llama 3.1 8B | 1.7% (2) | 0.9% (1) | **0.0%** (0) |

**The cross-model finding:** what Compass needs depends on the model's *failure mode*.
- **gpt-4o-mini** carries real signal in its confidence, so baseline Compass cuts compound
  failures by two thirds out of the box (54.8% → 18.3%).
- **The Qwens are overconfident** — verbalized confidence is a flat ~1.0, so baseline Compass is
  blind to the *first* high-risk action and can even make things worse (14B: 6.1% → 18.3%). The
  base-rate prior restores the gate and drives destructive failures to **zero** on both the 7B
  and the 14B — the fix generalizes across model sizes.
- **Llama 3.1 8B is timid**: it rarely takes a destructive action at all (1.7%), so there is
  little to gate; shrinkage still cleans up the last one.

The cost is coverage — the agent abstains and asks for help more often (e.g. Qwen2.5 7B
abstention rises to 45% under shrinkage), and selective task success dips a few points. "Zero"
means zero *on these 115 tasks*, not a proof of perfection. The open question
([FINDINGS.md](FINDINGS.md)) is recovering that lost coverage with an *earlier* honest signal.

**Is the confidence itself more honest?** Yes - that's the mechanism behind the numbers above.
Raw verbalized confidence is badly miscalibrated everywhere (models report ~0.9-1.0 while
succeeding <15% of the time). Compass's calibrated success probability lowers Expected
Calibration Error on every model, and the shrinkage variant most of all (e.g. Qwen2.5 14B
ECE 0.89 → 0.64, Brier 0.86 → 0.48). Details, the full ECE/Brier table, and the reliability
caveat in [FINDINGS.md](FINDINGS.md#5-calibration-is-the-confidence-itself-more-honest).

### Cross-domain check: a real MCP filesystem server

The finding isn't tied to one benchmark. A second suite runs on a purpose-built
**filesystem MCP server** (real JSON-RPC over stdio) with 12 cascading-failure tasks -
decoy files bait an early misidentification that destroys the *wrong* file.

![MCP cross-domain](analysis/figures/mcp_compound_failures.png)

Compass drives destructive failures to **0%** on both Qwen models here too (Qwen2.5 7B:
16.7% → 0%) *while holding or improving* task success - its intended regime. But
`gpt-4o-mini` marks the boundary: under this suite's safety-first policy it causes **0%**
compound failures unaided, so Compass adds no safety and its risk gate instead
false-abstains on correct writes, halving selective success (66.7% → 33.3%). The lesson:
Compass helps when the agent is miscalibrated enough to actually destroy things, and
costs task success when it is not. Llama 3.1 8B is the other edge - it abstains heavily
yet one action still slips through. Small suite (n=12), so directional. Full breakdown in
[FINDINGS.md §6](FINDINGS.md).

## Reproduce

```bash
uv sync
# frontier baseline
uv run python scripts/run_tau_eval.py --provider openai --model gpt-4o-mini
# local baseline (locked aggregator) — swap in qwen2.5:7b / llama3.1:8b
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:14b
# shrinkage variant (Phase 4)
uv run python scripts/run_tau_eval.py --provider ollama --model qwen2.5:14b \
  --calibration shrinkage --conditions compass
# cross-domain: custom filesystem MCP suite
uv run python scripts/run_mcp_eval.py --provider ollama --model qwen2.5:14b
```

Local-GPU / Windows runners and troubleshooting live in [RUNBOOK.md](RUNBOOK.md).

## Status & roadmap

Heading toward production, built in the open - not there yet.

- ✅ End-to-end calibrated agent + locked rule-based aggregator
- ✅ Full 115-task A/B on **gpt-4o-mini**: compound failures 54.8% -> 18.3%
- ✅ Full 115-task A/B across three local models (**Qwen2.5 7B / 14B**, **Llama 3.1 8B**);
  the shrinkage variant drives destructive failures to **0%** on all three
- ✅ Custom **filesystem MCP** suite (real stdio server, 12 cascading-failure tasks); the
  finding reproduces cross-domain on the Qwens, and `gpt-4o-mini` marks the boundary where
  the gate costs more than it saves
- ✅ Same bridge drives real off-the-shelf MCP servers live: official filesystem (14 tools)
  and GitHub (26 tools), risk-classed for gating (`scripts/mcp_real_servers.py`)
- 🔜 Recover the coverage that caution costs (an earlier, honest pre-action signal)

Contributions and PRs welcome - if agent reliability is your world, let's connect.

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the ground rules that keep the
results honest.

## License

[MIT](LICENSE) © Harish Chaurasia
