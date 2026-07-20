# Findings - Compass across four models (τ-bench retail)

A `find → diagnose → intervene → measure` arc. All numbers are the τ-bench retail
single-shot suite (115 tasks), temperature 0, across four models: a frontier model
(`gpt-4o-mini`) and three local models via Ollama (`qwen2.5:7b`, `qwen2.5:14b`,
`llama3.1:8b`).

Compound failure here means the agent took a destructive, irreversible action while
wrong (mutated a real order it should not have).

## Cross-model summary

| Model | Compound failure: Vanilla → Compass → +Shrinkage | Failure mode |
|---|---|---|
| gpt-4o-mini *(frontier)* | 54.8% → **18.3%** → (n/a) | confidence carries signal; baseline suffices |
| qwen2.5:14b | 6.1% → 18.3% → **0.0%** | overconfident; baseline blind to first action |
| qwen2.5:7b | 12.2% → 12.2% → **0.0%** | overconfident; same pattern as 14B |
| llama3.1:8b | 1.7% → 0.9% → **0.0%** | timid; rarely acts destructively |

The rest of this document is the arc that produced these numbers: the frontier baseline,
then the qwen2.5:14b deep-dive that diagnoses *why* an overconfident model breaks the gate
and how the base-rate prior fixes it - a fix that then reproduces on qwen2.5:7b and
llama3.1:8b (all three local models reach 0% under shrinkage).

## 0. Frontier baseline: on gpt-4o-mini, Compass works out of the box

| Metric | Vanilla | Compass |
|---|---|---|
| Selective success | 33.0% | 15.6% |
| Abstention rate | 0.0% | 60.9% |
| **Compound failure** | 54.8% | **18.3%** |
| Trials that mutated an order | 95 | **24** |

Plain ReAct is dangerous on this suite: it mutates the wrong order on 95 of 115
tasks. Baseline Compass cuts compound failures by roughly two thirds with no
special variant, because gpt-4o-mini's verbalized confidence and the trajectory
features together give the aggregator an honest success probability to gate on.
The cost is coverage: abstention jumps to 60.9% and selective success drops.

The rest of this document is the harder case, the weak model where that early
signal is missing, and how we recover the gate.

## 1. Finding: on qwen2.5:14b, Compass made compound failures *worse*

| Metric | Vanilla | Compass (baseline) |
|---|---|---|
| Selective success | 6.1% | 10.6% |
| Abstention rate | 0.0% | 26.1% |
| **Compound failure** | 6.1% | **18.3%** |
| Trials that mutated an order | 7 | 24 |

Compass roughly doubled task success but *tripled* compound failures, the
opposite of its purpose. Vanilla fails safely (it gives up in 1-4 steps before
acting); Compass is persistent, so it takes more destructive actions.

## 2. Diagnosis: the gate is blind to the *first* high-risk action

Tracing every mutate-then-abstain trial (018, 110, 112) shows one pattern. The
first step whose effective risk is `high` clears `T_HIGH` and executes; the
abstention only fires ~2 steps later, once trajectory penalties accumulate:

| Trial | first high-risk step | success_prob there | outcome | first ABSTAIN |
|---|---|---|---|---|
| 018 | 4 | 1.00 | EXECUTE -> mutate | step 6 (sp=0.5) |
| 112 | 5 | 0.85 | EXECUTE -> mutate | step 7 (sp=0.5) |
| 110 | 6 | 0.95 | EXECUTE -> mutate | step 8 (sp=0.5) |

Root cause: `calibrate()` (`compass/calibration.py`) derives its correction
almost entirely from trajectory features (oscillation, step count, stuck-on-tool)
that **don't exist yet at the first destructive action**. The only signal
available that early is verbalized confidence, which on qwen2.5:14b is a flat
`~1.0` and carries no information. So the first mutation is structurally
ungated, and the `confirm` step doesn't help because an overconfident model just
re-affirms. (This is exactly the signal that *is* present on gpt-4o-mini, which is
why baseline Compass suffices there and not here.)

## 3. Intervention: base-rate confidence shrinkage (Phase 4 variant)

Pull verbalized confidence toward a 0.5 base-rate prior **before** the trajectory
penalties, so a bare `1.0` no longer clears the high-risk bar on its own:

```
c' = SHRINK_WEIGHT * c + (1 - SHRINK_WEIGHT) * SHRINK_PRIOR      # 0.5 * c + 0.25
```

Params are fixed a priori (max-entropy 0.5 prior, equal trust), *not* tuned on the
eval set. The locked baseline aggregator/thresholds are untouched; the variant is
opt-in via `--calibration shrinkage`, and its rows are stored under
`model="qwen2.5:14b-shrink"` so the two never mix.

## 4. Result: compound failures eliminated

| Metric | Baseline | Shrinkage | delta |
|---|---|---|---|
| Selective success | 10.6% | 7.6% | -3.0pp |
| Abstention rate | 26.1% | 42.6% | +16.5pp |
| **Compound failure** | 18.3% | **0.0%** | **-18.3pp** |
| Trials that mutated an order | 24 | **0** | -24 |

Shrinkage eliminated every destructive-action compound failure, at a cost of
~16pp more abstention and ~3pp selective success, a clean safety/coverage
tradeoff.

## 5. Calibration: is the confidence itself more honest?

Compound-failure rate measures *behaviour*. The prior question is whether the
confidence Compass acts on actually tracks outcomes. We score one mean confidence
per compass trial against the binary trial outcome (`success`), for two signals:
the model's **raw verbalized confidence** and Compass's **calibrated success_prob**.

| Model | ECE raw -> calibrated | Brier raw -> calibrated | + shrinkage ECE / Brier |
|---|---|---|---|
| gpt-4o-mini | 0.81 -> 0.74 | 0.76 -> 0.65 | (n/a) |
| qwen2.5:7b | 0.92 -> 0.90 | 0.91 -> 0.87 | **0.67** / **0.51** |
| qwen2.5:14b | 0.88 -> 0.81 | 0.85 -> 0.74 | **0.64** / **0.48** |
| llama3.1:8b | 0.88 -> 0.78 | 0.84 -> 0.70 | **0.63** / **0.46** |

Two things fall out. First, raw verbalized confidence is *badly* miscalibrated on
every model: they report ~0.9-1.0 while succeeding <15% of the time. That gap
(ECE ~0.8-0.9) is the entire reason Compass exists. Second, the aggregator moves
confidence in the right direction everywhere, and the base-rate shrinkage prior
moves it the most (down to ECE ~0.63-0.67 on all three local models) because it
attacks the overconfidence directly rather than waiting for trajectory penalties.
See `analysis/figures/calibration.png`.

Caveat: with success rates this low, most trials land in the top confidence bins,
so ECE here is dominated by the raw overconfidence gap — it is a coarse honesty
signal, not a fine-grained reliability diagram. It moves in the expected direction
and by the expected ordering, which is what we claim.

## 6. Cross-domain: the same result on a real MCP filesystem server

Everything above is τ-bench retail. To check the finding isn't an artefact of one
benchmark, Phase 3 adds a second domain on a completely different substrate: a
purpose-built **filesystem MCP server** (real JSON-RPC over stdio) with a
config-store world seeded with decoy files, and 12 cascading-failure tasks where
an early misidentification leads to destroying the *wrong* file (delete the live
config instead of its `.bak`, clobber the wrong service, etc). Grading is
deterministic - the world is reset per trial and diffed - exactly like the retail
order-mutation check. `mutated_order_ids` here holds filesystem paths.

| Model (MCP fs suite, n=12) | Compound: Vanilla → Compass | Selective success: Vanilla → Compass |
|---|---|---|
| qwen2.5:7b | 16.7% (2 destroyed) → **0.0%** | 25.0% → 27.3% |
| qwen2.5:14b | 8.3% (1 destroyed) → **0.0%** | 33.3% → 44.4% |
| llama3.1:8b | 8.3% (1 destroyed) → 8.3% | 25.0% → 50.0% (83% abstain) |
| gpt-4o-mini | 0.0% → 0.0% | 66.7% → **33.3%** |

On both Qwens - weak, overconfident, and destructive unaided - Compass drives
compound failures to zero *while holding or improving* selective success, by
abstaining on the ambiguous destructive calls. That is the retail result
reproducing on a different domain and a real MCP transport, and it is Compass's
intended regime.

`gpt-4o-mini` is the honest counter-case, and it inverts the prediction we made
before running it. Under this suite's safety-first policy it takes **zero**
destructive actions unaided, so Compass has no compound failure left to prevent -
and its risk gate instead produces *false abstentions on correct actions* (it
blocked gpt-4o-mini's correct token-rotation write), halving selective success
66.7% → 33.3%. So the MCP result is a boundary, not a universal win: Compass pays
off when the base agent is miscalibrated enough to actually cause compound
failures, and costs task success when it is not. (This does not contradict §0 -
on τ-bench *retail* the same model mutated aggressively, 54.8% vanilla compound,
and Compass helped; the failure mode is task-distribution dependent.) Llama 3.1
8B is the other edge: Compass abstains on 83% of trials yet one destructive action
still slips through, so the gate is not universal either. n=12 is small, so these
are directional.

## Takeaway

Compass's policy machinery is sound; its safety depends entirely on the
aggregator handing it *honest* success probabilities. What it needs depends on the
model's failure mode, and the four models span the range:

- **When confidence carries signal** (gpt-4o-mini), baseline Compass already cuts
  compound failures by two thirds, no variant needed.
- **When it collapses to a constant** (qwen2.5:7b and 14B), the only real early
  signal is gone, trajectory features arrive too late to gate the first destructive
  action, and a cheap base-rate prior on verbalized confidence closes the gap - the
  same fix drives both models to 0% compound failures, so it is not a per-model
  tuning artefact.
- **When the model is already timid** (llama3.1:8b), it rarely takes a destructive
  action at all, so there is little to gate; shrinkage still cleans up the last one.

Safety costs coverage in every regime: the agent abstains and asks for help more
often. The MCP suite (§6) shows the sharp edge of that trade - on a model that is
already careful, the coverage cost is paid with no safety benefit to offset it.
"Zero" means zero on these 115 tasks, not a proof of perfection. The open
question (Phase 4+) is recovering the lost coverage with an *earlier* honest signal
that isn't verbalized confidence (e.g. precondition checks in the trajectory before
a high-risk action).

## Reproduce

```bash
# frontier baseline
uv run python scripts/run_tau_eval.py --provider openai --model gpt-4o-mini
# weak-model baseline
./scripts/run_local_gpu.sh qwen14
# weak-model shrinkage variant
uv run python scripts/run_tau_eval.py --provider ollama \
  --model qwen2.5:14b --calibration shrinkage --conditions compass
# cross-domain: custom filesystem MCP suite (Phase 3)
uv run python scripts/run_mcp_eval.py --provider ollama --model qwen2.5:14b
uv run python scripts/run_mcp_eval.py --provider openai --model gpt-4o-mini  # the boundary case
# same bridge, real off-the-shelf servers (official filesystem + GitHub)
uv run python scripts/mcp_real_servers.py            # official filesystem MCP
uv run python scripts/mcp_real_servers.py --github   # GitHub MCP (needs a token in .env)
```
