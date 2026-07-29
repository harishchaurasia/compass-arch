# Findings - Compass across four models

A `find → diagnose → intervene → measure` arc across four models: a frontier model
(`gpt-4o-mini`) and three local models via Ollama (`qwen2.5:7b`, `qwen2.5:14b`,
`llama3.1:8b`), temperature 0. §0-5 are the τ-bench retail single-shot suite (115
tasks); §6 is the cross-domain MCP filesystem suite (31 tasks); §7 is the verification
ablation, §8 the `T_HIGH` sweep, §9 a second real τ-bench domain (airline, 50 tasks), §10 a
from-the-traces test of the "precondition check" fix for the discrimination gap, and §11 a
first learned-probe attack on it with a semantic action-vs-request signal.
The short version: Compass reduces destructive failures, but on the weak models it does so
by *abstaining broadly* rather than by gating precisely, and §3/§7/§8 are the honest
accounting of exactly how.

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
then the qwen2.5:14b deep-dive that diagnoses *why* an overconfident model breaks the gate,
and what the base-rate prior actually does about it. All three local models reach 0% compound
failure under shrinkage - but §3 shows that "0%" is a categorical block on high-risk
execution, not a calibrated gate, and §7 (the verification ablation) and §8 (the T_HIGH
sweep) locate where the safety really comes from. Read those before quoting the 0%.
§9 then confirms the whole pattern holds on a second real τ-bench domain (airline): baseline
Compass again helps the frontier model and *worsens* qwen2.5:14b, and shrinkage again zeroes
compound by the same categorical block.

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
signal is missing, and what closing the gap there actually costs.

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

**What shrinkage actually does - read this before trusting the 0%.** The transform
maps confidence `c ∈ [0,1]` onto `[0.25, 0.75]`. Its ceiling is `0.5·1.0 + 0.25 =
0.75`, and trajectory penalties only ever push lower. The high-risk threshold is
`T_HIGH = 0.8`. So under shrinkage **no high-risk action can ever clear the gate** -
every high-risk step routes to ABSTAIN, unconditionally, regardless of whether it was
the right action. Compound failure goes to 0% *by construction*: the agent simply
never executes a high-risk action. Measured on the 31-task MCP suite, high-risk
actions that reach execution collapse from 22/31 (qwen2.5:14b), 17/31 (qwen2.5:7b),
14/31 (llama3.1:8b) at baseline to **0/31 on all three** under shrinkage. This is
avoidance, not discrimination - which is exactly why selective success does not
improve and abstention climbs (§4, §6). It is equivalent to a policy of "always
abstain on high-risk," and §8's threshold sweep shows you reach the same frontier by
moving `T_HIGH` above the ceiling directly. Shrinkage is a blunt safety switch, not a
sharper gate.

## 4. Result: compound failures eliminated

| Metric | Baseline | Shrinkage | delta |
|---|---|---|---|
| Selective success | 10.6% | 7.6% | -3.0pp |
| Abstention rate | 26.1% | 42.6% | +16.5pp |
| **Compound failure** | 18.3% | **0.0%** | **-18.3pp** |
| Trials that mutated an order | 24 | **0** | -24 |

Shrinkage eliminated every destructive-action compound failure, at a cost of
~16pp more abstention and ~3pp selective success. Read together with §3, that is
the whole story: it removed the failures by removing high-risk execution outright,
not by telling good high-risk actions from bad ones. Selective success does not
improve because the gate is not discriminating - it is closing.

## 5. Calibration: is the confidence itself more honest?

Compound-failure rate measures *behaviour*. The prior question is whether the
confidence Compass acts on actually tracks outcomes. We score one mean confidence
per compass trial against the binary trial outcome (`success`), for two signals:
the model's **raw verbalized confidence** and Compass's **calibrated success_prob**.

| Model | ECE raw -> calibrated | Brier raw -> calibrated | + shrinkage ECE / Brier |
|---|---|---|---|
| gpt-4o-mini | 0.81 -> 0.74 | 0.75 -> 0.65 | (n/a) |
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
so ECE here is dominated by the raw overconfidence gap - it is a coarse honesty
signal, not a fine-grained reliability diagram. It moves in the expected direction
and by the expected ordering, which is what we claim.

## 6. Cross-domain: a real MCP filesystem server (31 tasks)

Everything above is τ-bench retail. To check the finding isn't an artefact of one
benchmark, Phase 3 adds a second domain on a completely different substrate: a
purpose-built **filesystem MCP server** (real JSON-RPC over stdio) with a
config-store world seeded with decoy files, and **31 cascading-failure tasks** where
an early misidentification leads to destroying the *wrong* file (delete the live
config instead of its `.bak`, clobber the wrong service, etc). Grading is
deterministic - the world is reset per trial and diffed - exactly like the retail
order-mutation check. `mutated_order_ids` here holds filesystem paths. (An earlier
12-task version of this suite showed baseline Compass reaching 0% on the Qwens; on
the larger suite below it does not, and the correction matters - see the note.)

| Model (MCP fs, n=31) | Compound: Vanilla → Compass → +Shrink | Selective success: Vanilla → Compass |
|---|---|---|
| qwen2.5:7b  | 9.7% (3) → 3.2% (1) → **0.0%** | 19.4% → 26.3% |
| qwen2.5:14b | 12.9% (4) → 6.5% (2) → **0.0%** | 32.3% → 26.1% |
| llama3.1:8b | 9.7% (3) → 9.7% (3) → **0.0%** | 19.4% → **0.0%** (96.8% abstain) |
| gpt-4o-mini | 0.0% → 0.0% → 0.0%             | 74.2% → **33.3%** |

Three honest corrections to the retail-era story fall out of the bigger suite:

1. **Baseline Compass does not reach 0% on the Qwens here.** It roughly halves
   compound failures (qwen2.5:7b 3→1, qwen2.5:14b 4→2) by abstaining on some
   ambiguous destructive calls, but the first ungated high-risk action (§2) still
   gets through on the rest. Only **shrinkage** reaches 0% - and per §3 that is the
   categorical high-risk block, not a sharper gate. So the cross-domain reproduction
   is "shrinkage drives compound to 0% by refusing all high-risk execution," the same
   mechanism as retail, with the same caveat.

2. **llama3.1:8b is a degenerate case, not a win.** Baseline Compass abstains on
   96.8% of tasks (30/31) and scores **0% selective success** - it refuses to do
   anything useful - and *still* destroys 3 files, because those destructive writes
   execute mid-trajectory before the run finally abstains (28 of the 30 abstentions
   are genuine policy abstentions that end early, not step-budget timeouts). Shrinkage
   removes the 3 destructions but leaves selective success at 0%. On this suite Compass
   makes llama unusable; that belongs in the record, not a "0% compound" headline.

3. **gpt-4o-mini remains the boundary,** and the bigger suite sharpens it. Under this
   suite's safety-first policy it takes **zero** destructive actions unaided (74.2%
   selective success), so Compass has no compound failure to prevent - and its gate
   instead produces false abstentions on correct actions, halving selective success to
   33.3%. Shrinkage makes it worse (45.2% abstention). This does not contradict §0: on
   τ-bench *retail* the same model mutated aggressively (54.8% vanilla compound) and
   Compass helped. The payoff is task-distribution dependent.

Net: Compass pays off only when the base agent is miscalibrated enough to actually
cause compound failures *and* capable enough to still act usefully under the gate.
Outside that band it either costs coverage for no safety (gpt-4o-mini) or collapses
into near-total abstention (llama).

## 7. Ablation: where does the safety come from - verification or abstention?

Shipped Compass has two safety mechanisms bolted together: the **abstention gate**
(high-risk action below `T_HIGH` → refuse) and **verification** (SELF_VERIFY on
low-confidence steps, plus a mandatory confirm pass before any high-risk execute).
To separate them, the `--no-verification` arm strips verification and keeps only the
abstention gate, so any surviving safety is attributable to abstention alone. compass
vs noverify, compound-failure rate:

| Model | τ-bench compass → noverify | MCP compass → noverify | verification's effect |
|---|---|---|---|
| gpt-4o-mini | 18.3% → **33.0%** | 0% → 0% | **large** on retail: halves compound |
| qwen2.5:14b | 18.3% → 16.5% | 6.5% (2) → 6.5% (2) | negligible |
| qwen2.5:7b | 12.2% → 16.5% | 3.2% (1) → 6.5% (2) | small |
| llama3.1:8b | 0.9% → 0.0% | 9.7% (3) → 9.7% (3) | none (noise) |

The result splits by model capability. On **gpt-4o-mini**, verification is the
dominant safety mechanism: removing it nearly doubles retail compound failures
(18.3% → 33.0%). The confirm pass works because a capable model, asked to re-read the
request before a high-risk write, actually backs out of the wrong action. On the
**local models**, verification contributes little to nothing - their first high-risk
action clears `T_HIGH` on flat ~1.0 confidence (§2) and an overconfident model just
re-affirms it at the confirm prompt, so stripping verification changes compound
failures by at most one task. Their (smaller) safety comes from the abstention gate,
not verification.

Verification also carries the coverage cost. Stripping it recovers large amounts of
coverage everywhere: abstention on gpt-4o-mini retail drops 60.9% → 35.7%, on
qwen2.5:14b MCP 25.8% → 6.5%, and selective success rises in step (qwen2.5:14b MCP
26.1% → 44.8%). So verification buys safety on a capable model at a steep coverage
price, and buys almost nothing on a weak one. That is a real argument for making
verification conditional on the model, not a fixed part of the pipeline - a Phase 4
lever, flagged not taken.

## 8. Sensitivity: does the locked `T_HIGH = 0.8` sit on a knife-edge?

`T_HIGH` was locked on a 5-task dev split before eval. To check we aren't reporting a
number that only holds at one threshold, the `--t-high` sweep re-runs the MCP suite at
0.7 and 0.9 (0.8 is the shipped compass row). This is a sensitivity analysis, not
re-tuning: swept rows are stored under `model="<model>-thigh<value>"` and the shipped
0.8 in `policy.py` is untouched.

| T_HIGH | gpt-4o-mini sel / abstain / compound | qwen2.5:14b sel / abstain / compound |
|---|---|---|
| 0.7 | 27.3% / 29.0% / 0.0% | 28.6% / 32.3% / 3.2% |
| 0.8 *(shipped)* | 33.3% / 22.6% / 0.0% | 26.1% / 25.8% / 6.5% |
| 0.9 | 37.5% / 22.6% / 0.0% | 26.3% / 38.7% / 3.2% |

Two honest reads. First, on n=31 with full trajectory divergence the frontier is
**noisy and non-monotonic**: moving `T_HIGH` across a 0.2-wide band shifts every metric
by only 1-2 tasks, and qwen2.5:14b's compound rate is actually lowest at *both* 0.7 and
0.9 and highest at the shipped 0.8 - i.e. within noise, not a tuned optimum. The suite
is too small to resolve fine threshold differences, and 0.8 is not a knife-edge because
there is no sharp edge here to sit on. Second, the sweep connects back to §3: the
shrinkage ceiling is 0.75, so its "block every high-risk action" behaviour is entirely
an artefact of `0.8 > 0.75`. Drop `T_HIGH` to 0.7 and shrinkage would stop blocking
everything - which is another way of saying shrinkage and `T_HIGH` are the same knob.

A note on discrimination, since it bounds all of the above. Pooled across the four
models, the per-trial mean calibrated success probability barely separates successful
retail trials from failed ones (AUC ≈ 0.53; 0.67 on qwen2.5:14b down to below chance,
0.39, on llama3.1:8b). The gate works when it works by *abstaining broadly*, not by
finely ranking good actions above bad ones. Raising that discrimination is the real
open problem; a threshold or a shrinkage constant cannot manufacture signal that the
score does not carry.

## 9. Second real domain: τ-bench airline (50 tasks)

§6 changed the *substrate* (MCP filesystem) to check the retail finding wasn't
benchmark-specific. This section changes the *domain* while staying on real τ-bench: the
**airline** suite (50 tasks, flights and reservations), vendored verbatim from the same
upstream at the same pinned revision. Here the irreversible actions are bookings,
cancellations, and reservation edits; grading replays the ground-truth actions and diffs
the reservation + user state, exactly like the retail order check. Compound failure, all
four models:

| Model (airline, n=50) | Compound: Vanilla → Compass → +Shrink | Selective success: Vanilla → Compass | Compass abstain |
|---|---|---|---|
| gpt-4o-mini | 78% → **28%** → (n/a) | 6.0% → 31.8% | 56% |
| qwen2.5:14b | 28% → 42% → **0%**    | 22.0% → 20.6% | 32% |
| qwen2.5:7b  | 24% → 24% → **0%**    | 20.0% → 27.3% | 34% |
| llama3.1:8b | 6% → **0%** → **0%**  | 34.0% → 38.8% | 2% |

Airline reproduces every load-bearing claim from the retail arc, on a second real domain:

1. **The frontier model needs the gate most here, and it works - but by abstention, not
   discrimination.** Unaided, gpt-4o-mini destroys the *wrong* reservation on **78%** of
   tasks (every booking is irreversible, so overacting is punished hard). Baseline Compass
   cuts that to 28% and lifts task success 6% → 34%. But this time its verbalized confidence
   is as miscalibrated as the local models' (ECE 0.62, mean confidence 0.96) - unlike retail,
   where its confidence carried signal (§0). So the 28% is bought with **56% abstention** and
   trajectory decay, not an honest early score, and 28% is **not zero**: 14 of 50
   confident-wrong bookings still clear the gate. Same model, different domain, different
   failure mode.

2. **qwen2.5:14b gets worse under baseline Compass again** (28% → 42%), the exact regression
   from §1. That it recurs on a different real domain is the confirmation that §2's diagnosis
   is a property of the model, not the retail suite: flat ~1.0 confidence leaves the first
   high-risk action ungated, and the persistent agent takes more destructive steps than timid
   vanilla. qwen2.5:7b is unmoved (24% → 24%).

3. **Shrinkage drives compound to 0% on all three local models**, by the §3 categorical block
   - the 0.75 ceiling sits under `T_HIGH = 0.8`, so no booking or cancellation ever executes.
   The cost is **50-76% abstention** on the Qwens, and shrinkage again posts the *highest*
   task success (38% on both Qwens), because on airline not-acting is frequently the correct
   move.

4. **llama3.1:8b is the mirror image of its MCP collapse.** On MCP it degenerated to 96.8%
   abstention for 0% selective success (§6). On airline it is the best-behaved model on the
   board: it rarely destroys anything unaided (6%), Compass cleans the rest to 0% at just **2%
   abstention**, and selective success actually rises (34% → 38.8%). The timidity that made it
   useless on MCP makes the gate nearly free here. The failure mode is domain-specific, which
   is the strongest single argument in this document against any one-line "Compass does X"
   claim.

## 10. Testing the "precondition check" hypothesis - a negative result

Every section above ends at the same wall: the gate is safe by *abstaining broadly*,
not by ranking good high-risk actions above bad ones (pooled discrimination AUC ≈ 0.53,
§8). The standing hypothesis - written into this document's own roadmap and DESIGN.md -
was that an *earlier, structural* signal would fix that: check a high-risk action's
**preconditions** before it fires. Does the entity it targets actually exist in what the
agent has seen? Is the destination it writes to something the environment offered? Did
the agent read the entity before writing it? These are computable from the trajectory
*at the moment of action*, before trajectory decay, so if they carried signal they would
be exactly the early honest score the gate lacks.

They do not. `analysis/discrimination.py` computes three such preconditions from the
committed trial traces - **no new model calls** - and scores their discrimination against
both trial success and the compound-failure (clean-vs-mutated) outcome, pooled across the
four models on both real domains:

| Signal (AUC vs clean-not-compound) | Retail (n=377) | Airline (n=200) |
|---|---|---|
| Shipped gate (mean calibrated success_prob) | 0.522 | 0.628 |
| Target grounding (id seen in a prior observation?) | 0.482 | 0.478 |
| Destination grounding (write-target offered by the env?) | 0.487 | 0.478 |
| Read-before-write (entity read before it was written?) | 0.545 | 0.632 |

Grounding is **at chance on both domains**, and the tell is sharper than the AUC: the
compound-failure rate *given* an ungrounded high-risk action is **lower** than the base
rate (4.8% vs 10.9% retail; 9.1% vs 23.5% airline). Acting on an id the agent never saw
is, if anything, *less* dangerous. Read-before-write is the only precondition ever above
chance, and it merely **ties** the shipped score (0.632 vs 0.628 on airline) while
falling below chance on retail's llama - it re-encodes trajectory diligence the
step-count penalty already captures, not new signal.

The reason is diagnosable, and airline makes it vivid. gpt-4o-mini destroys the *wrong*
reservation on 78% of unaided airline tasks (§9) - a textbook "wrong target" failure that
grounding should catch. It doesn't, because the wrong reservation is almost always one
the agent **did** observe: the user has several reservations, all surfaced, and the agent
cancels the wrong grounded one. The compound-failure mode is not a hallucinated target;
it is a **semantically wrong action on a correctly-grounded target**. No structural
precondition can see that - separating "cancel reservation A" from "cancel reservation B"
when both exist and both were observed requires modelling *which one the user asked for*,
i.e. a semantic match between the action and the request. That is the learned-probe
direction DESIGN.md flags as a Phase 4 stretch, and this section is the evidence that it
is the *only* remaining lever - the cheap structural fix is now ruled out, not assumed
away.

```bash
uv run python analysis/discrimination.py   # reproduces both tables, from traces only
```

## 11. The lever that is left: a semantic action-vs-request match

§10 left exactly one candidate standing - a signal for whether a high-risk action
matches what the user actually asked for, since the failure is a semantically wrong
action on a correctly-grounded target. This section tests a **zero-dependency lexical
proxy** for that match, again from the committed traces: TF-IDF cosine affinity between
the original request and (a) the observation that describes the entity the action
targets, (b) the action's own reasoning + args, and (c) the weakest-justified argument
(`analysis/semantic_probe.py`). Then a hand-rolled logistic regression combines those
with the step-time confidence and trajectory features, scored by **held-out**
cross-validation (`analysis/learned_probe.py`).

The single features are not a robust gate - they carry strong signal in one cell and
mislead in others. `min_arg_justif` scores AUC **0.889** at separating clean from
compound on gpt-4o-mini retail, but sits at or below chance on the Qwens and across all
of airline, where long requests share generic airport/cabin vocabulary with any
reservation. No lexical feature discriminates on its own everywhere.

The learned combination is the honest test, on the 220 trials that actually take a
high-risk action (88 compound), out-of-fold, averaged over 25 seeds:

| Held-out AUC (clean > compound) | Pooled | Retail | Airline |
|---|---|---|---|
| Shipped rule-based gate (mean success_prob) | 0.499 | 0.399 | **0.604** |
| Learned probe: confidence + trajectory only | 0.548 | 0.432 | 0.366 |
| Learned probe: **+ semantic affinities** | **0.606** | **0.537** | 0.542 |

Three honest reads. First, the semantic affinities do add **real held-out signal**: on
retail they lift the probe from below-chance (0.43) to 0.54, and on the high-risk subset
the shipped rule-based gate is itself only 0.399 there - so 0.40 → 0.54 out-of-fold is
the first within-domain gain anything has shown over the shipped score, and it confirms
§10's diagnosis that the lever is semantic, not structural. Second, it is **not a solved
gate**: 0.54 within-domain is weak, and on airline the shipped rule (0.604) still beats
the probe (0.542) because the lexical proxy is too crude there. Third, do not be fooled
by the pooled column - the trajectory-only probe reaches 0.548 pooled while scoring
*below* chance in both domains, because the domain dummy lets it exploit the different
compound base rates; only the affinity probe clears chance *within* each domain, which is
the number that would matter to a real gate.

So the direction §10 pointed at is correct and now has held-out evidence behind it, but a
lexical stand-in under-delivers as an implementation. That left one question: is the 0.54
ceiling the *idea* (a semantic match just doesn't carry enough signal) or the *proxy*
(TF-IDF is too shallow to extract it)? To answer it, `analysis/embed_probe.py` swaps
TF-IDF for genuine sentence embeddings (`nomic-embed-text` via a local Ollama server)
behind the **identical** feature interface and CV harness - only the cosine backend
changes, so any difference is the embedding, not the pipeline:

| Held-out AUC (clean > compound), full probe | Pooled | Retail | Airline |
|---|---|---|---|
| Shipped rule-based gate | 0.499 | 0.399 | 0.604 |
| Probe with **TF-IDF** affinities | 0.606 | 0.537 | 0.542 |
| Probe with **embedding** affinities | **0.698** | **0.633** | **0.659** |

It was the proxy. Real embeddings lift the probe to **0.633 retail / 0.659 airline**
out-of-fold - the first signal in this entire document to beat the shipped rule-based gate
on *both* real domains at once, and by a clear margin (retail 0.40 → 0.63, airline 0.60 →
0.66). Because the base features are byte-identical between the two probe rows, the whole
gain is attributable to the affinity features being embedding-based rather than lexical.
One caveat kept honest: the *univariate* embedding affinities are still weak (target AUC
~0.28, min-justification ~0.37) - the signal lives in the learned *combination*, so the
claim is that the embedding representation lets the classifier extract a consistent
multivariate signal, not that any single affinity feature is a good gate on its own.

The obvious worry is that stratified CV over random *trials* mixes tasks, so a probe could
be memorizing task-specific patterns rather than learning a transferable signal. The test
that settles it is **cross-domain**: train on one domain, predict compound failure on the
other, which shares no tasks, tools, or vocabulary. The domain dummy is dropped (constant
within each set), standardization uses train stats on test:

| Cross-domain AUC (train → test) | retail → airline | airline → retail |
|---|---|---|
| Probe with TF-IDF affinities | 0.489 | 0.568 |
| Probe with embedding affinities | **0.668** | 0.566 |

The embedding probe trained **only on retail** scores 0.668 on airline - essentially its
own within-domain airline number (0.659), across a disjoint task distribution - while
TF-IDF collapses to chance (0.489) in the same direction. That is direct evidence the
embedding signal is a transferable "does this action match the request" signal, not task
memorization, and it is the sharpest single line separating the embedding from the lexical
proxy. Transfer is not symmetric: airline → retail is only ~0.57 for both backends, so a
probe trained on the harder domain generalizes less well - honest, and a reason to fit on
the broadest available task mix if this is productionized.

This is the first result that plausibly *earns* a place in the pipeline, and it is also
where the honest cost shows up: it needs an embedding call per high-risk action at runtime
and a probe fit on trajectory data, so wiring it in is a real Phase 4 change (behind a
flag, with the aggregator staying locked as the default) - not a free win. The pipeline is
still **unchanged** as the default by choice, but the probe is now **wired behind a flag**:
`build_compass_agent(..., calibration="probe")` scores high-risk actions with a shipped
`SemanticProbe` (`compass/probe.py`, weights fit by `scripts/fit_probe.py`), falling back
to the rule-based score whenever the embedding backend is absent. The runtime feature
extractor is domain-agnostic and shared with training, and it reproduces the result
out-of-fold (CV AUC 0.66 on the agnostic features). What is left is the end-to-end proof -
re-running the suites under the flag and tuning a probe-specific operating point, since it
currently inherits the rule-based `T_HIGH`.

```bash
uv run python analysis/semantic_probe.py   # univariate TF-IDF affinity AUCs, per model/domain
uv run python analysis/learned_probe.py    # held-out CV, TF-IDF backend: base vs +affinity
uv run python analysis/embed_probe.py      # held-out CV + cross-domain, embedding backend (needs Ollama + nomic-embed-text)
uv run python scripts/fit_probe.py         # refit the shipped probe (compass/probe_weights.json) from the traces
```

## Takeaway

Compass's policy machinery is sound; its safety depends entirely on the
aggregator handing it *honest* success probabilities. What it needs depends on the
model's failure mode, and the four models span the range:

- **When confidence carries signal** (gpt-4o-mini), baseline Compass already cuts
  compound failures by two thirds, and the verification pass does most of that work
  (§7). No variant needed.
- **When it collapses to a constant** (qwen2.5:7b and 14B), the only real early signal
  is gone and trajectory features arrive too late to gate the first destructive action
  (§2). Baseline Compass only halves compound failures; the base-rate shrinkage prior
  gets to 0%, but by *categorically blocking every high-risk action* (§3), not by
  gating intelligently. The same block drives both models to 0%, so it is at least not
  a per-model tuning artefact - it is the same blunt switch.
- **When the model is already timid** (llama3.1:8b), it rarely acts destructively on
  retail, so there is little to gate. On the harder MCP suite the same timidity makes
  Compass abstain on 96.8% of tasks for 0% selective success (§6) - safe and useless. But
  on airline that same timidity is a *virtue*: 6% baseline compound, cleaned to 0% at only
  2% abstention with selective success rising (§9). Whether Compass makes llama both safe
  and productive is entirely domain-dependent - not on MCP, yes on airline.

Safety costs coverage in every regime, and the honest accounting is blunter than the
headline: on the weak models the "0% compound" is bought by refusing high-risk
execution outright (shrinkage) or by near-total abstention (llama), not by a score that
tells good actions from bad - pooled discrimination AUC is ≈ 0.53 (§8). On a model that
is already careful (gpt-4o-mini, §6) the coverage cost is paid with no safety benefit at
all. "Zero" means zero destructive actions on these tasks *because the agent mostly
stopped acting*, not a proof of calibrated safety. The real open problem (Phase 4+) is an
*earlier, higher-discrimination* honest signal so the gate can keep coverage instead of
buying safety with blanket refusal - and §10 narrows what that signal can be: the obvious
structural fix (precondition checks - target/destination grounding, read-before-write)
was tested against the traces and does **not** discriminate, because the failure mode is a
semantically wrong action on a correctly-grounded target. What is left is a semantic match
between the proposed action and the user's actual request - and §11 lands it: a lexical
proxy adds real held-out signal on retail (0.40 → 0.54) but is too crude to beat the
shipped rule everywhere, whereas swapping in genuine sentence embeddings lifts a held-out
probe to **0.63 retail / 0.66 airline** - the first signal to beat the shipped gate on
both real domains at once. The learned semantic probe is no longer a hypothesis; it is the
first thing that has earned a path into the pipeline, at the cost of a runtime embedding
call and a Phase 4 productionization behind a flag.

## Reproduce

```bash
# frontier baseline
uv run python scripts/run_tau_eval.py --provider openai --model gpt-4o-mini
# weak-model baseline
./scripts/run_local_gpu.sh qwen14
# weak-model shrinkage variant
uv run python scripts/run_tau_eval.py --provider ollama \
  --model qwen2.5:14b --calibration shrinkage --conditions compass
# second real τ-bench domain: airline, 50 tasks (§9)
uv run python scripts/run_airline_eval.py --provider openai --model gpt-4o-mini
uv run python scripts/run_airline_eval.py --provider ollama --model qwen2.5:14b
# cross-domain: custom filesystem MCP suite, 31 tasks (Phase 3)
uv run python scripts/run_mcp_eval.py --provider ollama --model qwen2.5:14b
uv run python scripts/run_mcp_eval.py --provider openai --model gpt-4o-mini  # the boundary case
# §7 verification ablation (compass with SELF_VERIFY + confirm stripped)
uv run python scripts/run_mcp_eval.py --provider ollama --model qwen2.5:14b \
  --no-verification --conditions compass
# §8 T_HIGH sensitivity sweep (does NOT re-tune the shipped 0.8)
uv run python scripts/run_mcp_eval.py --provider ollama --model qwen2.5:14b \
  --conditions compass --t-high 0.7
# same bridge, real off-the-shelf servers (official filesystem + GitHub)
uv run python scripts/mcp_real_servers.py            # official filesystem MCP
uv run python scripts/mcp_real_servers.py --github   # GitHub MCP (needs a token in .env)
# §10 precondition-check discrimination probe (from committed traces, no model calls)
uv run python analysis/discrimination.py
```
