# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project is pre-1.0,
so the public API may still change between minor versions.

## [Unreleased]

### Added
- **Public API.** `from compass import get_model, build_compass_agent, run` - a one-call
  entry point (`compass.run`) that gates your own LangChain tools without touching the
  LangGraph state. Runnable end-to-end in `examples/gate_your_agent.py`.
- **Second real domain.** The full τ-bench **airline** split (50 tasks) alongside retail,
  vendored from upstream. `run_airline_eval.py` plus a domain-parametric `run_trial`.
- **Leaderboard.** `LEADERBOARD.md`, generated from the trial DB by `scripts/leaderboard.py`
  and kept in sync by CI against a committed aggregate snapshot (`results/leaderboard_data.json`).
- **Observability.** Gate decisions and abstentions log on the `compass` logger
  (`logging.getLogger("compass")`), silent by default. `GateResult` now also reports
  per-step `risk_levels`.
- **Contribution scaffolding.** Issue/PR templates and a "Good first contributions" ladder.
- Typing marker (`compass/py.typed`, PEP 561) so downstream users get types.
- **Discrimination probe.** `analysis/discrimination.py` tests, from the committed traces
  (no model calls), whether structural precondition checks would fix the weak gate. They
  don't - target/destination grounding and read-before-write score at chance on both real
  domains (`FINDINGS.md` §10). Documents *why* (the failure is a semantically wrong action
  on a correctly-grounded target) and narrows the roadmap to the learned semantic probe.
- **Learned semantic probe (offline).** `analysis/semantic_probe.py` + `analysis/learned_probe.py`
  build a zero-dependency lexical proxy for an action-vs-request match and score a
  hand-rolled logistic regression by held-out cross-validation. The semantic features add
  real out-of-fold discrimination on retail (0.40 → 0.54 AUC) but a lexical proxy is too
  crude to beat the shipped rule on airline (`FINDINGS.md` §11).
- **Embedding backend closes it.** `analysis/embed_probe.py` swaps TF-IDF for
  `nomic-embed-text` sentence embeddings (local Ollama) behind the same feature interface
  and CV harness; held-out AUC rises to 0.63 retail / 0.66 airline - the first signal to
  beat the shipped rule-based gate on both real domains at once, and it transfers across
  domains (train retail → test airline 0.67) where the lexical proxy collapses to chance.
  Embeddings are cached to a gitignored file so reruns are instant.
- **Semantic probe wired behind a flag.** `build_compass_agent(..., calibration="probe")`
  scores high-risk actions with the embedding probe (`compass/probe.py`, fitted weights in
  `compass/probe_weights.json` via `scripts/fit_probe.py`). The rule-based aggregator stays
  the locked default; the probe falls back to it for any step whenever the embedding backend
  is unavailable, so enabling the flag can never fail harder than default. Inference is pure
  Python (no numpy in the core); only the embedding call reaches out (stdlib HTTP to Ollama).
  `calibration_shrink=True` is now a synonym for `calibration="shrink"`.
- **Probe proven end-to-end.** Run live on gpt-4o-mini via the eval runners
  (`--calibration probe`), the probe gate cuts compound failure 18.3% → 0.9% (retail) and
  28% → 8% (airline). A threshold sweep (`--t-high 0.6`) shows the safety is discrimination,
  not caution: on retail the probe strictly dominates the rule-based gate (less abstention,
  less compound, higher selective success at once). Airline is far safer at low coverage but
  does NOT dominate at any swept threshold - the one that recovers coverage (0.4) pushes
  compound above baseline, so its similar offline AUC (0.66) does not buy a dominating
  operating point. Dominance is domain-specific (`FINDINGS.md` §12).
- **Probe holds across all four models.** Live runs on qwen2.5:14b/7b and llama3.1:8b (both
  suites) drive retail compound to near-zero on every model and fix the qwen2.5:14b regression
  where baseline Compass made things worse (18.3% → 0.0% retail, 42% → 18% airline), by scoring
  the action-vs-request match instead of the model's flat confidence (`FINDINGS.md` §13).
  Caveat recorded: the shipped probe is fit on these models' traces, so the out-of-sample
  evidence remains §11's held-out CV and cross-domain transfer.

### Changed
- CI now runs on a Python 3.11 / 3.12 matrix and verifies the leaderboard is in sync.
- Vanilla baseline tolerates tools that raise (`ToolNode(handle_tool_errors=True)`), so a
  single malformed call no longer aborts a trial.

### Known limitations
- The gate discriminates weakly (pooled AUC ~0.53); its safety comes largely from broad
  abstention, not from ranking good actions above bad. See `FINDINGS.md`. This is the primary
  reason Compass is not yet a production safety component.
