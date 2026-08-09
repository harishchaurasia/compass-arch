"""Does the embedding probe generalize to a model whose traces it never saw?

FINDINGS.md section 13 drove retail compound to near-zero on all four models with
`--calibration probe`, but with an honest caveat: the shipped probe is FIT on those
same models' traces. The held-out evidence so far is section 11's k-fold CV (folds
drawn at random, so every model still appears in its own training folds) and the
cross-DOMAIN transfer (train retail, test airline). Neither answers the sharper
question a deployer actually has: if I point this probe at a model it was never
trained on, does it still rank clean actions above compound ones?

This script answers it with leave-one-MODEL-out (LOMO) cross-validation. For each
of the four models it fits the logistic probe on the OTHER three and scores
discrimination on the held-out model's trials only - so the held-out model
contributes nothing, not even a fold, to the weights judging it. That is the real
out-of-model-distribution test.

Everything but the partition is shared with learned_probe / embed_probe (same
embedding backend, same _fit/_standardize/auc), so any drop from the section-11
random-fold AUC is attributable to the model shift, not a different pipeline.

Requires the same local Ollama + nomic-embed-text as embed_probe.py; embeddings
come from the shared analysis/.embed_cache.json, so after that script has run once
this is instant and offline.

Usage:  uv run python analysis/cross_model_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discrimination import BASE_MODELS, DOMAINS, auc  # noqa: E402
from embed_probe import Embedder  # noqa: E402
from learned_probe import (  # noqa: E402
    AFFINITY_FEATS,
    BASE_FEATS,
    _fit,
    _matrix,
    _standardize,
)
from semantic_probe import SemProbe, collect  # noqa: E402


def _auc_clean(scores: list[float], probes: list[SemProbe]) -> float:
    """AUC ranking clean (label 1) above compound (label 0) for these probes."""
    return auc(scores, [1 - p.compound for p in probes])


def leave_one_model_out(
    probes: list[SemProbe], feats: list[str], model: str
) -> tuple[float, float, int, int]:
    """Fit on every model EXCEPT `model`; score discrimination on `model`'s trials.

    Standardization uses TRAIN statistics applied to the held-out TEST rows, and
    the domain dummy is kept (both domains appear in train and test, so it only
    shifts the intercept per domain, exactly as the pooled probe does). Returns
    (probe_auc, shipped_auc, n_test, n_compound_test); the shipped bar is the same
    mean-success_prob score, computed on the identical held-out rows for a fair
    same-sample comparison. AUC is nan when the held-out set lacks either class.
    """
    train = [p for p in probes if p.model != model]
    test = [p for p in probes if p.model == model]
    assert all(p.model != model for p in train), "held-out model leaked into train"
    if not test:
        return float("nan"), float("nan"), 0, 0

    ytr = np.array([p.compound for p in train], dtype=float)
    xtr_raw = _matrix(train, feats, add_domain=True)
    xte_raw = _matrix(test, feats, add_domain=True)
    xtr, xte = _standardize(xtr_raw, xte_raw)
    w = _fit(xtr, ytr)
    p_clean = list(1.0 - 1.0 / (1.0 + np.exp(-xte @ w)))

    probe_auc = _auc_clean(p_clean, test)
    shipped_auc = _auc_clean([p.mean_success_prob for p in test], test)
    return probe_auc, shipped_auc, len(test), sum(p.compound for p in test)


def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:.3f}"


def main() -> int:
    try:
        embedder = Embedder()
        embedder.cosine("ping", "pong")  # fail fast if Ollama/model is down
    except Exception as e:  # noqa: BLE001 - surface any backend failure clearly
        print(
            f"Could not reach the embedding backend: {e}\n"
            "Start Ollama and `ollama pull nomic-embed-text`, then run "
            "analysis/embed_probe.py once to warm the cache.",
            file=sys.stderr,
        )
        return 1

    probes: list[SemProbe] = []
    for dom in DOMAINS:
        probes += collect(dom, make_sim=lambda _corpus: embedder)
    embedder.save()
    if not probes:
        print("No high-risk trials with traces.", file=sys.stderr)
        return 1

    feats = BASE_FEATS + AFFINITY_FEATS
    n_comp = sum(p.compound for p in probes)
    print(
        f"High-risk Compass trials (pooled, 4 models): {len(probes)}  "
        f"(compound {n_comp}, clean {len(probes) - n_comp})\n"
    )
    print("Leave-one-MODEL-out: fit on the other 3 models, score the held-out one.")
    print("Probe vs the shipped rule-based score on the SAME held-out trials.\n")
    print(
        f"  {'held-out model':16s} {'n':>4s} {'#comp':>5s}  "
        f"{'probe':>7s}  {'shipped':>7s}  {'delta':>7s}"
    )
    deltas: list[float] = []
    for m in BASE_MODELS:
        p_auc, s_auc, n, nc = leave_one_model_out(probes, feats, m)
        delta = p_auc - s_auc
        if delta == delta:  # not nan
            deltas.append(delta)
        d_str = "  nan" if delta != delta else f"{delta:+.3f}"
        print(
            f"  {m:16s} {n:>4d} {nc:>5d}  {_fmt(p_auc):>7s}  "
            f"{_fmt(s_auc):>7s}  {d_str:>7s}"
        )
    if deltas:
        print(
            f"\n  mean probe-minus-shipped across held-out models: "
            f"{np.mean(deltas):+.3f}"
        )

    # Per-(held-out model, domain): dominance was domain-specific in section 12,
    # so a pooled number can hide a domain where the transfer fails. Small n here
    # is expected; nan means the held-out domain slice lacked a compound or a clean.
    print("\nPer-domain breakdown of the held-out probe AUC (n / #comp in cells):")
    print(f"  {'held-out model':16s} {'retail':>16s}  {'airline':>16s}")
    for m in BASE_MODELS:
        cells = []
        for dom in DOMAINS:
            sub = [p for p in probes if p.model == m and p.domain == dom.name]
            train = [p for p in probes if p.model != m]  # still all other models
            if not sub:
                cells.append(f"{'--':>16s}")
                continue
            ytr = np.array([p.compound for p in train], dtype=float)
            xtr, xte = _standardize(
                _matrix(train, feats, add_domain=True),
                _matrix(sub, feats, add_domain=True),
            )
            w = _fit(xtr, ytr)
            p_clean = list(1.0 - 1.0 / (1.0 + np.exp(-xte @ w)))
            a = _auc_clean(p_clean, sub)
            cells.append(f"{_fmt(a)} (n={len(sub)},c={sum(p.compound for p in sub)})".rjust(16))
        print(f"  {m:16s} {cells[0]}  {cells[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
