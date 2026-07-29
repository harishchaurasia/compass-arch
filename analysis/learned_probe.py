"""Does a LEARNED probe beat the shipped rule-based gate on held-out trials?

The structural preconditions were at chance (analysis/discrimination.py, FINDINGS
section 10). The semantic affinity features (analysis/semantic_probe.py) carry
strong signal on gpt-4o-mini retail but are noisy-to-anti-correlated elsewhere -
no single feature is a robust discriminator. This script asks the only question
that settles whether the learned-probe path is worth wiring behind a flag: does a
logistic regression over ALL the step-time features (confidence + trajectory +
the three affinities) predict compound failure on HELD-OUT trials better than the
shipped mean-success_prob score?

Method, kept deliberately honest for a small, imbalanced sample:
  - Features are all knowable AT the first high-risk action (a real gate could use
    them); the shipped score (mean calibrated success_prob) is a post-hoc trial
    aggregate, included only as the comparison bar.
  - Stratified k-fold CV; the reported AUC is over OUT-OF-FOLD predictions only, so
    the probe is never scored on a trial it trained on. Averaged over many seeds
    because n is small and a single split is noisy.
  - Two probes: full (with affinities) and a confidence+trajectory-only ablation,
    to isolate what the semantic features actually add.

No sklearn: one hand-rolled L2 logistic regression (numpy). No model calls.

Usage:  uv run python analysis/learned_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discrimination import DOMAINS, auc  # noqa: E402
from semantic_probe import SemProbe, collect  # noqa: E402

N_FOLDS = 5
N_SEEDS = 25
L2 = 1.0
LR = 0.3
ITERS = 500

# Feature sets, by name, mapping to SemProbe attributes. "domain" is added as a
# 0/1 dummy separately so a pooled probe can shift its intercept per domain.
BASE_FEATS = ["confidence", "step_index", "n_prior_obs"]
AFFINITY_FEATS = ["target_affinity", "action_affinity", "min_arg_justif"]


def _matrix(probes: list[SemProbe], feats: list[str], add_domain: bool = True) -> np.ndarray:
    cols = [[getattr(p, f) for p in probes] for f in feats]
    if add_domain:
        cols.append([1.0 if p.domain == "airline" else 0.0 for p in probes])  # domain dummy
    return np.array(cols, dtype=float).T


def cross_domain_auc(train: list[SemProbe], test: list[SemProbe], feats: list[str]) -> float:
    """Train on one domain, score compound-failure discrimination on the OTHER.
    The domain dummy is dropped (it is constant within each set, so it would only
    add a useless intercept shift); standardization uses TRAIN stats applied to
    TEST. This is the real generalization test - the test domain shares no tasks,
    tools, or vocabulary with training, so a chance result would mean the probe
    was memorizing task-specific patterns rather than learning a transferable signal."""
    ytr = np.array([p.compound for p in train], dtype=float)
    yte = np.array([p.compound for p in test], dtype=float)
    xtr_raw = _matrix(train, feats, add_domain=False)
    xte_raw = _matrix(test, feats, add_domain=False)
    xtr, xte = _standardize(xtr_raw, xte_raw)
    w = _fit(xtr, ytr)
    p_clean = 1.0 - 1.0 / (1.0 + np.exp(-xte @ w))
    return auc(list(p_clean), [1 - int(v) for v in yte])


def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L2 logistic regression by gradient descent. x is already standardized and
    bias-augmented; returns the weight vector. Bias (last col, all ones) is left
    unregularized."""
    w = np.zeros(x.shape[1])
    reg = np.ones(x.shape[1]) * L2
    reg[-1] = 0.0  # don't penalize the bias
    for _ in range(ITERS):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        grad = x.T @ (p - y) / len(y) + reg * w / len(y)
        w -= LR * grad
    return w


def _standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    out = [(train - mu) / sd] + [(o - mu) / sd for o in others]
    return [np.hstack([m, np.ones((len(m), 1))]) for m in out]  # add bias column


def _stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


def _oof_predictions(x_raw: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Out-of-fold P(clean) for every row: each row is predicted only by folds it
    was not in."""
    preds = np.zeros(len(y))
    for fold in _stratified_folds(y, N_FOLDS, rng):
        mask = np.ones(len(y), dtype=bool)
        mask[fold] = False
        xtr, xte = _standardize(x_raw[mask], x_raw[fold])
        w = _fit(xtr, y[mask])
        preds[fold] = 1.0 / (1.0 + np.exp(-xte @ w))
    return preds  # P(compound); caller inverts to P(clean) for the clean-vs-compound AUC


def evaluate(probes: list[SemProbe], feats: list[str]) -> dict:
    y = np.array([p.compound for p in probes], dtype=float)
    clean = [1 - int(v) for v in y]
    x_raw = _matrix(probes, feats)
    dom = np.array([p.domain for p in probes])

    pooled, per_dom = [], {d: [] for d in set(dom)}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        pc = 1.0 - _oof_predictions(x_raw, y, rng)  # P(clean)
        pooled.append(auc(list(pc), clean))
        for d in per_dom:
            m = dom == d
            per_dom[d].append(auc(list(pc[m]), [clean[i] for i in np.where(m)[0]]))
    return {
        "pooled_mean": float(np.mean(pooled)),
        "pooled_std": float(np.std(pooled)),
        "per_domain": {d: float(np.mean(v)) for d, v in per_dom.items()},
    }


def main() -> int:
    probes: list[SemProbe] = []
    for dom in DOMAINS:
        probes += collect(dom)
    if not probes:
        print("No high-risk trials with traces.", file=sys.stderr)
        return 1

    y = np.array([p.compound for p in probes])
    clean = [1 - int(v) for v in y]
    dom = np.array([p.domain for p in probes])
    print(f"High-risk Compass trials (pooled): {len(probes)}  "
          f"(compound {int(y.sum())}, clean {len(probes) - int(y.sum())})\n")

    # Bar 1: the shipped rule-based score on exactly these trials.
    ship = [p.mean_success_prob for p in probes]
    print("Shipped rule-based gate (mean calibrated success_prob), for reference:")
    print(f"  pooled AUC(clean>compound) = {auc(ship, clean):.3f}")
    for d in sorted(set(dom)):
        m = dom == d
        print(f"    {d:8s} = {auc([ship[i] for i in np.where(m)[0]], [clean[i] for i in np.where(m)[0]]):.3f}")

    # Bar 2 & 3: learned probes, out-of-fold, averaged over seeds.
    print(f"\nLearned probe, held-out AUC (mean over {N_SEEDS} seeds, {N_FOLDS}-fold CV):")
    for label, feats in (
        ("confidence + trajectory only", BASE_FEATS),
        ("+ semantic affinities (full)", BASE_FEATS + AFFINITY_FEATS),
    ):
        r = evaluate(probes, feats)
        dom_str = "  ".join(f"{d}={r['per_domain'][d]:.3f}" for d in sorted(r["per_domain"]))
        print(f"  {label:32s} pooled={r['pooled_mean']:.3f} +/-{r['pooled_std']:.3f}   [{dom_str}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
