"""Fit the shipped semantic calibration probe (compass/probe_weights.json).

Trains the logistic probe FINDINGS section 11 validated, using the exact runtime
feature extractor (compass.probe.extract_features) so training and inference cannot
drift. Reads the committed traces (results/trials.db), computes features for the
first high-risk action of every retail + airline Compass trial, reports a k-fold CV
AUC as a sanity check against the section 11 numbers, then fits on ALL rows and writes
the weights.

The fitted weights are a small aggregate (six coefficients + standardization stats),
not raw trial data, so they are committed - same policy as the leaderboard snapshot.

Embeddings reuse the warm analysis/.embed_cache.json via a thin adapter, so this is
instant after embed_probe.py has run once. Requires numpy (dev/analysis only; the
runtime probe in compass/probe.py is pure Python).

Usage:  uv run python scripts/fit_probe.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from discrimination import AIRLINE, RETAIL, auc  # noqa: E402
from embed_probe import Embedder as _AnalysisEmbedder  # warm-cache embedder  # noqa: E402
from learned_probe import _fit, _standardize  # noqa: E402

from compass.probe import FEATURE_ORDER, extract_features  # noqa: E402
from compass.schemas import CompassAction, CompassStep  # noqa: E402

DB = ROOT / "results" / "trials.db"
OUT = ROOT / "compass" / "probe_weights.json"
EMBED_MODEL = "nomic-embed-text"
BASE_MODELS = ("gpt-4o-mini", "qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b")
DOMAINS = (RETAIL, AIRLINE)
N_FOLDS = 5


class _Adapter:
    """Exposes the analysis embedder (warm disk cache) under the compass.probe
    Embedder protocol (`.embed`, `.model_id`)."""

    model_id = EMBED_MODEL

    def __init__(self) -> None:
        self._e = _AnalysisEmbedder(EMBED_MODEL)

    def embed(self, text: str) -> list[float]:
        return self._e._embed(text)

    def save(self) -> None:
        self._e.save()


def _step_from(drow: dict) -> CompassStep:
    return CompassStep(
        reasoning=str(drow.get("reasoning", "")),
        action=CompassAction(
            tool=drow.get("tool"),
            args=drow.get("args") or {},
            final_answer=drow.get("final_answer"),
        ),
        confidence=float(drow.get("confidence", 0.5)),
        risk_level=drow.get("risk_level") if drow.get("risk_level") in ("low", "medium", "high") else "medium",
    )


def _rows(embedder: _Adapter):
    con = sqlite3.connect(DB)
    q = """
        SELECT model, success, mutated_order_ids, trace
        FROM trials
        WHERE condition = 'compass' AND model IN (%s)
          AND task_id LIKE ? AND trace != '{}'
    """ % ",".join("?" for _ in BASE_MODELS)
    X, y = [], []
    for dom in DOMAINS:
        for model, success, mut, tr in con.execute(q, (*BASE_MODELS, dom.task_prefix + "%")):
            trace = json.loads(tr)
            steps = trace.get("steps", [])
            first_idx = next((i for i, s in enumerate(steps) if s.get("tool") in dom.high_risk), None)
            if first_idx is None:
                continue
            msgs = trace.get("messages", [])
            request = str(msgs[0]["content"]) if msgs else ""
            observations = [
                str(m["content"]) for m in msgs
                if isinstance(m, dict) and str(m.get("content", "")).startswith("Tool '")
                and "returned" in str(m["content"])[:40]
            ]
            n_prior_tool_calls = sum(1 for s in steps[:first_idx] if s.get("tool"))
            feats = extract_features(
                request, observations, _step_from(steps[first_idx]),
                n_prior_tool_calls, first_idx, embedder,
            )
            X.append(feats)
            y.append(int(bool(json.loads(mut)) and not success))
    con.close()
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _cv_auc(X: np.ndarray, y: np.ndarray, seeds: int = 25) -> float:
    aucs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        folds = [[] for _ in range(N_FOLDS)]
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            for i, j in enumerate(idx):
                folds[i % N_FOLDS].append(j)
        preds = np.zeros(len(y))
        for fold in folds:
            fold = np.array(sorted(fold))
            mask = np.ones(len(y), dtype=bool)
            mask[fold] = False
            xtr, xte = _standardize(X[mask], X[fold])
            w = _fit(xtr, y[mask])
            preds[fold] = 1.0 / (1.0 + np.exp(-xte @ w))
        aucs.append(auc(list(1 - preds), [1 - int(v) for v in y]))
    return float(np.mean(aucs))


def main() -> int:
    if not DB.exists():
        print(f"No trial DB at {DB}", file=sys.stderr)
        return 1
    embedder = _Adapter()
    X, y = _rows(embedder)
    embedder.save()
    print(f"Fitting on {len(y)} high-risk actions ({int(y.sum())} compound).")
    print(f"Sanity CV AUC (should track FINDINGS section 11 ~0.63-0.66): {_cv_auc(X, y):.3f}")

    # Final fit on ALL rows: standardize, fit, store mean/std + weights (bias last).
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = (X - mean) / std
    z_aug = np.hstack([z, np.ones((len(z), 1))])
    w = _fit(z_aug, y)

    spec = {
        "embed_model": EMBED_MODEL,
        "feature_order": list(FEATURE_ORDER),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),  # len = n_features + 1, bias last
        "n_train": int(len(y)),
        "n_compound": int(y.sum()),
        "fit_date": date.today().isoformat(),
    }
    OUT.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")

    # Operating-point diagnostic: where do the probe's success probabilities land,
    # and how would the locked T_HIGH = 0.8 gate behave on them? The probe inherits
    # that threshold, so this says whether 0.8 is sane or needs probe-specific tuning.
    p_clean = 1.0 - 1.0 / (1.0 + np.exp(-z_aug @ w))
    clean_ps = p_clean[y == 0]
    comp_ps = p_clean[y == 1]

    def _q(a):
        return f"min {a.min():.2f}  median {np.median(a):.2f}  max {a.max():.2f}"

    print("\nProbe success_prob (in-sample) by outcome:")
    print(f"  clean actions   ({len(clean_ps)}): {_q(clean_ps)}")
    print(f"  compound actions ({len(comp_ps)}): {_q(comp_ps)}")
    for t in (0.6, 0.8):
        exec_clean = float((clean_ps >= t).mean())
        exec_comp = float((comp_ps >= t).mean())
        print(f"  at threshold {t:.1f}: would EXECUTE {exec_clean:.0%} of clean, "
              f"{exec_comp:.0%} of compound actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
