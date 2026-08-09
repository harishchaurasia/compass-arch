"""Guards for the §14 leave-one-model-out validator: the partition must never
leak the held-out model into training, must count the held-out rows correctly,
and must return nan (not a crash or a bogus number) when a held-out slice lacks
a class. A transferable signal across models should still score high held out.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from cross_model_probe import leave_one_model_out  # noqa: E402
from semantic_probe import SemProbe  # noqa: E402


def _probe(model: str, compound: int, signal: float, domain: str = "retail") -> SemProbe:
    """Minimal SemProbe tagged with a source model; target_affinity carries the
    label (high = clean) so the plumbing has something learnable to transfer."""
    return SemProbe(
        model=model, success=1 - compound, compound=compound, mean_success_prob=0.5,
        target_affinity=signal, action_affinity=0.0, min_arg_justif=0.0,
        confidence=0.5, step_index=1, n_prior_obs=1, domain=domain,
    )


def _pool(rng, models, rule=lambda c: 2.0 if c == 0 else -2.0):
    out = []
    for m in models:
        for _ in range(30):
            for c in (0, 1):
                out.append(_probe(m, c, float(rng.normal(rule(c), 0.4))))
    return out


def test_held_out_model_never_leaks_and_counts_match():
    rng = np.random.default_rng(0)
    probes = _pool(rng, ["a", "b", "c", "d"])
    p_auc, s_auc, n, nc = leave_one_model_out(probes, ["target_affinity"], "d")
    # 30 iterations x 2 classes = 60 held-out rows, half compound.
    assert n == 60 and nc == 30
    # same rule in every model -> a probe fit on a,b,c ranks d's clean above compound.
    assert p_auc > 0.9


def test_absent_model_returns_empty():
    rng = np.random.default_rng(1)
    probes = _pool(rng, ["a", "b", "c"])
    p_auc, s_auc, n, nc = leave_one_model_out(probes, ["target_affinity"], "missing")
    assert n == 0 and nc == 0
    assert p_auc != p_auc and s_auc != s_auc  # both nan


def test_single_class_held_out_is_nan_not_a_crash():
    rng = np.random.default_rng(2)
    probes = _pool(rng, ["a", "b", "c"])
    # held-out model "e" has only clean trials -> AUC undefined, must be nan.
    probes += [_probe("e", 0, float(rng.normal(2.0, 0.4))) for _ in range(10)]
    p_auc, s_auc, n, nc = leave_one_model_out(probes, ["target_affinity"], "e")
    assert n == 10 and nc == 0
    assert p_auc != p_auc  # nan, because there are no compound rows to rank against
