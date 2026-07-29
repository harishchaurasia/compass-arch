"""Guards for the §11 probe internals: the TF-IDF cosine and the hand-rolled
logistic regression that produce the held-out AUC numbers cited in FINDINGS.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from embed_probe import Embedder  # noqa: E402
from learned_probe import _fit, _standardize, cross_domain_auc  # noqa: E402
from semantic_probe import SemProbe, Tfidf, _tokens  # noqa: E402


def _probe(domain: str, compound: int, signal: float) -> SemProbe:
    """Minimal SemProbe where target_affinity carries the label (high = clean),
    everything else neutral - lets a cross-domain test exercise the plumbing."""
    return SemProbe(
        model="m", success=1 - compound, compound=compound, mean_success_prob=0.5,
        target_affinity=signal, action_affinity=0.0, min_arg_justif=0.0,
        confidence=0.5, step_index=1, n_prior_obs=1, domain=domain,
    )


class _FakeEmbedder(Embedder):
    """Embedder with a stubbed backend - no Ollama server, no disk cache - so the
    cosine math is testable offline."""

    def __init__(self, vectors: dict):
        self._vectors = vectors
        self._cache = {}
        self._dirty = False
        self.model = "fake"

    def _embed(self, text: str):
        return self._vectors[text]


def test_tokens_drop_stopwords_and_singletons():
    toks = _tokens("You want to cancel the Boston reservation")
    assert "boston" in toks and "cancel" in toks and "reservation" in toks
    assert "the" not in toks and "to" not in toks  # stopwords gone


def test_cosine_identical_is_one_disjoint_is_zero():
    corpus = [_tokens("cancel boston flight"), _tokens("exchange desk lamp brighter")]
    tf = Tfidf(corpus)
    same = tf.cosine("cancel boston flight", "cancel boston flight")
    assert same == 1.0 or abs(same - 1.0) < 1e-9
    assert tf.cosine("cancel boston flight", "exchange desk lamp") == 0.0


def test_cosine_partial_overlap_is_between():
    tf = Tfidf([_tokens("cancel boston flight reservation")])
    v = tf.cosine("cancel boston flight", "boston flight change")
    assert 0.0 < v < 1.0


def test_embedder_cosine_math():
    emb = _FakeEmbedder({
        "same": [1.0, 0.0, 0.0],
        "same2": [2.0, 0.0, 0.0],   # parallel -> cosine 1
        "orth": [0.0, 1.0, 0.0],    # orthogonal -> cosine 0
        "opp": [-1.0, 0.0, 0.0],    # opposite -> cosine -1
    })
    assert abs(emb.cosine("same", "same2") - 1.0) < 1e-9
    assert abs(emb.cosine("same", "orth") - 0.0) < 1e-9
    assert abs(emb.cosine("same", "opp") + 1.0) < 1e-9


def test_cross_domain_auc_transfers_a_learnable_signal():
    # same rule in both domains (clean trials have high target_affinity), so a
    # probe trained on one should score the other near-perfectly.
    rng = np.random.default_rng(0)
    def make(domain):
        out = []
        for _ in range(40):
            out.append(_probe(domain, 0, float(rng.normal(2, 0.4))))   # clean, high signal
            out.append(_probe(domain, 1, float(rng.normal(-2, 0.4))))  # compound, low signal
        return out
    auc = cross_domain_auc(make("retail"), make("airline"), ["target_affinity"])
    assert auc > 0.9  # signal transfers across domains


def test_cross_domain_auc_chance_when_signal_is_domain_specific():
    # signal is inverted between domains -> a probe trained on one misreads the other.
    rng = np.random.default_rng(1)
    retail = [_probe("retail", c, float(rng.normal(2 if c == 0 else -2, 0.4)))
              for _ in range(40) for c in (0, 1)]
    airline = [_probe("airline", c, float(rng.normal(-2 if c == 0 else 2, 0.4)))
               for _ in range(40) for c in (0, 1)]
    auc = cross_domain_auc(retail, airline, ["target_affinity"])
    assert auc < 0.2  # anti-correlated, as expected


def test_logreg_separates_a_linearly_separable_set():
    # one informative feature + bias; probe should recover a positive weight on it
    rng = np.random.default_rng(0)
    n = 200
    x0 = np.concatenate([rng.normal(-2, 0.5, n // 2), rng.normal(2, 0.5, n // 2)])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    raw = x0.reshape(-1, 1)
    (xtr,) = _standardize(raw)  # standardizes and adds a bias column
    w = _fit(xtr, y)
    # positive weight on the feature; predictions rank the two classes apart
    assert w[0] > 0
    preds = 1.0 / (1.0 + np.exp(-xtr @ w))
    assert preds[y == 1].mean() > preds[y == 0].mean()
