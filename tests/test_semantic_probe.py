"""Guards for the §11 probe internals: the TF-IDF cosine and the hand-rolled
logistic regression that produce the held-out AUC numbers cited in FINDINGS.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from embed_probe import Embedder  # noqa: E402
from learned_probe import _fit, _standardize  # noqa: E402
from semantic_probe import Tfidf, _tokens  # noqa: E402


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
