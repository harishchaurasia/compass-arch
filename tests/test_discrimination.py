"""Guards for the discrimination probe's load-bearing metric (analysis/discrimination.py).

The §10 finding rests on the AUC values this helper produces, so pin its
behaviour on known-answer cases and confirm the grounding-token extractor
recognises the id shapes the probe depends on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from discrimination import _id_tokens, auc  # noqa: E402


def test_auc_perfect_separation():
    # every positive scores above every negative → 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0


def test_auc_perfect_inversion():
    assert auc([0.2, 0.1, 0.9, 0.8], [1, 1, 0, 0]) == 0.0


def test_auc_ties_are_half():
    # identical scores across classes → pure ties → 0.5
    assert auc([0.5, 0.5], [1, 0]) == 0.5


def test_auc_degenerate_is_nan():
    # all one class → undefined; helper returns nan (probe prints "nan")
    v = auc([0.9, 0.8], [1, 1])
    assert v != v  # nan


def test_id_tokens_recognises_entity_shapes():
    assert _id_tokens("W2378156") == {"W2378156"}
    assert _id_tokens("yusuf_rossi_9620") == {"yusuf_rossi_9620"}
    # nested list/dict args are flattened
    assert _id_tokens(["keyboard_001", "thermostat_001"]) == {"keyboard_001", "thermostat_001"}
    assert _id_tokens({"order_id": "W123", "item_ids": ["a_1"]}) == {"W123", "a_1"}


def test_id_tokens_ignores_bare_words():
    # names / option words have no digit → not id tokens to ground
    assert _id_tokens("Yusuf") == set()
    assert _id_tokens("clicky") == set()
