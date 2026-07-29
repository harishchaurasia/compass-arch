"""Semantic calibration probe (Phase 4 stretch, opt-in behind a flag).

The locked rule-based aggregator (calibration.py) is safe only by abstaining
broadly - it barely ranks good high-risk actions above bad (pooled discrimination
AUC ~0.53; FINDINGS.md sections 8, 10). FINDINGS section 11 shows the missing signal
is a *semantic* match between the proposed action and the user's request, and that a
logistic probe over sentence-embedding affinities recovers it: held-out AUC 0.63
retail / 0.66 airline, and it transfers across domains (train retail -> test airline
0.67) where a lexical proxy collapses to chance.

This module is the runtime side of that result. It is NOT the default: the rule-based
aggregator stays locked and shipped. When `build_compass_agent(..., calibration="probe")`
is set, high-risk actions are scored by this probe instead, and the gate falls back to
the rule-based score whenever the embedding backend is unavailable, so turning the flag
on can never harder-fail than the default.

Design notes kept honest:
  - Inference is pure Python (one dot product + sigmoid), so the core adds no numpy
    dependency; only the embedding call reaches out (stdlib urllib to a local Ollama).
  - The shipped weights (probe_weights.json) were fit with `nomic-embed-text`; a
    different embedder would put the affinity features on a different scale and
    invalidate them, so the model id is pinned and checked.
  - Features are extracted the SAME way here and in scripts/fit_probe.py (both import
    this module), so training and inference cannot drift.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from compass.schemas import CompassStep

WEIGHTS_PATH = Path(__file__).resolve().parent / "probe_weights.json"

# Canonical feature order. Fit and inference both rely on this exact sequence;
# probe_weights.json stores mean/std/weights aligned to it.
FEATURE_ORDER = (
    "confidence",         # verbalized confidence at the action
    "step_index",         # how deep in the trajectory the action fires
    "n_prior_tool_calls", # tool calls already made (a domain-agnostic diligence proxy)
    "target_affinity",    # request vs the observation describing the action's target
    "action_affinity",    # request vs the action's own reasoning + args
    "min_arg_justif",     # weakest request-vs-introducing-observation affinity across args
)

# Same id shape as analysis/discrimination.py: a whole alphanumeric/underscore run
# carrying a digit (order ids W2378156, user ids yusuf_rossi_9620, item ids keyboard_001).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_HAS_DIGIT = re.compile(r"\d")

_OBS_PREFIX = "Tool '"  # how execute() formats an observation: "Tool 'x' returned: ..."


class ProbeUnavailable(RuntimeError):
    """Raised when the probe cannot score (e.g. the embedding backend is down).
    The agent catches this and falls back to the rule-based aggregator."""


class Embedder(Protocol):
    """Anything that maps text -> vector. `model_id` lets the probe verify the
    shipped weights were fit with the same embedding model."""

    model_id: str

    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Default backend: `nomic-embed-text` on a local Ollama server, with an
    in-process cache so repeated texts (the request, re-seen observations) cost
    one call each."""

    def __init__(self, model_id: str = "nomic-embed-text",
                 url: str = "http://localhost:11434/api/embeddings", max_chars: int = 2000):
        self.model_id = model_id
        self._url = url
        self._max_chars = max_chars
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        text = text[: self._max_chars]
        key = hashlib.sha1(text.encode()).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        body = json.dumps({"model": self.model_id, "prompt": text}).encode()
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                vec = json.loads(resp.read())["embedding"]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            raise ProbeUnavailable(f"embedding call failed: {e}") from e
        self._cache[key] = vec
        return vec


def _id_tokens(value: object) -> set[str]:
    out: set[str] = set()
    if isinstance(value, str):
        out.update(t for t in _TOKEN_RE.findall(value) if _HAS_DIGIT.search(t))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out |= _id_tokens(v)
    elif isinstance(value, dict):
        for v in value.values():
            out |= _id_tokens(v)
    elif isinstance(value, (int, float)):
        out.update(t for t in _TOKEN_RE.findall(str(value)) if _HAS_DIGIT.search(t))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def extract_features(
    request: str,
    observations: list[str],
    step: CompassStep,
    n_prior_tool_calls: int,
    step_index: int,
    embedder: Embedder,
) -> list[float]:
    """Feature vector (in FEATURE_ORDER) for one high-risk action. Shared verbatim
    by training (scripts/fit_probe.py) and inference, so they cannot diverge.

    `observations` are the tool-return strings seen so far; `request` is the user's
    original instruction. Raises ProbeUnavailable if the embedder cannot answer."""
    args = step.action.args if isinstance(step.action.args, dict) else {}

    req_vec = embedder.embed(request)
    per_id: list[float] = []
    for tok in _id_tokens(args):
        intro = next((o for o in observations if tok in o), None)
        if intro is not None:
            per_id.append(_cosine(req_vec, embedder.embed(intro)))
    target_affinity = max(per_id) if per_id else 0.0
    min_arg_justif = min(per_id) if per_id else 0.0

    action_text = f"{step.reasoning} {json.dumps(args)}"
    action_affinity = _cosine(req_vec, embedder.embed(action_text))

    return [
        float(step.confidence),
        float(step_index),
        float(n_prior_tool_calls),
        target_affinity,
        action_affinity,
        min_arg_justif,
    ]


def observations_from_messages(messages: list) -> list[str]:
    """Pull the tool-return observations out of a running message list, matching
    how execute() formats them ("Tool 'name' returned: ...")."""
    out = []
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str) and content.startswith(_OBS_PREFIX) and "returned" in content[:40]:
            out.append(content)
    return out


def request_from_messages(messages: list) -> str:
    return str(getattr(messages[0], "content", "")) if messages else ""


class SemanticProbe:
    """Loads fitted weights and scores a high-risk action -> success probability
    (P the action is NOT a compound failure). Same policy thresholds apply."""

    def __init__(self, spec: dict, embedder: Embedder):
        self.mean = spec["mean"]
        self.std = spec["std"]
        self.weights = spec["weights"]          # len = n_features + 1 (bias last)
        self.feature_order = tuple(spec["feature_order"])
        self.embed_model = spec["embed_model"]
        self.embedder = embedder
        if self.feature_order != FEATURE_ORDER:
            raise ValueError(f"probe weights feature order {self.feature_order} "
                             f"does not match this build's {FEATURE_ORDER}")
        if getattr(embedder, "model_id", None) != self.embed_model:
            raise ValueError(f"probe was fit with embed model {self.embed_model!r} but the "
                             f"embedder provides {getattr(embedder, 'model_id', None)!r}")

    @classmethod
    def load(cls, embedder: Embedder | None = None, weights_path: Path = WEIGHTS_PATH) -> SemanticProbe:
        if not weights_path.exists():
            raise ProbeUnavailable(f"no fitted probe weights at {weights_path}; "
                                   f"run scripts/fit_probe.py to create them")
        spec = json.loads(weights_path.read_text(encoding="utf-8"))
        return cls(spec, embedder or OllamaEmbedder(spec["embed_model"]))

    def success_prob(self, request: str, observations: list[str], step: CompassStep,
                     n_prior_tool_calls: int, step_index: int) -> float:
        feats = extract_features(
            request, observations, step, n_prior_tool_calls, step_index, self.embedder
        )
        z = [(f - m) / s if s else 0.0 for f, m, s in zip(feats, self.mean, self.std)]
        logit = sum(w * x for w, x in zip(self.weights, z)) + self.weights[-1]
        p_compound = 1.0 / (1.0 + math.exp(-logit))
        return 1.0 - p_compound  # success = not a compound failure
