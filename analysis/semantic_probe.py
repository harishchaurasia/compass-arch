"""Offline SEMANTIC probe: does an action-vs-request affinity feature carry the
discrimination signal that structural preconditions (analysis/discrimination.py,
FINDINGS.md section 10) did not?

Section 10 ruled out the cheap structural fix: the compound-failure mode is a
semantically wrong action on a *correctly grounded* target - cancelling the wrong
(but real) reservation, or exchanging to the wrong (but real) item. Catching that
needs a signal of whether the action matches what the user actually asked for.

This probe builds a zero-dependency *lexical proxy* for that semantic match -
TF-IDF cosine affinity between the original request and (a) the observation that
describes the entity the high-risk action targets, and (b) the action itself -
and measures each feature's univariate discrimination the same way section 10
did (held-out is unnecessary for a univariate score; these are not fit to the
labels). Run from the committed traces, NO model calls.

Read the result honestly: TF-IDF is a lexical stand-in for a semantic quantity.
If an affinity feature clears the shipped score, a learned probe over these
features is worth wiring behind a flag. If it is flat, the signal likely needs
real embeddings (a local model), which is the next rung - not proof none exists.

Usage:  uv run python analysis/semantic_probe.py
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# Reuse the domain scaffolding, AUC, and id tokeniser from the structural probe.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from discrimination import BASE_MODELS, DOMAINS, Domain, _id_tokens, auc  # noqa: E402

DB = Path(__file__).resolve().parents[1] / "results" / "trials.db"

_WORD_RE = re.compile(r"[a-z0-9_]+")
# High-frequency filler that would dominate a raw cosine without adding meaning.
_STOP = frozenset(
    "the a an and or to of for you your want i is are be with on in at it this that "
    "would like need please can could my me we our as from".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class Tfidf:
    """Minimal TF-IDF cosine over a fixed corpus of documents. Hand-rolled so
    the probe stays numpy-free; the corpus is every request + observation in the
    analysed trials, so IDF reflects this suite's own vocabulary."""

    def __init__(self, docs: list[list[str]]):
        n = len(docs)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        # Smoothed IDF; +1 keeps a term that appears everywhere at a small positive weight.
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}

    def cosine(self, a: str, b: str) -> float:
        va, vb = self._vec(_tokens(a)), self._vec(_tokens(b))
        if not va or not vb:
            return 0.0
        dot = sum(va[t] * vb.get(t, 0.0) for t in va)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        return dot / (na * nb) if na and nb else 0.0


def _observations(trace: dict) -> list[str]:
    """Tool-return observations, in order - the environment's answers to the agent."""
    out = []
    for m in trace.get("messages", []):
        if not isinstance(m, dict):
            continue
        c = str(m.get("content", ""))
        if c.startswith("Tool ") and "returned" in c[:40]:
            out.append(c)
    return out


def _request(trace: dict) -> str:
    msgs = trace.get("messages", [])
    return str(msgs[0].get("content", "")) if msgs else ""


@dataclass
class SemProbe:
    model: str
    success: int
    compound: int
    mean_success_prob: float
    target_affinity: float   # request vs the observation describing the action's target
    action_affinity: float   # request vs the action's own reasoning + args
    min_arg_justif: float    # weakest request-vs-introducing-observation affinity across arg ids
    # Step-time features (all knowable AT the first high-risk action, so a probe
    # trained on them could gate in real time - unlike mean_success_prob, a
    # post-hoc trial aggregate kept only as the shipped-score comparison).
    confidence: float        # verbalized confidence at that step
    step_index: int          # position of the first high-risk action in the trajectory
    n_prior_obs: int         # tool-return observations seen before it
    domain: str


def probe_trial(dom: Domain, tfidf: Tfidf, model: str, success: int,
                success_probs: list[float], mutated: list, trace: dict) -> SemProbe | None:
    if not success_probs:
        return None
    steps = trace.get("steps", [])
    first_idx = next((i for i, s in enumerate(steps) if s.get("tool") in dom.high_risk), None)
    if first_idx is None:
        return None  # no high-risk action -> nothing to judge
    first_high = steps[first_idx]

    req = _request(trace)
    obs = _observations(trace)
    args = first_high.get("args", {})
    # Observations that landed before this action: the reads among the prior steps.
    n_prior_obs = sum(1 for s in steps[:first_idx] if s.get("tool") in dom.read_tools)

    # (a) target affinity: the observation that introduced any id this action
    # targets should, for a *right* target, read like the request.
    ids = _id_tokens(args)
    per_id: list[float] = []
    for tok in ids:
        intro = next((o for o in obs if tok in o), None)
        if intro is not None:
            per_id.append(tfidf.cosine(req, intro))
    target_affinity = max(per_id) if per_id else 0.0
    min_arg_justif = min(per_id) if per_id else 0.0

    # (b) action affinity: does the action's own text read like the request?
    action_text = str(first_high.get("reasoning", "")) + " " + json.dumps(args)
    action_affinity = tfidf.cosine(req, action_text)

    return SemProbe(
        model=model,
        success=success,
        compound=int(bool(mutated) and not success),
        mean_success_prob=mean(success_probs),
        target_affinity=target_affinity,
        action_affinity=action_affinity,
        min_arg_justif=min_arg_justif,
        confidence=float(first_high.get("confidence", 0.5)),
        step_index=first_idx,
        n_prior_obs=n_prior_obs,
        domain=dom.name,
    )


def load_rows(dom: Domain) -> list[tuple]:
    con = sqlite3.connect(DB)
    q = """
        SELECT model, success, success_probs, mutated_order_ids, trace
        FROM trials
        WHERE condition = 'compass' AND model IN (%s)
          AND task_id LIKE ? AND trace != '{}'
    """ % ",".join("?" for _ in BASE_MODELS)
    rows = con.execute(q, (*BASE_MODELS, dom.task_prefix + "%")).fetchall()
    con.close()
    return rows


def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:.3f}"


def collect(dom: Domain) -> list[SemProbe]:
    """Feature records for every high-risk Compass trial in `dom` (no printing).
    Shared by report() and analysis/learned_probe.py so both see identical features."""
    rows = load_rows(dom)
    # IDF corpus: every request + observation in this domain's analysed trials.
    corpus: list[list[str]] = []
    parsed = []
    for model, success, sp, mut, tr in rows:
        trace = json.loads(tr)
        corpus.append(_tokens(_request(trace)))
        corpus.extend(_tokens(o) for o in _observations(trace))
        parsed.append((model, success, json.loads(sp), json.loads(mut), trace))
    tfidf = Tfidf(corpus)
    return [p for args in parsed if (p := probe_trial(dom, tfidf, *args)) is not None]


def report(dom: Domain) -> list[SemProbe]:
    probes = collect(dom)
    if not probes:
        print(f"[{dom.name}] no high-risk Compass trials with traces.", file=sys.stderr)
        return []

    print("=" * 74)
    print(f"DOMAIN: {dom.name}  (trials with a high-risk action + trace: {len(probes)})")
    print("=" * 74)
    print("AUC ranking CLEAN trials above compound failures (higher feature = safer):")
    hdr = (f"  {'model':16s} {'n':>4s} {'#comp':>5s}  {'shipped':>7s}  "
           f"{'target':>7s}  {'action':>7s}  {'minjust':>7s}")
    print(hdr)
    pool = {k: [] for k in ("ship", "target", "action", "minjust", "lbl")}
    for m in BASE_MODELS:
        r = [p for p in probes if p.model == m]
        if not r:
            continue
        clean = [1 - p.compound for p in r]
        cols = {
            "ship": [p.mean_success_prob for p in r],
            "target": [p.target_affinity for p in r],
            "action": [p.action_affinity for p in r],
            "minjust": [p.min_arg_justif for p in r],
        }
        for k, v in cols.items():
            pool[k] += v
        pool["lbl"] += clean
        print(f"  {m:16s} {len(r):>4d} {sum(p.compound for p in r):>5d}  "
              f"{_fmt(auc(cols['ship'], clean)):>7s}  {_fmt(auc(cols['target'], clean)):>7s}  "
              f"{_fmt(auc(cols['action'], clean)):>7s}  {_fmt(auc(cols['minjust'], clean)):>7s}")
    print(f"  {'POOLED':16s} {len(pool['lbl']):>4d} {sum(1 - y for y in pool['lbl']):>5d}  "
          f"{_fmt(auc(pool['ship'], pool['lbl'])):>7s}  {_fmt(auc(pool['target'], pool['lbl'])):>7s}  "
          f"{_fmt(auc(pool['action'], pool['lbl'])):>7s}  {_fmt(auc(pool['minjust'], pool['lbl'])):>7s}")
    print()
    return probes


def main() -> int:
    if not DB.exists():
        print(f"No trial DB at {DB}", file=sys.stderr)
        return 1
    for dom in DOMAINS:
        report(dom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
