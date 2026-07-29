"""Offline discrimination probe: does an entity-grounding precondition carry
signal that verbalized-confidence-driven success_prob does not?

Runs entirely from the committed trial traces (results/trials.db) - NO model
calls. For every retail Compass trial with a trace it:

  1. reproduces the shipped gate's per-trial score (mean calibrated
     success_prob, already stored) and its discrimination AUC, and
  2. computes a *precondition* feature from the trace - whether the entity ids
     a high-risk action targets were actually surfaced by a prior observation
     (or the original request) - and measures that feature's own AUC.

The point is to answer, cheaply and honestly, whether the grounding signal
exists at all before wiring it into the locked pipeline. If an ungrounded
high-risk action (acting on an id the agent never saw) predicts compound
failure, that is an *earlier* signal than trajectory decay and worth building.

Usage:  uv run python analysis/discrimination.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

DB = Path(__file__).resolve().parents[1] / "results" / "trials.db"

# The four models with a full A/B (base condition, no ablation suffix).
BASE_MODELS = ("gpt-4o-mini", "qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b")


@dataclass(frozen=True)
class Domain:
    name: str
    task_prefix: str  # trials.db task_id LIKE prefix
    high_risk: frozenset  # high-risk tool names (mirror of TOOL_RISK, inlined)
    dest_args: frozenset  # arg names naming a write *destination*
    read_tools: frozenset  # low-risk reads that surface an entity's state


# Inlined so the probe reads the DB without importing the agent stack / langchain.
RETAIL = Domain(
    name="retail",
    task_prefix="tau_retail_",
    high_risk=frozenset(
        {
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
        }
    ),
    dest_args=frozenset(
        {
            "new_item_ids",
            "payment_method_id",
            "item_ids",
            "address1",
            "address2",
            "city",
            "state",
            "zip",
            "country",
        }
    ),
    read_tools=frozenset(
        {
            "get_order_details",
            "get_user_details",
            "get_product_details",
        }
    ),
)

AIRLINE = Domain(
    name="airline",
    task_prefix="tau_airline_",
    high_risk=frozenset(
        {
            "book_reservation",
            "cancel_reservation",
            "send_certificate",
            "update_reservation_baggages",
            "update_reservation_flights",
            "update_reservation_passengers",
        }
    ),
    dest_args=frozenset(
        {
            "reservation_id",
            "flights",
            "cabin",
            "passengers",
            "total_baggages",
            "nonfree_baggages",
            "payment_id",
        }
    ),
    read_tools=frozenset(
        {
            "get_reservation_details",
            "get_user_details",
            "search_direct_flight",
            "search_onestop_flight",
            "list_all_airports",
        }
    ),
)

DOMAINS = (RETAIL, AIRLINE)

# An id-like token: a whole alphanumeric/underscore run that contains at least
# one digit - order ids (W2378156), user ids (yusuf_rossi_9620), item ids
# (keyboard_001), payment ids (gift_card_1234). The full run matters: grounding
# tests each token as a substring of what the agent observed, so tokenising down
# to a bare "9620" would let a 4-digit collision ground the whole id. Bare words
# (option names, "Yusuf") have no digit and are ignored - not ids to ground.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_HAS_DIGIT = re.compile(r"\d")


def auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC (fraction of pos>neg pairs, ties at 0.5)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _id_tokens(value: object) -> set[str]:
    """Id-like tokens (whole digit-bearing runs) inside an arg value."""
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


def _observed_text(trace: dict) -> str:
    """Everything the environment surfaced to the agent: the original request
    plus every tool-return observation. This is the pool an id must appear in
    to count as grounded. (Whole-trace pool is a deliberate over-approximation
    - it can only make grounding look *better* than a strict before-this-step
    pool would, so any ungrounded action it flags is unambiguously ungrounded.)
    """
    parts: list[str] = []
    for m in trace.get("messages", []):
        if not isinstance(m, dict):
            continue
        content = str(m.get("content", ""))
        # Confirm/self_verify nudges echo the pending action's args back at the
        # model; counting them as "observed" would ground an id against the very
        # action being judged. Skip them - keep only the request and real returns.
        if content.startswith("You are about to take a HIGH-risk action"):
            continue
        if content.startswith("Low confidence ("):
            continue
        parts.append(content)
    return "\n".join(parts)


@dataclass
class TrialProbe:
    model: str
    success: int
    compound: int  # not success AND mutated something
    mean_success_prob: float
    n_high_risk: int  # high-risk actions the agent proposed
    n_ungrounded: int  # of those, how many targeted an unseen id
    min_grounding: (
        float  # worst grounding ratio across high-risk actions (1.0 = all grounded)
    )
    min_dest_grounding: float  # same, but only destination args
    read_before_write: (
        float  # fraction of high-risk writes preceded by a read of the same entity
    )


def probe_trial(
    dom: Domain,
    model: str,
    success: int,
    success_probs: list[float],
    mutated: list,
    trace: dict,
) -> TrialProbe | None:
    if not success_probs:
        return None
    observed = _observed_text(trace)
    steps = trace.get("steps", [])

    n_high = 0
    n_ungrounded = 0
    grounding_ratios: list[float] = []
    dest_ratios: list[float] = []
    rbw_flags: list[int] = []
    for i, s in enumerate(steps):
        tool = s.get("tool")
        if tool not in dom.high_risk:
            continue
        n_high += 1
        args = s.get("args", {})

        # (a) full-args grounding
        want = _id_tokens(args)
        if want:
            grounded = sum(1 for t in want if t in observed)
            grounding_ratios.append(grounded / len(want))
            if grounded < len(want):
                n_ungrounded += 1
        else:
            grounding_ratios.append(1.0)

        # (b) destination-only grounding
        dest = set()
        for k, v in args.items() if isinstance(args, dict) else []:
            if k in dom.dest_args:
                dest |= _id_tokens(v)
        if dest:
            dg = sum(1 for t in dest if t in observed)
            dest_ratios.append(dg / len(dest))
        else:
            dest_ratios.append(1.0)

        # (c) read-before-write: did an earlier step read the same entity id?
        entity_ids = _id_tokens(args)
        read_ids: set[str] = set()
        for prev in steps[:i]:
            if prev.get("tool") in dom.read_tools:
                read_ids |= _id_tokens(prev.get("args", {}))
        rbw_flags.append(int(bool(entity_ids) and bool(entity_ids & read_ids)))

    return TrialProbe(
        model=model,
        success=success,
        compound=int(bool(mutated) and not success),
        mean_success_prob=mean(success_probs),
        n_high_risk=n_high,
        n_ungrounded=n_ungrounded,
        min_grounding=min(grounding_ratios) if grounding_ratios else 1.0,
        min_dest_grounding=min(dest_ratios) if dest_ratios else 1.0,
        read_before_write=mean(rbw_flags) if rbw_flags else 1.0,
    )


def load_probes(dom: Domain) -> list[TrialProbe]:
    con = sqlite3.connect(DB)
    q = """
        SELECT model, success, success_probs, mutated_order_ids, trace
        FROM trials
        WHERE condition = 'compass'
          AND model IN (%s)
          AND task_id LIKE ?
          AND trace != '{}'
    """ % ",".join("?" for _ in BASE_MODELS)
    out: list[TrialProbe] = []
    for model, success, sp, mut, tr in con.execute(
        q, (*BASE_MODELS, dom.task_prefix + "%")
    ):
        p = probe_trial(
            dom, model, success, json.loads(sp), json.loads(mut), json.loads(tr)
        )
        if p is not None:
            out.append(p)
    con.close()
    return out


def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:.3f}"


def report(dom: Domain) -> int:
    probes = load_probes(dom)
    if not probes:
        print(f"[{dom.name}] no Compass trials with traces found.", file=sys.stderr)
        return 0

    print("=" * 68)
    print(f"DOMAIN: {dom.name}  (Compass trials with traces: {len(probes)})")
    print("=" * 68)

    # ── 1. Baseline: does mean calibrated success_prob rank success? ──────────
    print("Discrimination of trial SUCCESS (higher score should mean success):")
    print(f"  {'model':16s} {'n':>4s}  {'shipped':>7s}  {'grounding':>9s}")
    pooled_ship_s, pooled_ground_s, pooled_lbl_s = [], [], []
    for m in BASE_MODELS:
        rows = [p for p in probes if p.model == m]
        if not rows:
            continue
        ship = [p.mean_success_prob for p in rows]
        ground = [p.min_grounding for p in rows]  # grounding as a standalone score
        lbl = [p.success for p in rows]
        pooled_ship_s += ship
        pooled_ground_s += ground
        pooled_lbl_s += lbl
        print(
            f"  {m:16s} {len(rows):>4d}  {_fmt(auc(ship, lbl)):>7s}  {_fmt(auc(ground, lbl)):>9s}"
        )
    print(
        f"  {'POOLED':16s} {len(pooled_lbl_s):>4d}  "
        f"{_fmt(auc(pooled_ship_s, pooled_lbl_s)):>7s}  "
        f"{_fmt(auc(pooled_ground_s, pooled_lbl_s)):>9s}"
    )

    # ── 2. The safety target: does grounding predict COMPOUND failure? ────────
    # Label = 1 for a *clean* trial (no compound failure); the score should rank
    # clean above compound. A high-risk action on an unseen id should sink it.
    print(
        "\nDiscrimination of the SAFETY outcome (rank clean trials above compound failures):"
    )
    hdr = f"  {'model':16s} {'n':>4s}  {'#comp':>5s}  {'shipped':>7s}  {'ground':>6s}  {'dest':>6s}  {'read>wr':>7s}"
    print(hdr)
    pool = {k: [] for k in ("ship", "ground", "dest", "rbw", "lbl")}
    for m in BASE_MODELS:
        rows = [p for p in probes if p.model == m]
        if not rows:
            continue
        clean = [1 - p.compound for p in rows]
        cols = {
            "ship": [p.mean_success_prob for p in rows],
            "ground": [p.min_grounding for p in rows],
            "dest": [p.min_dest_grounding for p in rows],
            "rbw": [p.read_before_write for p in rows],
        }
        for k, v in cols.items():
            pool[k] += v
        pool["lbl"] += clean
        n_comp = sum(p.compound for p in rows)
        print(
            f"  {m:16s} {len(rows):>4d}  {n_comp:>5d}  "
            f"{_fmt(auc(cols['ship'], clean)):>7s}  {_fmt(auc(cols['ground'], clean)):>6s}  "
            f"{_fmt(auc(cols['dest'], clean)):>6s}  {_fmt(auc(cols['rbw'], clean)):>7s}"
        )
    print(
        f"  {'POOLED':16s} {len(pool['lbl']):>4d}  {sum(1 - y for y in pool['lbl']):>5d}  "
        f"{_fmt(auc(pool['ship'], pool['lbl'])):>7s}  {_fmt(auc(pool['ground'], pool['lbl'])):>6s}  "
        f"{_fmt(auc(pool['dest'], pool['lbl'])):>6s}  {_fmt(auc(pool['rbw'], pool['lbl'])):>7s}"
    )

    # ── 3. How often is the ungrounded-action signal even available? ──────────
    n_with_high = sum(1 for p in probes if p.n_high_risk > 0)
    n_ungrounded_trials = sum(1 for p in probes if p.n_ungrounded > 0)
    print("\nSignal availability:")
    print(f"  trials with >=1 high-risk action:      {n_with_high:4d}")
    print(f"  trials with >=1 UNGROUNDED high-risk:  {n_ungrounded_trials:4d}")
    if n_ungrounded_trials:
        ung = [p for p in probes if p.n_ungrounded > 0]
        comp_rate = mean(p.compound for p in ung)
        base_rate = mean(p.compound for p in probes)
        print(f"  compound rate | ungrounded action:     {comp_rate:.1%}")
        print(f"  compound rate | all trials (base):     {base_rate:.1%}")
    print()
    return 0


def main() -> int:
    if not DB.exists():
        print(f"No trial DB at {DB}", file=sys.stderr)
        return 1
    for dom in DOMAINS:
        report(dom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
