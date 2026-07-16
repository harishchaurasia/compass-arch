"""Render the writeup figures (README + LinkedIn) as PNGs in analysis/figures/.

Reads results/trials.db; renders one figure set spanning every model present
(re-run after new model data lands and the charts refresh).

Run: uv run python analysis/make_figures.py
"""
import json
import sqlite3
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from pilot_115_analysis import categorize, load_tasks, t_high_sensitivity_points

from eval.metrics import brier_score, ece

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "analysis" / "figures"

# palette roles (dataviz reference palette, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"     # accent / compass / "compass better"
AQUA = "#1baf7a"     # second line series
ORANGE = "#eb6834"   # "vanilla better"
GRAY = "#b5b3ac"     # de-emphasis bar fill (vanilla baseline, neutral rows)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "svg.fonttype": "none",
})


def models_in_db(min_per_condition: int = 100) -> list[str]:
    """Models with a full vanilla+compass tau_retail run (both conditions
    >= min_per_condition trials). Filters out smoke tests and compass-only
    variant rows so the charts never render partial data."""
    db = sqlite3.connect(ROOT / "results" / "trials.db")
    counts: dict[str, dict[str, int]] = {}
    for model, cond, n in db.execute(
        "SELECT model, condition, COUNT(*) FROM trials "
        "WHERE task_id LIKE 'tau_retail%' GROUP BY model, condition"
    ):
        counts.setdefault(model, {})[cond] = n
    return sorted(
        m for m, c in counts.items()
        if c.get("vanilla", 0) >= min_per_condition
        and c.get("compass", 0) >= min_per_condition
    )


def metrics_for(model: str) -> dict:
    tasks = load_tasks(model)
    n = len(tasks)
    out = {}
    for cond in ("vanilla", "compass"):
        rows = [d[cond] for d in tasks.values()]
        out[cond] = {
            "success": sum(r["success"] for r in rows) / n,
            "abstained": sum(r["abstained"] for r in rows) / n,
            "compound": sum(r["wrong_mutation"] for r in rows) / n,
        }
    return out


def _bar_label(ax, bars, color=INK):
    for b in bars:
        ax.annotate(f"{b.get_width():.0%}" if b.get_width() < 1 else "100%",
                    (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color=color)


def cond_metrics(model: str, cond: str) -> dict | None:
    """success / abstention / wrong-mutation rates for one model+condition,
    read straight from the tau_retail rows (works for compass-only variants).
    Returns None when the model+condition has no rows yet, so a not-yet-run
    variant simply drops out of the chart instead of crashing it."""
    db = sqlite3.connect(ROOT / "results" / "trials.db")
    rows = db.execute(
        "SELECT success, abstained, mutated_order_ids FROM trials "
        "WHERE task_id LIKE 'tau_retail%' AND model = ? AND condition = ?",
        (model, cond),
    ).fetchall()
    n = len(rows)
    if n == 0:
        return None
    return {
        "success": sum(bool(s) for s, _, _ in rows) / n,
        "abstained": sum(bool(a) for _, a, _ in rows) / n,
        "compound": sum(bool(json.loads(m)) and not s for s, _, m in rows) / n,
    }


# A calibration variant to overlay on its base model's panel (model -> variant).
# Every overconfident local model gets a shrinkage run; gpt-4o-mini doesn't need one.
SHRINK_OF = {
    "qwen2.5:14b": "qwen2.5:14b-shrink",
    "qwen2.5:7b": "qwen2.5:7b-shrink",
    "llama3.1:8b": "llama3.1:8b-shrink",
}

# fixed series styling; the shrinkage bar only appears where SHRINK_OF has data
SERIES = [
    ("Vanilla ReAct", GRAY, lambda m: cond_metrics(m, "vanilla")),
    ("Compass", BLUE, lambda m: cond_metrics(m, "compass")),
    ("Compass + shrinkage", AQUA,
     lambda m: cond_metrics(SHRINK_OF[m], "compass") if m in SHRINK_OF else None),
]


def fig_headline(models: list[str]) -> None:
    """Grouped horizontal bars per model, laid out on a grid. Vanilla vs Compass
    everywhere; the base-rate-prior (shrinkage) variant is overlaid only where it
    was run."""
    metric_labels = [("success", "Task success"),
                     ("compound", "Compound failure\n(destructive action while wrong)"),
                     ("abstained", "Abstention")]
    panels = {m: [(lbl, col, fn(m)) for lbl, col, fn in SERIES if fn(m) is not None]
              for m in models}
    ncols = 1 if len(models) == 1 else 2
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.6 * ncols, 3.0 * nrows + 1.6), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, model in zip(flat, models):
        series = panels[model]
        nser = len(series)
        ys = list(range(len(metric_labels)))
        group_h = 0.8
        h = group_h / nser
        for si, (_lbl, color, mets) in enumerate(series):
            offset = (si - (nser - 1) / 2) * h
            bars = ax.barh([y + offset for y in ys],
                           [mets[k] for k, _ in metric_labels],
                           height=h * 0.86, color=color)
            _bar_label(ax, bars, INK)
        ax.set_yticks(ys, [lbl for _, lbl in metric_labels], fontsize=9.5, color=INK)
        ax.set_xlim(0, 1.06)
        ax.invert_yaxis()
        ax.set_title(model, fontsize=12, color=INK_2, loc="left")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    for ax in flat[len(models):]:  # hide any unused grid cell
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in SERIES]
    fig.legend(handles, [lbl for lbl, _, _ in SERIES], loc="lower center",
               ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Compass trades some task success for far fewer destructive failures",
                 fontsize=14, fontweight="bold", x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.955,
             "115 τ-bench retail tasks (single-shot), per model. Where a model's confidence "
             "is flat and overconfident, baseline Compass can't gate the first risky action; "
             "the base-rate-prior (shrinkage) variant recovers the gate.",
             fontsize=8.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(FIG_DIR / "headline_metrics.png", dpi=200)
    plt.close(fig)


CAT_STYLE = {
    # category → (display label, story role)
    "both_fail_v_mutates":    ("Both fail - vanilla mutates DB, Compass doesn't", "compass"),
    "vanilla_only_abstained": ("Vanilla succeeds - Compass abstained", "vanilla"),
    "vanilla_only_committed": ("Vanilla succeeds - Compass acted and failed", "vanilla"),
    "both_fail_both_mutate":  ("Both fail - both mutate the DB", "neutral"),
    "both_fail_clean":        ("Both fail - no destructive action", "neutral"),
    "compass_only":           ("Compass succeeds - vanilla fails", "compass"),
    "both_succeed":           ("Both succeed", "neutral"),
}
ROLE_COLOR = {"compass": BLUE, "vanilla": ORANGE, "neutral": GRAY}


def fig_categories(model: str) -> None:
    tasks = load_tasks(model)
    counts: dict[str, int] = {}
    for d in tasks.values():
        cat = categorize(d["vanilla"], d["compass"])
        counts[cat] = counts.get(cat, 0) + 1
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    labels = [CAT_STYLE[c][0] for c, _ in items]
    colors = [ROLE_COLOR[CAT_STYLE[c][1]] for c, _ in items]
    bars = ax.barh(labels, [n for _, n in items], color=colors, height=0.62)
    for b, (_, n) in zip(bars, items):
        ax.annotate(str(n), (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(n for _, n in items) * 1.08)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=10, labelcolor=INK, length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[r])
               for r in ("compass", "vanilla", "neutral")]
    ax.legend(handles, ["Compass better", "Vanilla better", "Neither"],
              loc="lower right", frameon=False, fontsize=9)
    ax.set_title(f"Where each agent wins - all 115 tasks ({model})",
                 fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "outcome_categories.png", dpi=200)
    plt.close(fig)


def fig_threshold(model: str) -> None:
    pts = t_high_sensitivity_points(load_tasks(model))
    ts = [p["t_high"] for p in pts]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(ts, [p["abstain_rate"] for p in pts], color=BLUE, linewidth=2,
            marker="o", markersize=5)
    ax.plot(ts, [p["pre_mutation_rate"] for p in pts], color=AQUA, linewidth=2,
            marker="o", markersize=5)
    ax.annotate("Abstention rate\n(all tasks)", (ts[-1], pts[-1]["abstain_rate"]),
                xytext=(10, -6), textcoords="offset points", fontsize=10,
                color=INK, ha="left", va="top")
    ax.annotate("Abstains BEFORE the first\ndestructive call (traced trials)",
                (ts[-1], pts[-1]["pre_mutation_rate"]),
                xytext=(10, 6), textcoords="offset points", fontsize=10,
                color=INK, ha="left", va="bottom")
    ax.axvline(0.8, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("shipped\nT_HIGH", (0.8, 0.97), fontsize=9, color=MUTED,
                ha="center", va="top")
    ax.set_xlabel("T_HIGH - confidence threshold for high-risk actions")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(min(ts) - 0.01, max(ts) + 0.14)
    ax.set_xticks([t for t in ts if t not in (0.75, 0.85)])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Calibration fires too late: abstention rarely precedes the damage",
                 fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.text(0.005, 0.008,
             "Offline replay of observed trajectories; pre-mutation curve uses the "
             "traced subset. Trajectories would diverge under other thresholds.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / "threshold_sensitivity.png", dpi=200)
    plt.close(fig)


# Base models to score calibration on (friendly label -> model id). The shrink
# variant, where present (SHRINK_OF), is drawn as a third bar on the same row.
CALIB_MODELS = [
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("Qwen2.5 7B", "qwen2.5:7b"),
    ("Qwen2.5 14B", "qwen2.5:14b"),
    ("Llama 3.1 8B", "llama3.1:8b"),
]


def _calib_scores(model: str, col: str) -> tuple[float, float] | None:
    """(ECE, Brier) for one model's compass rows, reducing `col` to one mean
    confidence per trial paired with that run's own binary outcome."""
    db = sqlite3.connect(ROOT / "results" / "trials.db")
    recs = db.execute(
        f"SELECT {col}, success FROM trials "
        "WHERE task_id LIKE 'tau_retail%' AND model = ? AND condition = 'compass'",
        (model,),
    ).fetchall()
    conf, outs = [], []
    for c, succ in recs:
        c = json.loads(c)
        if not c:
            continue
        conf.append(mean(c))
        outs.append(int(succ))
    if not outs:
        return None
    return ece(conf, outs), brier_score(conf, outs)


def calibration_rows() -> list[dict]:
    """Per base model: raw verbalized confidence, Compass's calibrated
    success_prob, and (where a shrinkage run exists) the shrinkage-calibrated
    success_prob. Skips models absent from the DB."""
    rows = []
    for label, model in CALIB_MODELS:
        verb = _calib_scores(model, "confidence_scores")
        calib = _calib_scores(model, "success_probs")
        if verb is None or calib is None:
            continue
        entry = {"label": label,
                 "verbalized": {"ece": verb[0], "brier": verb[1]},
                 "calibrated": {"ece": calib[0], "brier": calib[1]}}
        shrink_model = SHRINK_OF.get(model)
        shrink = _calib_scores(shrink_model, "success_probs") if shrink_model else None
        if shrink is not None:
            entry["shrink"] = {"ece": shrink[0], "brier": shrink[1]}
        rows.append(entry)
    return rows


def fig_calibration() -> None:
    """Two panels (ECE | Brier). Per model: raw verbalized confidence vs Compass's
    calibrated success_prob vs the shrinkage variant. Lower is better - the gap is
    the honesty Compass adds."""
    data = calibration_rows()
    if not data:
        return
    labels = [d["label"] for d in data]
    metrics = [("ece", "Expected Calibration Error"), ("brier", "Brier score")]
    series = [("Raw verbalized confidence", GRAY, "verbalized"),
              ("Compass calibrated", BLUE, "calibrated"),
              ("Compass + shrinkage", AQUA, "shrink")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ys = list(range(len(labels)))
    for ax, (key, mlabel) in zip(axes, metrics):
        for i, d in enumerate(data):
            present = [s for s in series if s[2] in d]
            nser = len(present)
            h = 0.8 / nser
            for si, (_lbl, color, skey) in enumerate(present):
                offset = (si - (nser - 1) / 2) * h
                b = ax.barh(ys[i] + offset, d[skey][key], height=h * 0.86, color=color)[0]
                ax.annotate(f"{b.get_width():.2f}",
                            (b.get_width(), b.get_y() + b.get_height() / 2),
                            xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK)
        ax.set_yticks(ys, labels, fontsize=10, color=INK)
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()
        ax.set_title(mlabel, fontsize=11, color=INK_2, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[1].set_yticklabels([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in series]
    fig.legend(handles, [s[0] for s in series], loc="lower center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Compass makes an agent's confidence more honest (lower is better)",
                 fontsize=14, fontweight="bold", x=0.01, y=0.98, ha="left", va="top")
    fig.text(0.01, 0.905,
             "Raw models are wildly overconfident: they report ~0.9-1.0 while succeeding "
             "<15% of the time. One mean confidence per trial vs the trial outcome, "
             "115 τ-bench retail tasks.",
             fontsize=9, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(FIG_DIR / "calibration.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    models = models_in_db()
    fig_headline(models)
    fig_categories(models[0])
    fig_threshold(models[0])
    fig_calibration()
    print(f"figures written to {FIG_DIR.relative_to(ROOT)}: "
          "headline_metrics.png, outcome_categories.png, threshold_sensitivity.png, "
          "calibration.png")


if __name__ == "__main__":
    main()
