"""Render the writeup figures (README + LinkedIn) as PNGs in analysis/figures/.

Reads results/trials.db; renders one figure set spanning every model present
(re-run after new model data lands and the charts refresh).

Run: uv run python analysis/make_figures.py
"""
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from pilot_115_analysis import categorize, load_tasks, t_high_sensitivity_points

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


def models_in_db() -> list[str]:
    db = sqlite3.connect(ROOT / "results" / "trials.db")
    return [r[0] for r in db.execute(
        "SELECT DISTINCT model FROM trials WHERE task_id LIKE 'tau_retail%' ORDER BY model"
    )]


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


def fig_headline(models: list[str]) -> None:
    """Grouped horizontal bars: vanilla (gray baseline) vs compass (accent)."""
    metric_labels = [("success", "Task success"),
                     ("compound", "Compound failure\n(destructive action while wrong)"),
                     ("abstained", "Abstention")]
    fig, axes = plt.subplots(
        1, len(models), figsize=(5.4 * len(models) + 1, 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        m = metrics_for(model)
        ys = range(len(metric_labels))
        h = 0.32
        van = ax.barh([y + h / 2 + 0.02 for y in ys],
                      [m["vanilla"][k] for k, _ in metric_labels],
                      height=h, color=GRAY, label="Vanilla ReAct")
        com = ax.barh([y - h / 2 - 0.02 for y in ys],
                      [m["compass"][k] for k, _ in metric_labels],
                      height=h, color=BLUE, label="Compass")
        _bar_label(ax, van, INK_2)
        _bar_label(ax, com, INK)
        ax.set_yticks(list(ys), [lbl for _, lbl in metric_labels], fontsize=10, color=INK)
        ax.set_xlim(0, 1.02)
        ax.invert_yaxis()
        ax.set_title(model, fontsize=11, color=INK_2, loc="left")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0][0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Compass trades task success for far fewer destructive failures",
                 fontsize=13, fontweight="bold", x=0.01, y=0.985, ha="left", va="top")
    fig.text(0.01, 0.885, "115 τ-bench retail tasks (single-shot) × 2 conditions per model",
             fontsize=9, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(FIG_DIR / "headline_metrics.png", dpi=200)
    plt.close(fig)


CAT_STYLE = {
    # category → (display label, story role)
    "both_fail_v_mutates":    ("Both fail — vanilla mutates DB, Compass doesn't", "compass"),
    "vanilla_only_abstained": ("Vanilla succeeds — Compass abstained", "vanilla"),
    "vanilla_only_committed": ("Vanilla succeeds — Compass acted and failed", "vanilla"),
    "both_fail_both_mutate":  ("Both fail — both mutate the DB", "neutral"),
    "both_fail_clean":        ("Both fail — no destructive action", "neutral"),
    "compass_only":           ("Compass succeeds — vanilla fails", "compass"),
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
    ax.set_title(f"Where each agent wins — all 115 tasks ({model})",
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
    ax.set_xlabel("T_HIGH — confidence threshold for high-risk actions")
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


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    models = models_in_db()
    fig_headline(models)
    fig_categories(models[0])
    fig_threshold(models[0])
    print(f"figures written to {FIG_DIR.relative_to(ROOT)}: "
          "headline_metrics.png, outcome_categories.png, threshold_sensitivity.png")


if __name__ == "__main__":
    main()
