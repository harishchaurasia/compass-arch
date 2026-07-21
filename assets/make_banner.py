"""Generate the README hero banner. Run: uv run python assets/make_banner.py

Kept as a script so the banner is reproducible and tweakable rather than a
binary blob nobody can regenerate.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

BG = "#0B0D10"
FG = "#FFFFFF"
ACCENT = "#2F81F7"
MUTED = "#8B949E"
OUT = Path(__file__).parent / "banner.png"


def main() -> None:
    fig = plt.figure(figsize=(7.5, 3), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(BG)
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy, r = 21.0, 15.0, 8.4

    # pixel ring: square dots around the bezel, cardinals brighter + larger
    for i in range(32):
        a = 2 * math.pi * i / 32
        cardinal = i % 8 == 0
        s = 1.15 if cardinal else 0.62
        ax.add_patch(Rectangle(
            (cx + r * math.cos(a) - s / 2, cy + r * math.sin(a) - s / 2),
            s, s, facecolor=FG if cardinal else MUTED,
            alpha=1.0 if cardinal else 0.55, linewidth=0,
        ))

    # needle: lit half points NE (the "bearing"), dim half trails behind
    def pt(deg: float, rad: float) -> tuple[float, float]:
        a = math.radians(deg)
        return cx + rad * math.cos(a), cy + rad * math.sin(a)

    ax.add_patch(Polygon([pt(48, 5.9), pt(138, 1.9), pt(318, 1.9)],
                         closed=True, facecolor=ACCENT, linewidth=0))
    ax.add_patch(Polygon([pt(228, 5.9), pt(138, 1.9), pt(318, 1.9)],
                         closed=True, facecolor=MUTED, alpha=0.75, linewidth=0))
    ax.add_patch(Circle((cx, cy), 0.85, facecolor=BG, edgecolor=FG, linewidth=0.9))

    ax.text(36, 16.6, "Compass", color=FG, fontsize=40, fontweight="bold",
            va="center", ha="left", family="DejaVu Sans")
    ax.text(36.7, 9.0, "know when you don't know", color=MUTED, fontsize=12.5,
            va="center", ha="left", family="DejaVu Sans")

    fig.savefig(OUT, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
