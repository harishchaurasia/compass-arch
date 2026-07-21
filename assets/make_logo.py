"""Generate the README logo tile. Run: uv run python assets/make_logo.py

A rounded app-icon tile on a transparent background, so it sits cleanly on both
the light and dark GitHub themes instead of painting a black bar across the page.
Kept as a script so the mark is reproducible rather than an opaque binary.
"""
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

TILE_TOP = "#1B2430"
TILE_BOT = "#0B0D10"
FG = "#FFFFFF"
ACCENT = "#2F81F7"
MUTED = "#8B949E"
OUT = Path(__file__).parent / "logo.png"


def main() -> None:
    fig = plt.figure(figsize=(2.56, 2.56), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # rounded tile with a soft vertical gradient
    tile = FancyBboxPatch(
        (5, 5), 90, 90,
        boxstyle="round,pad=0,rounding_size=20",
        linewidth=0, facecolor=TILE_BOT, zorder=1,
    )
    ax.add_patch(tile)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(
        grad, extent=(5, 95, 5, 95), origin="lower", aspect="auto", zorder=2,
        cmap=LinearSegmentedColormap.from_list("tile", [TILE_BOT, TILE_TOP]),
    )
    im.set_clip_path(tile)

    cx, cy, r = 50.0, 50.0, 27.0

    # pixel bezel: cardinals brighter and larger
    for i in range(32):
        a = 2 * math.pi * i / 32
        cardinal = i % 8 == 0
        s = 3.7 if cardinal else 2.0
        ax.add_patch(Rectangle(
            (cx + r * math.cos(a) - s / 2, cy + r * math.sin(a) - s / 2),
            s, s, facecolor=FG if cardinal else MUTED,
            alpha=1.0 if cardinal else 0.55, linewidth=0, zorder=3,
        ))

    def pt(deg: float, rad: float) -> tuple[float, float]:
        a = math.radians(deg)
        return cx + rad * math.cos(a), cy + rad * math.sin(a)

    # needle: lit half takes the bearing, dim half trails
    ax.add_patch(Polygon([pt(48, 19.0), pt(138, 6.1), pt(318, 6.1)],
                         closed=True, facecolor=ACCENT, linewidth=0, zorder=4))
    ax.add_patch(Polygon([pt(228, 19.0), pt(138, 6.1), pt(318, 6.1)],
                         closed=True, facecolor=MUTED, alpha=0.8, linewidth=0, zorder=4))
    ax.add_patch(Circle((cx, cy), 2.7, facecolor=TILE_BOT, edgecolor=FG,
                        linewidth=1.4, zorder=5))

    fig.savefig(OUT, transparent=True, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
