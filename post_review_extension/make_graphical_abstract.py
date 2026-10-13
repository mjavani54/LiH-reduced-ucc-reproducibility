#!/usr/bin/env python3
"""Create the revised graphical abstract with exact, editable text."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
BLUE = "#4C7A90"
BLUE_FILL = "#E9F1F5"
GREEN = "#2F6B45"
GREEN_FILL = "#EAF4EC"
AMBER = "#A56516"
AMBER_FILL = "#FFF4E3"
GRAY = "#5F6368"


def box(ax, x, y, width, height, edge, fill, title, detail) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=2.6,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", fontsize=15.8, weight="bold", color=NAVY)
    ax.text(x + width / 2, y + height * 0.29, detail, ha="center", va="center", fontsize=16, color=GRAY)


def arrow(ax, start, end, label) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=22, linewidth=2.2, color=GRAY))
    ax.text((start[0] + end[0]) / 2 + 0.025, (start[1] + end[1]) / 2, label, va="center", fontsize=12.5, color=GRAY)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=250)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Circle((0.40, 0.93), 0.065, facecolor="#AEB4BA", edgecolor=GRAY, linewidth=2.5))
    ax.add_patch(Circle((0.61, 0.93), 0.042, facecolor="#F1F3F5", edgecolor=GRAY, linewidth=2.5))
    ax.plot([0.465, 0.568], [0.93, 0.93], color=GRAY, linewidth=6, solid_capstyle="round")
    ax.text(0.40, 0.93, "Li", ha="center", va="center", color="white", fontsize=25, weight="bold")
    ax.text(0.61, 0.93, "H", ha="center", va="center", color="#2F3337", fontsize=21, weight="bold")

    box(ax, 0.14, 0.72, 0.72, 0.125, BLUE, BLUE_FILL, "Compact-space discovery", "Reduced-10 of 24 operators")
    arrow(ax, (0.50, 0.70), (0.50, 0.635), "orbital-aware transfer")
    box(ax, 0.14, 0.49, 0.72, 0.125, GREEN, GREEN_FILL, "Symmetry-aware block selection", "Partner-47 of 99 operators")

    ax.text(0.50, 0.405, "Strict numerical energy and S² agreement", ha="center", va="center", fontsize=18, weight="bold", color=GREEN)
    ax.text(0.50, 0.365, "at 3 selection + 4 held-out LiH geometries", ha="center", va="center", fontsize=12.5, color=GRAY)
    ax.text(0.50, 0.312, "≈53% fewer compiled CNOTs and depth than Full-99", ha="center", va="center", fontsize=13.2, weight="bold", color=NAVY)

    ax.add_patch(FancyArrowPatch((0.50, 0.285), (0.50, 0.225), arrowstyle="-|>", mutation_scale=22, linewidth=2.2, color=GRAY))
    boundary = FancyBboxPatch(
        (0.14, 0.075),
        0.72,
        0.125,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=2.4,
        edgecolor=AMBER,
        facecolor=AMBER_FILL,
    )
    ax.add_patch(boundary)
    ax.text(0.50, 0.155, "Explicit transfer boundary", ha="center", va="center", fontsize=18, weight="bold", color=AMBER)
    ax.text(0.50, 0.112, "Linear H₄ exceeds chemical accuracy when stretched", ha="center", va="center", fontsize=13.2, color=GRAY)

    fig.savefig(OUTPUT / "graphical_abstract_revised.png", bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(OUTPUT / "graphical_abstract_revised.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

