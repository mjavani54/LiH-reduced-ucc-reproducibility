from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RESULTS = ROOT / "results"
PHASE7D = RESULTS / "phase7d"

BLUE = "#264653"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
ORANGE = "#F4A261"
RED = "#E76F51"
GRAY = "#6B7280"
INK = "#1F2937"


def finish(fig: plt.Figure, name: str) -> None:
    # Preserve the declared physical dimensions: these figures are designed at
    # the manuscript's final 3.02-inch column width.
    png_path = OUT / f"{name}.png"
    pdf_path = OUT / f"{name}.pdf"
    png_temporary = OUT / f".{name}.png.tmp"
    pdf_temporary = OUT / f".{name}.pdf.tmp"
    try:
        fig.savefig(
            png_temporary,
            format="png",
            dpi=450,
            facecolor="white",
        )
        fig.savefig(
            pdf_temporary,
            format="pdf",
            facecolor="white",
        )
        if png_temporary.stat().st_size == 0 or pdf_temporary.stat().st_size == 0:
            raise RuntimeError(f"Empty figure export for {name}")
        png_temporary.replace(png_path)
        pdf_temporary.replace(pdf_path)
    finally:
        plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
        }
    )


def figure1_ablation() -> None:
    conclusions = json.loads(
        (RESULTS / "phase6" / "phase6_final_conclusions.json").read_text()
    )
    penalties = conclusions["consolidated_leave_one_out_results"][
        "maximum_best_energy_penalty_by_operator_ha"
    ]
    order = ["D15", "D0", "D12", "D3", "S0", "D10", "D5", "S4", "S3", "S7"]
    vals = np.array([penalties[k] for k in order])
    colors = [RED if v >= 1.6e-3 else TEAL for v in vals]

    # Build at the final one-column print width so typography is not scaled down later.
    fig, ax = plt.subplots(figsize=(3.02, 3.00))
    x = np.arange(len(order))
    ax.bar(x, vals * 1000, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(1.6, color=INK, linestyle="--", linewidth=1.1, label="chemical-accuracy scale (1.6 mHa)")
    ax.set_xticks(x, order, rotation=42, ha="right")
    ax.set_ylabel("Maximum leave-one-out\npenalty (mHa)")
    ax.set_xlabel("Omitted compact-support operator")
    ax.set_title("Every compact-support\noperator is needed for\nstrict equivalence", pad=5)
    ax.grid(axis="y")
    line_handle, = ax.get_legend_handles_labels()[0]
    ax.legend(
        handles=[line_handle, Patch(facecolor=RED, edgecolor="white")],
        labels=["chemical accuracy (1.6 mHa)", "red: fails at ≥1 geometry"],
        frameon=False,
        loc="upper right",
        fontsize=7.1,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.74, bottom=0.24)
    finish(fig, "figure1_leave_one_out_ablation")


def figure2_transfer() -> None:
    a1 = json.loads(
        (RESULTS / "phase7a1" / "phase7a1_embedding_conclusions.json").read_text()
    )
    b = json.loads(
        (RESULTS / "phase7b" / "phase7b_transfer_conclusions.json").read_text()
    )
    geoms = [1.595, 2.5, 3.0]
    overlap = [g["rotated_2e5o_quality"] for g in a1["geometry_results"]]
    compact_error_maximum = 3.530189474076906e-10  # archived all-geometry maximum, not three observations
    trunc_error = [
        b["active_space_truncation_context"][f"{r:.6f}"]["rotated_2e5o_error_vs_full_noncore_ha"]
        for r in geoms
    ]

    # Stack the panels so each uses the full journal-column width.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.02, 5.55), gridspec_kw={"hspace": 0.53})
    ax1.plot(geoms, overlap, marker="o", color=BLUE, linewidth=1.8)
    ax1.axhline(0.75, color=GRAY, linestyle="--", linewidth=1.0, label="frozen reliability threshold")
    ax1.set_ylim(0.70, 1.01)
    ax1.set_xlabel("Li–H distance (Å)")
    ax1.set_ylabel("Minimum group singular value")
    ax1.set_title("(a) Rotated 2e,5o embedding", pad=6)
    ax1.grid(True)
    ax1.legend(frameon=False, loc="lower left", fontsize=7.2, handlelength=1.5)

    ax2.plot(geoms, trunc_error, marker="o", color=RED, linewidth=1.8, label="2e,5o model truncation")
    ax2.axhline(compact_error_maximum, color=TEAL, linewidth=1.8, linestyle=":", label="compact10 maximum error")
    ax2.axhline(1.6e-3, color=INK, linestyle="--", linewidth=1.0, label="chemical accuracy")
    ax2.set_yscale("log")
    ax2.set_xlabel("Li–H distance (Å)")
    ax2.set_ylabel("Absolute energy error (Ha)")
    ax2.set_title("(b) Support transfer versus\nmodel truncation", pad=6)
    ax2.grid(True, which="both")
    ax2.legend(frameon=False, loc="center left", fontsize=7.0, handlelength=1.5)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.95, bottom=0.09)
    finish(fig, "figure2_cross_basis_transfer")


def figure3_augmentation_path() -> None:
    path = pd.read_csv(
        RESULTS / "phase7c" / "phase7c_deterministic_path_summary.csv"
    )
    pivot = path.pivot(index="total_operator_count", columns="bond_length_angstrom", values="best_error_vs_full_2e10o_exact_ha")
    worst = pivot.max(axis=1)
    fig, ax = plt.subplots(figsize=(3.02, 3.55))
    ax.plot(worst.index, worst.values, color=BLUE, linewidth=1.8, marker="o", markersize=3.2, label="worst of three geometries")
    ax.axhline(1.6e-3, color=INK, linestyle="--", linewidth=1.0, label="chemical accuracy")
    ax.axhline(8e-4, color=GRAY, linestyle=":", linewidth=1.1, label="guard band")
    ax.axhline(1e-6, color=TEAL, linestyle="--", linewidth=1.0, label="strict equivalence")
    for n, label, color in [(46, "46: first chemical", ORANGE), (47, "47: spin-complete strict", RED)]:
        y = float(worst.loc[n])
        ax.scatter([n], [y], s=58, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(
            label,
            xy=(n, y),
            xytext=(-72 if n == 46 else -18, 15 if n == 46 else 45),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": color},
            color=color,
            fontsize=7.2,
            ha="left" if n == 46 else "right",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Total UCCSD operators\nin augmented support")
    ax.set_ylabel("Worst-geometry error (Ha)")
    ax.set_title("Deterministic expanded-space\naugmentation path", pad=7)
    ax.grid(True, which="both")
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "worst geometry"
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        fontsize=6.9,
        handlelength=1.5,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.82, bottom=0.33)
    finish(fig, "figure3_expanded_space_augmentation")


def figure4_confirmation() -> None:
    df = pd.read_csv(PHASE7D / "phase7d_variant_geometry_summary.csv")
    order = ["compact10", "chemical_boundary46", "spin_complete_candidate47", "full99"]
    labels = ["compact10", "boundary46", "candidate47", "full99"]
    colors = [BLUE, ORANGE, TEAL, GRAY]
    markers = ["o", "^", "s", "D"]

    # Stack the panels so the title, axes, ticks, and curves remain legible at print size.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.02, 4.65), gridspec_kw={"hspace": 0.40})
    for v, lab, c, m in zip(order, labels, colors, markers):
        sub = df[df["variant"] == v].sort_values("bond_length_angstrom")
        ax1.plot(sub["bond_length_angstrom"], sub["best_error_vs_full_exact_ha"], color=c, marker=m, linewidth=1.7, label=lab)
        ax2.plot(sub["bond_length_angstrom"], sub["best_spin_square"], color=c, marker=m, linewidth=1.7, label=lab)
    ax1.axhline(1.6e-3, color=INK, linestyle="--", linewidth=1.0)
    ax1.axhline(1e-6, color=GRAY, linestyle=":", linewidth=1.0)
    ax1.set_yscale("log")
    ax1.set_xlabel("Li–H distance (Å)")
    ax1.set_ylabel("Error vs exact 2e,10o energy (Ha)")
    ax1.set_title("(a) Frozen Qiskit\nenergy confirmation", pad=6)
    ax1.grid(True, which="both")

    ax2.axhline(1e-6, color=INK, linestyle="--", linewidth=1.0, label="singlet tolerance")
    ax2.set_yscale("log")
    ax2.set_xlabel("Li–H distance (Å)")
    ax2.set_ylabel(r"$\langle S^2\rangle$")
    ax2.set_title("(b) Spin purity", pad=6)
    ax2.grid(True, which="both")
    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labs,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=7.3,
        handlelength=1.6,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.24, right=0.97, top=0.88, bottom=0.15)
    finish(fig, "figure4_frozen_confirmation")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    figure1_ablation()
    figure2_transfer()
    figure3_augmentation_path()
    figure4_confirmation()


if __name__ == "__main__":
    main()
