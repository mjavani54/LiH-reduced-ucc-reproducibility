#!/usr/bin/env python3
"""Create publication figures for the post-review benchmark extension."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "reduced": "#0072B2",
    "boundary": "#D55E00",
    "partner": "#009E73",
    "full": "#6F6F6F",
    "accent": "#CC79A7",
    "gold": "#E69F00",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_lih_transfer() -> None:
    data = pd.read_csv(RESULTS / "main_and_holdout_support_results.csv")
    mapping = {
        "compact10": ("Reduced-10", COLORS["reduced"], "o"),
        "chemical_boundary46": ("Boundary-46", COLORS["boundary"], "s"),
        "spin_complete_candidate47": ("Partner-47", COLORS["partner"], "D"),
        "full99": ("Full-99", COLORS["full"], "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), sharex=True)
    for key, (label, color, marker) in mapping.items():
        subset = data[data["variant"] == key].sort_values("bond_length_angstrom")
        axes[0].plot(
            subset["bond_length_angstrom"],
            np.maximum(subset["absolute_error_ha"], 1e-16),
            label=label,
            color=color,
            marker=marker,
            lw=1.5,
            ms=4.2,
        )
        axes[1].plot(
            subset["bond_length_angstrom"],
            np.maximum(subset["s2"], 1e-16),
            color=color,
            marker=marker,
            lw=1.5,
            ms=4.2,
        )
        held = subset[subset["geometry_role"] == "held_out"]
        for axis, column in zip(axes, ("absolute_error_ha", "s2")):
            axis.scatter(
                held["bond_length_angstrom"],
                np.maximum(held[column], 1e-16),
                marker=marker,
                s=36,
                facecolors="white",
                edgecolors=color,
                linewidths=1.1,
                zorder=5,
            )
    axes[0].axhline(1.6e-3, color="#555555", ls="--", lw=0.9, label="1.6 mHa")
    axes[0].axhline(1e-6, color="#999999", ls=":", lw=0.9, label="1 μHa")
    axes[1].axhline(1e-3, color="#999999", ls=":", lw=0.9)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Absolute energy error (Ha)")
    axes[1].set_ylabel(r"$\langle S^2\rangle$")
    for axis in axes:
        axis.set_xlabel("Li–H distance (Å)")
        axis.grid(axis="y", which="both", alpha=0.2)
        axis.set_xticks([1.4, 1.595, 1.8, 2.1, 2.5, 2.75, 3.0])
        axis.tick_params(axis="x", rotation=35)
    axes[0].set_title("a  Frozen-support energy error")
    axes[1].set_title("b  Optimized-state spin diagnostic")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.text(0.5, -0.02, "Open symbols denote geometries not used in expanded-space selection.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, "figure1_lih_heldout_transfer")


def figure_controls() -> None:
    exhaustive = pd.read_csv(RESULTS / "boundary46_plus_one_exhaustive_summary.csv")
    random = pd.read_csv(RESULTS / "partner_closed_random_control_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8))

    colors = np.where(exhaustive["added_is_missing_spin_partner"], COLORS["partner"], "#8A8A8A")
    axes[0].scatter(
        np.arange(1, len(exhaustive) + 1),
        np.maximum(exhaustive["worst_absolute_error_ha"], 1e-16),
        c=colors,
        s=np.where(exhaustive["added_is_missing_spin_partner"], 40, 18),
        edgecolors="none",
    )
    partner_row = exhaustive[exhaustive["added_is_missing_spin_partner"]].iloc[0]
    partner_x = int(exhaustive.index[exhaustive["added_is_missing_spin_partner"]][0]) + 1
    axes[0].annotate(
        "E17 (missing partner)",
        (partner_x, partner_row["worst_absolute_error_ha"]),
        xytext=(partner_x + 16, 2e-11),
        arrowprops={"arrowstyle": "->", "color": COLORS["partner"], "lw": 0.8},
        ha="center",
        color=COLORS["partner"],
        fontsize=8,
    )
    axes[0].axhline(1e-6, color="#777777", ls=":", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("One added operator (53 alternatives)")
    axes[0].set_ylabel("Worst error over three geometries (Ha)")
    axes[0].set_title("a  Boundary-46 + one operator")
    axes[0].grid(axis="y", which="both", alpha=0.2)

    x = np.arange(1, len(random) + 1)
    axes[1].scatter(x, random["worst_absolute_error_ha"], color="#8A8A8A", s=17, alpha=0.8)
    axes[1].axhline(1.6876e-14, color=COLORS["partner"], lw=1.4, label="Ranked Partner-47")
    axes[1].axhline(1.6e-3, color="#555555", ls="--", lw=0.9, label="1.6 mHa")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Matched random support")
    axes[1].set_ylabel("Worst error over three geometries (Ha)")
    axes[1].set_title("b  100 spin-partner-closed controls")
    axes[1].grid(axis="y", which="both", alpha=0.2)
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "figure2_matched_controls")


def figure_adapt_resources() -> None:
    adapt = pd.read_csv(RESULTS / "adapt_vqe_iteration_history.csv")
    resources = pd.read_csv(RESULTS / "matched_circuit_resources.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8))
    geometry_colors = {1.595: COLORS["reduced"], 2.5: COLORS["gold"], 3.0: COLORS["accent"]}
    for spacing, color in geometry_colors.items():
        subset = adapt[np.isclose(adapt["bond_length_angstrom"], spacing)]
        axes[0].plot(
            subset["operator_count"],
            np.maximum(subset["absolute_error_ha"], 1e-16),
            color=color,
            lw=1.4,
            label=f"{spacing:g} Å",
        )
    axes[0].axhline(1.6e-3, color="#555555", ls="--", lw=0.9)
    axes[0].axhline(1e-6, color="#999999", ls=":", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("ADAPT operator count")
    axes[0].set_ylabel("Absolute energy error (Ha)")
    axes[0].set_title("a  Geometry-specific fermionic ADAPT")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", which="both", alpha=0.2)

    ordered = resources.set_index("support").loc[["boundary46", "partner47", "full99"]]
    labels = ["Boundary-46", "Partner-47", "Full-99"]
    x = np.arange(3)
    width = 0.36
    axes[1].bar(x - width / 2, ordered["compiled_cx"], width, color=COLORS["reduced"], label="CNOT count")
    axes[1].bar(x + width / 2, ordered["compiled_depth"], width, color=COLORS["gold"], label="Depth")
    axes[1].set_xticks(x, labels, rotation=18)
    axes[1].set_ylabel("Compiled resource count")
    axes[1].set_title("b  Matched 20-qubit compilation")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "figure3_adapt_and_resources")


def figure_block_sensitivity() -> None:
    blocks = pd.read_csv(RESULTS / "spin_partner_block_path_results.csv")
    ranking = pd.read_csv(RESULTS / "ranking_sensitivity.csv")
    worst = blocks.groupby("operator_count", as_index=False)["absolute_error_ha"].max()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8))
    axes[0].plot(
        worst["operator_count"],
        np.maximum(worst["absolute_error_ha"], 1e-16),
        color=COLORS["partner"],
        marker="o",
        ms=3.5,
        lw=1.4,
    )
    axes[0].axhline(1.6e-3, color="#555555", ls="--", lw=0.9)
    axes[0].axhline(1e-6, color="#999999", ls=":", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Operators after spin-partner-block addition")
    axes[0].set_ylabel("Worst error over three geometries (Ha)")
    axes[0].set_title("a  Spin-partner-block augmentation")
    axes[0].grid(axis="y", which="both", alpha=0.2)
    axes[0].annotate(
        "final block: E8 + E17",
        (47, float(worst.loc[worst["operator_count"] == 47, "absolute_error_ha"].iloc[0])),
        xytext=(42.5, 2e-10),
        arrowprops={"arrowstyle": "->", "color": COLORS["partner"], "lw": 0.8},
        ha="right",
        color=COLORS["partner"],
        fontsize=8,
    )

    floor = 1e-8
    subset = ranking[ranking["denominator_floor_ha"] == floor].sort_values(
        "scf_tolerance", ascending=False
    )
    x = np.arange(len(subset))
    labels = [f"$10^{{{int(round(np.log10(v)))}}}$" for v in subset["scf_tolerance"]]
    axes[1].semilogy(x, subset["e8_score"], marker="o", color=COLORS["reduced"], label="E8")
    axes[1].semilogy(x, subset["e17_score"], marker="s", color=COLORS["accent"], label="E17")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("SCF convergence tolerance")
    axes[1].set_ylabel("Perturbative ranking score")
    axes[1].set_title("b  Tail-score sensitivity")
    axes[1].grid(axis="y", which="both", alpha=0.2)
    axes[1].legend(frameon=False)
    axes[1].text(
        0.03,
        0.06,
        "Top-37 set unchanged\nfor all 9 conditions",
        transform=axes[1].transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    fig.tight_layout()
    save(fig, "figure4_block_ranking_sensitivity")


def figure_h4_boundary() -> None:
    data = pd.read_csv(RESULTS / "h4_transfer_results.csv")
    mapping = {
        "reference_ADAPT_prefix_spin_closed": ("Frozen 10-op support", COLORS["partner"], "o"),
        "full_factorized_UCCSD": ("Full 26-op UCCSD", COLORS["full"], "s"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharex=True)
    for key, (label, color, marker) in mapping.items():
        subset = data[data["support_name"] == key].sort_values("spacing_angstrom")
        axes[0].plot(subset["spacing_angstrom"], subset["absolute_error_ha"], color=color, marker=marker, label=label)
        axes[1].plot(subset["spacing_angstrom"], np.maximum(subset["spin_squared"], 1e-16), color=color, marker=marker)
        held = subset[subset["geometry_role"] == "held_out"]
        axes[0].scatter(held["spacing_angstrom"], held["absolute_error_ha"], facecolors="white", edgecolors=color, marker=marker, s=38, zorder=4)
        axes[1].scatter(held["spacing_angstrom"], np.maximum(held["spin_squared"], 1e-16), facecolors="white", edgecolors=color, marker=marker, s=38, zorder=4)
    axes[0].axhline(1.6e-3, color="#555555", ls="--", lw=0.9)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Absolute FCI error (Ha)")
    axes[1].set_ylabel(r"$\langle S^2\rangle$")
    for axis in axes:
        axis.set_xlabel("H–H spacing in linear H₄ (Å)")
        axis.grid(axis="y", which="both", alpha=0.2)
    axes[0].set_title("a  Transfer boundary in a 4e,4o model")
    axes[1].set_title("b  Optimized-state spin diagnostic")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "figure5_h4_boundary")


def main() -> None:
    style()
    figure_lih_transfer()
    figure_controls()
    figure_adapt_resources()
    figure_block_sensitivity()
    figure_h4_boundary()


if __name__ == "__main__":
    main()
