#!/usr/bin/env python3
"""Render main Fig. 2 (endpoint versus route-valid scoring) from display CSVs only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from _nmi_editorial_style import (
    PALETTE,
    apply_editorial_style,
    clean_axis,
    figure_size,
    panel_label,
    save_figure,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "paper" / "source_data" / "fig_program_aware_score_and_robustness"
PAPER_STEM = ROOT / "paper" / "figures" / "fig_program_aware_score_impact"
NATURE_STEM = ROOT / "Nature" / "figs" / "fig_program_aware_score_impact"
CANDIDATE_DIR = ROOT / "paper" / "figures" / "candidates" / "main"

FAMILY_DISPLAY = {
    "wave": "Wave",
    "Burgers": "Burgers",
    "heat": "Heat",
    "Helmholtz_low_k": "Helmholtz",
    "Poisson": "Poisson",
    "linear_elasticity": "Elasticity",
}
FACTOR_DISPLAY = {
    "O": "Operator",
    "G": "Geometry",
    "B": "Boundary",
    "D": "Discretisation",
    "S": "Structure",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_data():
    headline = read_csv("fig2_panel_a.csv")[0]
    families = read_csv("fig2_panel_c_family.csv")
    cells = read_csv("fig2_panel_d_ogb.csv")
    effects = read_csv("fig2_factor_effects.csv")
    prompt_csv = ROOT / "paper" / "source_data" / "fig3_exact_formula_audit" / "prompt_comparison_by_family.csv"
    with prompt_csv.open() as handle:
        prohibition = {row["family"]: int(row["prohibition_endpoint"]) / int(row["prohibition_calls"]) for row in csv.DictReader(handle)}
    for row in families:
        row["prohibition_rate"] = prohibition[row["family"]]
    headline["prohibition_rate"] = prohibition["All"]
    assert int(headline["endpoint_passes"]) == 1162
    assert int(headline["route_valid_passes"]) == 1037
    assert int(headline["confirmed_invalid"]) == 125
    assert sum(int(row["confirmed_invalid"]) for row in families) == 125
    assert sum(int(row["confirmed_invalid"]) for row in cells) == 125
    assert sum(int(row["scheduled"]) for row in families) == int(headline["scheduled"])
    assert {row["estimand"] for row in effects} == {"endpoint", "route_valid", "route_valid_minus_endpoint"}
    return headline, families, cells, effects


def panel_family_dumbbell(ax, headline, families):
    ordered = sorted(families, key=lambda row: float(row["absolute_inflation"]), reverse=True)
    rows = [(FAMILY_DISPLAY[row["family"]], row) for row in ordered] + [("All families", headline)]
    n = len(rows)
    y = np.arange(n)[::-1].astype(float)
    y[-1] -= 0.35
    for yy, (label, row) in zip(y, rows, strict=True):
        endpoint = 100 * float(row["endpoint_pass_rate"])
        route = 100 * float(row["route_valid_pass_rate"])
        gap = 100 * float(row["absolute_difference" if "absolute_difference" in row else "absolute_inflation"])
        ax.plot([route, endpoint], [yy, yy], color=PALETTE["claim_invalid"], lw=1.3, solid_capstyle="butt", zorder=1)
        ax.scatter([route], [yy], s=20, color=PALETTE["route_valid"], zorder=3, edgecolor="white", linewidth=0.5)
        ax.scatter([endpoint], [yy], s=20, facecolor="white", edgecolor=PALETTE["text"], linewidth=0.8, zorder=3)
        ax.scatter([100 * float(row["prohibition_rate"])], [yy], s=16, marker="D", color=PALETTE["reference_hidden"], edgecolor="white", linewidth=0.5, zorder=4)
        if gap > 0:
            ax.text((route + endpoint) / 2, yy + 0.14, f"{gap:.1f}", ha="center", va="bottom", fontsize=7.0, color=PALETTE["claim_invalid"])
        else:
            ax.text(endpoint, yy + 0.14, f"{gap:.1f}", ha="center", va="bottom", fontsize=7.0, color=PALETTE["claim_invalid"])
    labels = [label for label, row in rows]
    ax.set_yticks(y, labels)
    ax.axhline(y[-1] + 0.68, color=PALETTE["grid"], lw=0.7)
    ax.set_xlim(40, 102)
    ax.set_xticks([40, 60, 80, 100])
    ax.set_ylim(y[-1] - 2.0, y[0] + 0.7)
    ax.set_xlabel("Pass rate among scheduled calls (%)")
    ax.set_title("Pass rates by PDE family", loc="left")
    clean_axis(ax, grid_axis="x")
    handles = [
        Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=PALETTE["text"], markeredgewidth=0.8, ms=4.5, label="Endpoint"),
        Line2D([], [], marker="o", ls="none", color=PALETTE["route_valid"], ms=4.5, label="Route-valid"),
        Line2D([], [], marker="D", ls="none", color=PALETTE["reference_hidden"], ms=3.8, label="Prohibition prompt"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7.0, handletextpad=0.5, labelspacing=0.35, borderaxespad=0.2)
    panel_label(ax, "a", x=-0.22, y=1.07)


def panel_cells(ax, cells, fig):
    lookup = {(int(row["O"]), int(row["G"]), int(row["B"])): row for row in cells}
    columns = [(0, 0), (0, 1), (1, 0), (1, 1)]
    matrix = np.array([[100 * float(lookup[(o, g, b)]["absolute_inflation"]) for g, b in columns] for o in (0, 1)])
    invalid = np.array([[int(lookup[(o, g, b)]["confirmed_invalid"]) for g, b in columns] for o in (0, 1)])
    scheduled = np.array([[int(lookup[(o, g, b)]["scheduled"]) for g, b in columns] for o in (0, 1)])
    cmap = ListedColormap(["#FCF7F5", "#F1DCD4", "#E4B9A9", "#D79278", PALETTE["claim_invalid"]])
    bounds = [0, 5, 10, 15, 20, 25]
    norm = BoundaryNorm(bounds, cmap.N)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for row in range(2):
        for col in range(4):
            value = matrix[row, col]
            dark = value >= 15
            text_color = "white" if dark else PALETTE["text"]
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7.0, color=text_color)
    ax.set_xticks(range(4), ["G0 B0", "G0 B1", "G1 B0", "G1 B1"])
    ax.set_yticks(range(2), ["O0", "O1"])
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, ticks=bounds)
    cbar.ax.tick_params(labelsize=7.0, length=2.0)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Endpoint minus route-valid (pp)", fontsize=7.0)
    ax.set_title("Pass-rate gap by design cell", loc="left")
    ax.set_xlabel("Geometry (G), boundary condition (B)")
    ax.set_ylabel("Operator (O)")
    panel_label(ax, "b", x=-0.16, y=1.12)


def factor_rows(effects, estimand):
    lookup = {row["term"]: row for row in effects if row["estimand"] == estimand}
    return [lookup[term] for term in FACTOR_DISPLAY]


def panel_coefficients(ax, effects):
    endpoint = factor_rows(effects, "endpoint")
    route = factor_rows(effects, "route_valid")
    y = np.arange(len(FACTOR_DISPLAY))[::-1].astype(float)
    ax.axvline(0, color=PALETTE["spine"], lw=0.7, zorder=0)
    for rows, offset, color, face, label in (
        (endpoint, 0.17, PALETTE["text"], "white", "Endpoint"),
        (route, -0.17, PALETTE["route_valid"], PALETTE["route_valid"], "Route-valid"),
    ):
        est = np.array([float(r["estimate"]) for r in rows])
        low = np.array([float(r["ci95_low"]) for r in rows])
        high = np.array([float(r["ci95_high"]) for r in rows])
        ax.errorbar(est, y + offset, xerr=np.vstack([est - low, high - est]), fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, zorder=2)
        ax.scatter(est, y + offset, s=16, facecolor=face, edgecolor=color, linewidth=0.8, zorder=3, label=label)
    ax.set_yticks(y, list(FACTOR_DISPLAY.values()))
    ax.set_xlim(-0.40, 0.30)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2])
    ax.set_xlabel("Coefficient on per-task pass fraction")
    ax.set_title("Factor coefficients", loc="left")
    clean_axis(ax, grid_axis="x")
    ax.legend(loc="lower left", fontsize=7.0, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    panel_label(ax, "c", x=-0.22, y=1.12)


def panel_difference(ax, effects):
    diff = factor_rows(effects, "route_valid_minus_endpoint")
    y = np.arange(len(FACTOR_DISPLAY))[::-1].astype(float)
    est = np.array([float(r["estimate"]) for r in diff])
    low = np.array([float(r["ci95_low"]) for r in diff])
    high = np.array([float(r["ci95_high"]) for r in diff])
    ax.axvline(0, color=PALETTE["spine"], lw=0.7, zorder=0)
    colors = [PALETTE["text"] if h < 0 or l > 0 else "#A3ABB2" for l, h in zip(low, high, strict=True)]
    for yy, e, l, h, color in zip(y, est, low, high, colors, strict=True):
        ax.errorbar([e], [yy], xerr=[[e - l], [h - e]], fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, zorder=2)
        ax.scatter([e], [yy], s=16, color=color, zorder=3)
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlim(-0.15, 0.12)
    ax.set_xticks([-0.1, 0.0, 0.1])
    ax.set_xlabel("Route-valid − endpoint")
    ax.set_title("Paired difference", loc="left")
    clean_axis(ax, grid_axis="x")
    panel_label(ax, "d", x=-0.10, y=1.12)


def build_figure(layout: str = "A"):
    headline, families, cells, effects = load_data()
    fig = plt.figure(figsize=figure_size(88))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[0.95, 1.0], wspace=0.55, hspace=0.62, left=0.085, right=0.955, bottom=0.12, top=0.93)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1:])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    panel_family_dumbbell(ax_a, headline, families)
    panel_cells(ax_b, cells, fig)
    panel_coefficients(ax_c, effects)
    panel_difference(ax_d, effects)
    return fig


def render(*, candidates: bool = True) -> None:
    apply_editorial_style()
    if candidates:
        CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
        fig = build_figure("A")
        fig.savefig(CANDIDATE_DIR / "fig2_candidate_A.png", dpi=110)
        plt.close(fig)
    fig = build_figure("A")
    save_figure(fig, PAPER_STEM, tiff=True)
    fig = build_figure("A")
    save_figure(fig, NATURE_STEM, tiff=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-candidates", action="store_true")
    args = parser.parse_args()
    render(candidates=not args.no_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
