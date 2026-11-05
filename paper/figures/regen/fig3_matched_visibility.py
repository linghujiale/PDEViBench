#!/usr/bin/env python3
"""Render the matched reference-visibility main figure from prespecified Source Data."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _nmi_editorial_style import (
    PALETTE,
    apply_editorial_style,
    clean_axis,
    figure_size,
    log_tick_labels,
    panel_label,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "paper" / "source_data" / "fig3_matched_visibility"
PAPER_OUT = ROOT / "paper" / "figures" / "fig3_matched_visibility"
NATURE_OUT = ROOT / "Nature" / "figs" / "fig3_matched_visibility"

SUPPLIED = "#7D6B9E"
RETAINED = PALETTE["reference_hidden"]
INK = PALETTE["text"]
MID = PALETTE["secondary"]
LIGHT = PALETTE["grid"]
PALE_BLUE = "#EEF2F6"
PALE_TEAL = "#EDF5F4"
WARN = PALETTE["claim_invalid"]
PALE_WARN = "#E2E5E8"

ORIGINAL_OPERATOR = "\u2212\u0394u \u2212 k\u00b2u = f"
ALIGNED_OPERATOR = "\u0394u + k\u00b2u = f"


def read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_pair_matrix(ax, pair_rows):
    counts = defaultdict(int)
    for row in pair_rows:
        counts[row["paired_category"]] += 1
    matrix = np.array(
        [
            [counts["both_pass"], counts["hidden_only"]],
            [counts["visible_only"], counts["neither"]],
        ]
    )
    assert matrix.sum() == len(pair_rows)
    colors = [[PALE_TEAL, LIGHT], [PALE_WARN, LIGHT]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor="white", linewidth=2))
            ax.text(j + 0.5, i + 0.5, str(matrix[i, j]), ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5], ["pass", "fail"])
    ax.set_yticks([0.5, 1.5], ["pass", "fail"])
    ax.tick_params(length=0, pad=2)
    ax.xaxis.tick_top()
    ax.set_xlabel("Field supplied", color=SUPPLIED, labelpad=4)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Field retained", color=RETAINED, labelpad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def pretty_route(route: str) -> str:
    return {
        "time_stepping_solver": "Time stepping",
        "sparse_fd_linear_solve": "Sparse solve",
        "nonlinear_or_iterative_solver": "Iterative",
        "coordinate_transform_or_polar_solver": "Polar grid",
        "masked_domain_solver": "Masked domain",
    }[route]


def draw_routes(ax, route_rows, arm_rows):
    arm_programs = {row["arm"]: int(row["solver_programs_deterministic"]) for row in arm_rows}
    route_names = [
        "time_stepping_solver",
        "sparse_fd_linear_solve",
        "nonlinear_or_iterative_solver",
        "coordinate_transform_or_polar_solver",
        "masked_domain_solver",
    ]
    counts = {(row["arm"], row["route_category_deterministic"]): int(row["program_count"]) for row in route_rows}
    y = np.arange(len(route_names))[::-1]
    for arm, offset, color, marker in (("visible", 0.15, SUPPLIED, "o"), ("hidden", -0.15, RETAINED, "s")):
        values = [counts.get((arm, route), 0) for route in route_names]
        assert sum(values) == arm_programs[arm]
        label = "Field supplied" if arm == "visible" else "Field retained"
        ax.scatter(values, y + offset, s=24, color=color, marker=marker, label=label, zorder=3, edgecolor="white", linewidth=0.4)
        for value, yy in zip(values, y + offset, strict=True):
            ax.plot([0, value], [yy, yy], color=color, alpha=0.25, linewidth=1.2, zorder=1)
    ax.set_yticks(y, [pretty_route(route) for route in route_names])
    ax.set_xlim(0, 62)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel(f"Programs per solver class ({arm_programs['visible']} per condition)")
    ax.set_title("Solver classes of submitted programs", loc="left")
    clean_axis(ax, grid_axis="x")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7.0, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)


def draw_alignment(ax, primary_obs, summary):
    discordant = summary["primary"]["discordant_source_cases"]
    assert len(discordant) == 1
    case_id = discordant[0]
    original = {"visible": [0, 0], "hidden": [0, 0]}
    for row in primary_obs:
        if row["source_case_id"] == case_id:
            original[row["arm"]][0] += int(row["endpoint_pass"])
            original[row["arm"]][1] += 1
    diagnostic = summary["corrected_case_diagnostic"]["arm_summary"]
    aligned = {arm: [int(diagnostic[arm]["endpoint_pass"]), int(diagnostic[arm]["scheduled"])] for arm in ("visible", "hidden")}
    groups = [("Original operator\n" + ORIGINAL_OPERATOR, original), ("Aligned operator\n" + ALIGNED_OPERATOR, aligned)]
    for i, (label, data) in enumerate(groups):
        for j, arm in enumerate(("visible", "hidden")):
            passes, total = data[arm]
            face = PALE_TEAL if passes == total else PALE_WARN
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{passes}/{total}", ha="center", va="center", fontsize=8, color=INK)
        name, formula = label.split("\n")
        ax.text(-0.08, i + 0.38, name, ha="right", va="center", fontsize=7.0, color=INK)
        ax.text(-0.08, i + 0.64, formula, ha="right", va="center", fontsize=7.0, color=MID)
    x_lo, x_hi = -1.25, 4.9
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Field\nsupplied", "Field\nretained"], linespacing=1.1)
    for tick, color in zip(ax.get_xticklabels(), (SUPPLIED, RETAINED)):
        tick.set_color(color)
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Helmholtz task with a sign error", loc="left", pad=20)

    tau = case_tau(case_id)
    errors = {"original": {"visible": [], "hidden": []}, "aligned": {arm: [float(v) for v in diagnostic[arm]["rel_l2"]] for arm in ("visible", "hidden")}}
    for row in primary_obs:
        if row["source_case_id"] == case_id:
            errors["original"][row["arm"]].append(float(row["rel_l2"]))
    assert all(len(v) == 2 for g in errors.values() for v in g.values())
    x0 = (2.3 - x_lo) / (x_hi - x_lo)
    sx = ax.inset_axes([x0, 0.0, 1.0 - x0 - 0.02, 1.0])
    for k, key in enumerate(("original", "aligned")):
        sx.axhline(k + 0.5, color=PALETTE["grid"], lw=0.4, zorder=0)
        for arm, dy, color, marker in (("visible", -0.13, SUPPLIED, "o"), ("hidden", 0.13, RETAINED, "s")):
            vals = errors[key][arm]
            sx.scatter(vals, [k + 0.5 + dy] * len(vals), s=22, marker=marker, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    sx.axvline(tau, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    sx.text(tau, -0.02, r"$\tau_{\mathrm{acc}}$", ha="center", va="bottom", fontsize=7.0, color=INK)
    sx.set_xscale("log"); sx.set_xlim(1e-5, 2e1); sx.set_ylim(2, 0)
    sx.set_xticks([1e-4, 1e-2, 1.0], [r"$\mathregular{10^{-4}}$", r"$\mathregular{10^{-2}}$", "1"])
    sx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    sx.set_yticks([])
    for side in ("left", "top", "right"):
        sx.spines[side].set_visible(False)
    sx.set_xlabel(r"Relative $\mathregular{L^2}$ error", labelpad=2)
    sx.set_facecolor("none")


def case_tau(case_id):
    taus = set()
    for path in sorted((ROOT / "benchmark" / "cases").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if case_id in line:
                    rec = json.loads(line)
                    if rec.get("id") == case_id:
                        taus.add(float(rec["thresholds"]["tau_acc"]))
    assert len(taus) == 1, taus
    return taus.pop()


def draw_error_scatter(ax, primary_obs, pair_rows):
    pairs = defaultdict(dict)
    for row in primary_obs:
        pairs[row["pair_id"]][row["arm"]] = row
    assert len(pairs) == len(pair_rows) == 128
    categories = {row["pair_id"]: row["paired_category"] for row in pair_rows}
    xs, ys, concordant = [], [], []
    for pair_id, arms in pairs.items():
        xs.append(float(arms["visible"]["rel_l2"]))
        ys.append(float(arms["hidden"]["rel_l2"]))
        concordant.append(categories[pair_id] == "both_pass")
    xs, ys, concordant = np.array(xs), np.array(ys), np.array(concordant)
    lims = (5e-6, 3e1)
    ax.plot(lims, lims, color=PALETTE["spine"], lw=0.7, ls="--", zorder=1)
    ax.scatter(xs[concordant], ys[concordant], s=14, color=PALETTE["route_valid"], alpha=0.75, edgecolor="none", zorder=2, label="Both pass")
    ax.scatter(xs[~concordant], ys[~concordant], s=26, color=PALETTE["secondary"], edgecolor="white", linewidth=0.5, zorder=3, label="Retained fails")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    log_tick_labels(ax, "x", [-5, -3, -1, 1])
    log_tick_labels(ax, "y", [-5, -3, -1, 1])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"Relative $\mathregular{L^2}$ error, field supplied", labelpad=12)
    ax.set_ylabel(r"Relative $\mathregular{L^2}$ error, field retained", labelpad=18)
    clean_axis(ax)
    ax.set_title("Per-pair error", loc="left")
    ax.legend(loc="lower right", fontsize=7.0, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.1, markerscale=1.2)


def main() -> int:
    apply_editorial_style()
    pair_rows = read_csv("paired_endpoint_outcomes.csv")
    route_rows = read_csv("program_route_categories.csv")
    arm_rows = read_csv("arm_summary.csv")
    primary_obs = read_csv("primary_observations.csv")
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    if summary["primary"]["calls"] != 256 or len(pair_rows) != 128 or len(primary_obs) != 256:
        raise AssertionError("Source Data do not match prespecified design")

    fig = plt.figure(figsize=figure_size(85), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[0.62, 0.78, 1.32], height_ratios=[1.0, 1.0], wspace=0.55, hspace=0.62, left=0.075, right=0.985, top=0.91, bottom=0.11)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2])
    draw_pair_matrix(ax_a, pair_rows)
    draw_routes(ax_b, route_rows, arm_rows)
    _p = ax_b.get_position(); ax_b.set_position([_p.x0 + 0.03, _p.y0, _p.width - 0.03, _p.height])
    draw_alignment(ax_c, primary_obs, summary)
    draw_error_scatter(ax_d, primary_obs, pair_rows)
    for letter, (x, y) in {"a": (0.008, 0.975), "b": (0.27, 0.975), "c": (0.008, 0.512), "d": (0.565, 0.47)}.items():
        fig.text(x, y, letter, fontsize=8.0, fontweight="bold", color=INK, va="top", ha="left")

    PAPER_OUT.parent.mkdir(parents=True, exist_ok=True)
    NATURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "svg": {"bbox_inches": None},
        "pdf": {"bbox_inches": None},
        "png": {"dpi": 300, "bbox_inches": None},
        "tiff": {"dpi": 300, "bbox_inches": None, "pil_kwargs": {"compression": "tiff_lzw"}},
    }
    for suffix, kwargs in outputs.items():
        path = PAPER_OUT.with_suffix("." + suffix)
        fig.savefig(path, facecolor="white", **kwargs)
        shutil.copy2(path, NATURE_OUT.with_suffix("." + suffix))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
