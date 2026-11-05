#!/usr/bin/env python3
"""Render Fig. 2 (route review and route test) and Supplementary Fig. 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from _nmi_editorial_style import (
    PALETTE,
    apply_editorial_style,
    clean_axis,
    figure_size,
    finding_title,
    panel_label,
    save_figure,
)

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper" / "figures"
NATURE = ROOT / "Nature" / "figs"
AXES = ["O", "G", "B", "D", "S", "C"]

OUTCOME_COLORS = {
    "pass": PALETTE["route_valid"],
    "F-Acc": "#9AA7B1",
    "F-Exec": "#7E8992",
    "F-Time": "#B2BAC1",
    "HOST_EVAL_DOMAIN_MASK_MISMATCH": PALETTE["neutral"],
    "host-mask no-verdict": PALETTE["neutral"],
    "agent-error / timeout / other": "#E5E8EB",
    "other": "#E5E8EB",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_pair(fig_builder, paper_stem: Path, nature_stem: Path) -> None:
    fig = fig_builder()
    save_figure(fig, paper_stem, tiff=True)
    fig = fig_builder()
    save_figure(fig, nature_stem, tiff=True)


def forest(ax, rows, *, estimate_key="estimate", low_key="ci95_low", high_key="ci95_high", color=None, labels=None, xlabel="Estimate"):
    y = np.arange(len(rows))[::-1]
    estimates = np.array([float(row[estimate_key]) for row in rows])
    lows = np.array([float(row[low_key]) for row in rows])
    highs = np.array([float(row[high_key]) for row in rows])
    ax.axvline(0, color=PALETTE["spine"], lw=0.7, zorder=0)
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        ms=4.0,
        color=color or PALETTE["route_valid"],
        ecolor=color or PALETTE["route_valid"],
        elinewidth=0.9,
        capsize=2,
    )
    ax.set_yticks(y, labels or [row.get("label", "") for row in rows])
    ax.set_xlabel(xlabel)
    clean_axis(ax, grid_axis="x")


ROUTE_ROWS = [("known_numerical", "Known numerical"), ("known_direct", "Known direct or mixed"),
              ("constructed", "Constructed from targets"), ("dependency", "Dependency-complex")]
ROUTE_CLASS = {"follows": ("Follows the change", PALETTE["route_valid"], PALETTE["route_valid"], "o"),
               "instance_specific": ("Instance-specific", "#FFFFFF", PALETTE["route_valid"], "o"),
               "target_derived": ("Target-derived", PALETTE["claim_invalid"], PALETTE["claim_invalid"], "o")}
INK = PALETTE["text"]; MUTED = PALETTE["secondary"]


def route_test_rows() -> list[dict]:
    """Supplementary Data 1: per-program errors of the two-stage route test (135 programs)."""
    rows = read_csv(ROOT / "paper/source_data/supplementary_data_1_route_test/supplementary_data_1_route_test.csv")
    assert len(rows) == 135
    out = []
    for r in rows:
        if r["set"] == "dependency-complex":
            grp = "dependency"
        elif r["set"] == "constructed":
            grp = "constructed"
        else:
            grp = "known_numerical" if r["ledger_route"] == "route_valid" else "known_direct"
        bound = max(0.01, 10.0 * float(r["e0"]))
        ev = float(r["e_v"]); evin = float(r["e_v_in"]) if r["e_v_in"] not in ("", "nan", "NaN") else None
        cls = r["route_test_class_final"]
        if ev <= bound:
            assert cls == "follows"
        elif evin is not None and evin <= bound:
            assert cls == "instance_specific"
        else:
            assert cls == "target_derived"
        out.append({"group": grp, "s1": ev / bound, "s2": None if evin is None else evin / bound, "cls": cls})
    n = Counter(r["group"] for r in out)
    assert [n[g] for g, _ in ROUTE_ROWS] == [20, 20, 5, 90]
    return out


def _log_ticks(ax, exps, labels=None):
    ax.set_xticks([10.0**k for k in exps])
    ax.set_xticklabels(labels or [("1" if k == 0 else rf"$\mathregular{{10^{{{k}}}}}$") for k in exps])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _rows_axis(ax, n_rows):
    ax.set_ylim(n_rows - 0.45, -0.55)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)


def sina_offsets(x, rng, half_width=0.26):
    """Vertical jitter scaled by the local density of log10(x) (sina layout)."""
    lx = np.log10(np.asarray(x, dtype=float))
    if len(lx) < 3:
        return rng.uniform(-0.5, 0.5, len(lx)) * half_width
    bw = 0.25
    dens = np.array([np.exp(-0.5 * ((lx - v) / bw) ** 2).sum() for v in lx])
    dens = dens / dens.max()
    return rng.uniform(-1, 1, len(lx)) * half_width * np.maximum(dens, 0.18)


def draw_route_test(ax1, ax2, rows) -> None:
    rng = np.random.default_rng(7)
    ypos = {g: k for k, (g, _) in enumerate(ROUTE_ROWS)}
    for ax, key in [(ax1, "s1"), (ax2, "s2")]:
        for k in range(len(ROUTE_ROWS)):
            ax.axhline(k, color=PALETTE["grid"], lw=0.4, zorder=0)
        for cls, (label, face, edge, marker) in ROUTE_CLASS.items():
            pts = [r for r in rows if r["cls"] == cls and r[key] is not None]
            x = np.array([max(r[key], 1e-10) for r in pts])
            g_arr = np.array([ypos[r["group"]] for r in pts]); y = g_arr.astype(float)
            for gk in set(g_arr.tolist()):
                m = g_arr == gk; y[m] = gk + sina_offsets(x[m], rng)
            ax.scatter(x, y, s=9, marker=marker, facecolor=face, edgecolor=edge, linewidth=0.6, alpha=0.9, zorder=3)
        ax.axvline(1.0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.set_xscale("log"); ax.set_xlim(10**-10.3, 10**4.3)
        _log_ticks(ax, [-8, -4, 0, 4])
        _rows_axis(ax, len(ROUTE_ROWS))
        ax.set_xlabel("Error / pass bound")
        counts = Counter(r["group"] for r in rows if r[key] is not None)
        ax.set_yticks(range(len(ROUTE_ROWS)))
        if key == "s1":
            ax.set_yticklabels([lab for _, lab in ROUTE_ROWS])
        else:
            ax.set_yticklabels([])
        for g, k in ypos.items():
            ax.text(1.0, k, f" {counts[g]}", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    ha="left", va="center", fontsize=7, color=MUTED)
        ax.text(1.0, -0.55, " n", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                ha="left", va="bottom", fontsize=7, color=MUTED)
        ax.text(1.0, -0.62, "pass bound", ha="center", va="bottom", fontsize=7, color=INK)
    ax1.set_title("Changed problem", loc="left", pad=12)
    ax2.set_title("Data keeping only the hard-coded structure", loc="left", pad=12)
    handles = [Line2D([], [], ls="none", marker="o", markersize=3.6, markerfacecolor=f, markeredgecolor=e, markeredgewidth=0.6, label=l)
               for l, f, e, m in ROUTE_CLASS.values()]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.74), frameon=False, fontsize=7,
               handletextpad=0.2, labelspacing=0.35, borderaxespad=0.0)


def render_ed1() -> None:
    """Main Fig. 2: route review of the primary-panel passes and the route test (180 x 128 mm)."""
    source = ROOT / "paper/source_data/fig3_exact_formula_audit"
    summary = read_json(source / "summary.json")
    evidence = read_csv(source / "audit_evidence.csv")
    counts = summary["classification_counts"]
    n_near_zero = int(summary["n_near_zero_runs"])
    assert [counts["DIRECT_EXACT_FORMULA"], counts["MIXED_SOLVER_WITH_EXACT_HINTS"], counts["NUMERICAL_SOLVER_NEAR_ZERO"]] == [120, 5, 0]
    assert sum(counts.values()) == n_near_zero == 125

    cards = [
        ("Numerical solve", False, "heat, square", "1.451e-04",
         ["u = solve_A(rhs).astype(np.float64, copy=False)"],
         ["solution = U_eval.astype(np.float64, copy=False).reshape(-1)"]),
        ("Direct reconstruction", True, "Poisson, square", "8.079e-17",
         ["# no solver call"],
         ["solution_2d = (np.exp(X + Y - 1.0)", "               * np.sin(pi * X) * np.sin(pi * Y))"]),
        ("Mixed return", True, "heat, annulus", "0.0",
         ['_ = spla.spsolve(sp.eye(1, format="csc"), np.ones(1))'],
         ["solution_grid = (np.exp(-T_final)", "                 * np.sin(pi * (rr - r_in) / L)", "                 * np.cos(2.0 * tt))"]),
    ]
    by_label = {row["review_label"]: row for row in evidence}
    assert by_label["DIRECT_EXACT_FORMULA"]["rel_l2"] == cards[1][3]
    assert by_label["NUMERICAL_SOLVER"]["rel_l2"] == cards[0][3]

    def sci(value: str) -> str:
        v = float(value)
        if v == 0.0:
            return "0"
        exp = int(np.floor(np.log10(abs(v))))
        return rf"{v / 10**exp:.1f}$\times\mathregular{{10^{{{exp}}}}}$"

    errs = read_csv(source / "primary_pass_errors.csv")
    assert len(errs) == 1162
    route_rows = route_test_rows()
    groups = [("route_valid", "Numerical solve", PALETTE["route_valid"]),
              ("direct_recovery", "Direct reconstruction", PALETTE["claim_invalid"]),
              ("mixed_route", "Mixed return", PALETTE["claim_invalid"])]

    def build():
        fig = plt.figure(figsize=figure_size(128))
        ax_a = fig.add_axes([0.205, 0.585, 0.30, 0.335])
        ax_b = fig.add_axes([0.55, 0.545, 0.445, 0.40]); ax_b.set_axis_off()
        ax_c = fig.add_axes([0.205, 0.095, 0.30, 0.305])
        ax_d = fig.add_axes([0.625, 0.095, 0.30, 0.305])

        rng = np.random.default_rng(3)
        zero_x = 10**-18.5
        for k, (key, label, color) in enumerate(groups):
            ax_a.axhline(k, color=PALETTE["grid"], lw=0.4, zorder=0)
            v = np.array([float(r["rel_l2"]) for r in errs if r["route_status"] == key])
            x = np.where(v > 0, v, zero_x * 10 ** rng.uniform(-0.15, 0.15, len(v)))
            y = k + sina_offsets(x, rng, 0.30)
            face = color if key != "mixed_route" else "#E9C2B3"
            ax_a.scatter(x, y, s=3.5 if key == "route_valid" else 9, facecolor=face, edgecolor=color,
                         linewidth=0.5 if key != "route_valid" else 0.0, alpha=0.35 if key == "route_valid" else 0.9, zorder=3)
            ax_a.text(1.0, k, f" {len(v):,}", transform=matplotlib.transforms.blended_transform_factory(ax_a.transAxes, ax_a.transData),
                      ha="left", va="center", fontsize=7, color=MUTED)
        ax_a.text(1.0, -0.55, " n", transform=matplotlib.transforms.blended_transform_factory(ax_a.transAxes, ax_a.transData),
                  ha="left", va="bottom", fontsize=7, color=MUTED)
        ax_a.set_xscale("log"); ax_a.set_xlim(10**-19.2, 10**-0.6)
        _log_ticks(ax_a, [-18.5, -16, -12, -8, -4], ["0", r"$\mathregular{10^{-16}}$", r"$\mathregular{10^{-12}}$", r"$\mathregular{10^{-8}}$", r"$\mathregular{10^{-4}}$"])
        _rows_axis(ax_a, 3)
        ax_a.set_yticks(range(3)); ax_a.set_yticklabels([g[1] for g in groups])
        for xb in (10**-17.55, 10**-17.35):
            ax_a.plot([xb, xb], [2.47, 2.63], color=PALETTE["spine"], lw=0.6, clip_on=False, transform=ax_a.transData, zorder=5)
        ax_a.axvline(1e-2, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax_a.axvline(1e-14, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=2)
        ax_a.text(1e-2, -0.62, r"$\tau_{\mathrm{acc}}$", ha="center", va="bottom", fontsize=7, color=INK)
        ax_a.set_xlabel(r"Relative $\mathregular{L^2}$ error")
        ax_a.text(1e-14, -0.62, "inspection", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax_a.set_title("Errors of the 1,162 primary-panel passes", loc="left", pad=19)

        ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
        ax_b.text(0.0, 1.072, "Returned-field lines of three passing programs", ha="left", va="bottom", fontsize=7.5, color=INK)
        line_h, y = 0.062, 0.97
        for label, invalid, where, err, context, returned in cards:
            color = PALETTE["claim_invalid"] if invalid else PALETTE["route_valid"]
            ax_b.text(0.0, y, label, ha="left", va="top", fontsize=7, color=color)
            ax_b.text(0.995, y, f"{where}, error {sci(err)}", ha="right", va="top", fontsize=7, color=MUTED)
            y -= 0.075
            n_lines = len(context) + len(returned)
            top = y + 0.012
            ax_b.add_patch(Rectangle((0.0, top - n_lines * line_h - 0.022), 1.0, n_lines * line_h + 0.022, facecolor="#F4F5F6", edgecolor="none", zorder=0))
            ax_b.add_patch(Rectangle((0.0, top - n_lines * line_h - 0.022), 0.006, n_lines * line_h + 0.022, facecolor=color, edgecolor="none", zorder=1))
            for line in context:
                ax_b.text(0.025, y, line, ha="left", va="top", fontsize=6.0, family="monospace", color=MUTED if line.startswith("#") else INK)
                y -= line_h
            for line in returned:
                ax_b.text(0.025, y, line, ha="left", va="top", fontsize=6.0, family="monospace", color=color)
                y -= line_h
            y -= 0.06

        draw_route_test(ax_c, ax_d, route_rows)
        for letter, (x, yy) in {"a": (0.01, 0.985), "b": (0.53, 0.985), "c": (0.01, 0.47), "d": (0.555, 0.47)}.items():
            fig.text(x, yy, letter, fontsize=8, fontweight="bold", color=INK, va="top")
        return fig

    with matplotlib.rc_context():
        apply_editorial_style()
        save_pair(build, PAPER / "fig3_exact_formula_audit", NATURE / "fig3_exact_formula_review")


def render_ed2() -> None:
    report = read_json(ROOT / "paper/source_data/supp_note_formula_blind_controls/wave1_report.json")
    per_case = report["per_case"]
    cells = ["O0G0B0D0S0C0", "O0G0B1D0S0C0", "O0G1B0D0S0C0", "O0G1B1D0S0C0"]
    labels = ["G0 B0", "G0 B1", "G1 B0", "G1 B1"]
    descriptors = ["square · Dirichlet", "square · mixed", "multi-geometry", "annulus · mixed"]
    counts = {cell: Counter() for cell in cells}
    errors = defaultdict(list)
    for entry in per_case:
        a = entry["axes"]
        cell = f"O{a['O']}G{a['G']}B{a['B']}D{a['D']}S{a['S']}C{a['C']}"
        counts[cell]["pass"] += int(entry["n_passing"])
        counts[cell]["F-Acc"] += int(entry["n_verdicts"]) - int(entry["n_passing"])
        counts[cell]["mask/other"] += int(entry["n_no_verdict"])
        if int(entry["n_passing"]):
            errors[cell].append(float(entry["rel_l2_median"]))
    assert all(sum(counts[cell].values()) == 36 for cell in cells)

    def build():
        fig = plt.figure(figsize=figure_size(92))
        grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.46)
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        y = np.arange(4)[::-1]
        left = np.zeros(4)
        for key, color, legend_label in [("pass", PALETTE["route_valid"], "Endpoint pass"), ("F-Acc", "#9AA7B1", "Accuracy failure"), ("mask/other", PALETTE["neutral"], "Mask/other")]:
            values = np.array([counts[cell][key] for cell in cells])
            ax_a.barh(y, values, left=left, height=0.58, color=color, label=legend_label)
            for yy, start, value in zip(y, left, values, strict=True):
                if value >= 4:
                    ax_a.text(start + value / 2, yy, str(value), ha="center", va="center", fontsize=6.5, color="white" if key != "mask/other" else PALETTE["text"], fontweight="semibold")
            left += values
        ax_a.set_yticks(y, [f"{lab}\n{desc}" for lab, desc in zip(labels, descriptors, strict=True)])
        ax_a.set_xlim(0, 36)
        ax_a.set_xlabel("Programs per cell (n=36)")
        ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=1.0)
        clean_axis(ax_a, grid_axis="x")
        ax_a.set_title("Outcome composition by G/B cell", pad=16)
        panel_label(ax_a, "a", x=-0.30, y=1.13)

        data = [errors[cell] if errors[cell] else [np.nan] for cell in cells]
        ax_b.boxplot(data, vert=False, positions=y, widths=0.48, patch_artist=True, boxprops={"facecolor": "#DCE6EE", "edgecolor": PALETTE["route_valid"], "lw": 0.8}, whiskerprops={"color": PALETTE["route_valid"], "lw": 0.8}, capprops={"color": PALETTE["route_valid"], "lw": 0.8}, medianprops={"color": PALETTE["claim_invalid"], "lw": 1.1}, flierprops={"marker": "o", "markersize": 2.6, "markerfacecolor": "white", "markeredgecolor": PALETTE["route_valid"]})
        ax_b.set_xscale("log")
        ax_b.set_yticks(y, labels)
        ax_b.set_xlabel("Per-case median relative L2 among passes")
        clean_axis(ax_b, grid_axis="x")
        ax_b.set_title("Accuracy among passing programs", pad=16)
        panel_label(ax_b, "b", x=-0.13, y=1.13)
        fig.subplots_adjust(left=0.16, right=0.99, bottom=0.15, top=0.86)
        return fig

    save_pair(build, PAPER / "fig_extdata3_formula_blind", NATURE / "edfig3_formulablind")


def render_ed3() -> None:
    source = ROOT / "paper/source_data/fig_external_naturalistic_validation"
    rows = read_csv(source / "run_level_source_data.csv")
    summary = read_json(source / "summary.json")
    assert len(rows) == 128 and all(row["endpoint_pass"] == "True" and row["method_pass"] == "True" for row in rows)
    families = ["heterogeneous_diffusion", "advection_diffusion_reaction", "lshape_poisson", "coupled_elliptic"]
    family_labels = ["Heterogeneous diffusion", "Advection–diffusion–reaction", "L-shaped Poisson", "Coupled elliptic"]
    compact_family_labels = ["Heterogeneous\ndiffusion", "Advection–diffusion–\nreaction", "L-shaped\nPoisson", "Coupled elliptic"]
    agents = ["Codex GPT-5.5", "Claude Opus 4.7"]

    def build():
        fig = plt.figure(figsize=figure_size(98))
        grid = fig.add_gridspec(1, 3, width_ratios=[0.78, 1.18, 0.95], wspace=0.48)
        ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]
        finding_title(fig, "Reference-hidden naturalistic tasks remain viable and converged")
        y = np.arange(4)[::-1]
        for offset, agent, color, marker in [(-0.12, agents[0], PALETTE["codex"], "o"), (0.12, agents[1], PALETTE["claude"], "s")]:
            values = []
            for family in families:
                subset = [row for row in rows if row["agent"] == agent and row["family"] == family]
                values.append(sum(row["endpoint_pass"] == "True" for row in subset))
            ax_a.scatter(values, y + offset, color=color, marker=marker, s=26, label=agent.split()[0])
        for yy in y:
            ax_a.text(16.35, yy, "16/16", va="center", fontsize=6.5, color=PALETTE["secondary"])
        ax_a.set_yticks(y, compact_family_labels)
        ax_a.set_xlim(14.8, 17.1)
        ax_a.set_xticks([15, 16])
        ax_a.set_xlabel("Output-level passes")
        ax_a.legend(loc="upper left", fontsize=6.5)
        clean_axis(ax_a, grid_axis="x")
        ax_a.set_title("All 128 records pass\n0 invalid routes detected", fontsize=7.8)
        panel_label(ax_a, "a", x=-0.18, y=1.06)

        rng = np.random.default_rng(20260814)
        positions = np.arange(4)[::-1]
        for agent_index, (agent, color) in enumerate([(agents[0], PALETTE["codex"]), (agents[1], PALETTE["claude"])]):
            for pos, family in zip(positions, families, strict=True):
                vals = np.array([float(row["normalized_rel_l2"]) for row in rows if row["agent"] == agent and row["family"] == family])
                jitter = rng.uniform(-0.08, 0.08, len(vals)) + (agent_index - 0.5) * 0.20
                ax_b.scatter(vals, pos + jitter, s=10, color=color, alpha=0.65, edgecolor="none")
                ax_b.plot([np.median(vals)] * 2, [pos - 0.15 + (agent_index - 0.5) * 0.20, pos + 0.15 + (agent_index - 0.5) * 0.20], color=color, lw=1.1)
        ax_b.set_xscale("log")
        ax_b.axvline(1.0, color=PALETTE["spine"], lw=0.8, ls="--")
        ax_b.set_yticks(positions, family_labels)
        ax_b.set_xlabel("Relative L2 / threshold")
        clean_axis(ax_b, grid_axis="x")
        ax_b.set_title("Accepted records retain\naccuracy margin", fontsize=7.8)
        panel_label(ax_b, "b", x=-0.14, y=1.06)

        conv = summary["reference_convergence"]
        median = 100 * float(conv["median_gap_relative_l2"])
        maximum = 100 * float(conv["maximum_gap_relative_l2"])
        limit = 100 * float(conv["acceptance_limit_relative_l2"])
        ax_c.hlines([1, 0], 0, [median, maximum], color=[PALETTE["route_valid"], PALETTE["reference_hidden"]], lw=2)
        ax_c.scatter([median, maximum], [1, 0], color=[PALETTE["route_valid"], PALETTE["reference_hidden"]], s=32)
        ax_c.axvline(limit, color=PALETTE["claim_invalid"], lw=0.9, ls="--")
        ax_c.text(limit, 0.96, "5% limit", transform=ax_c.get_xaxis_transform(), ha="right", va="top", fontsize=6.5, color=PALETTE["claim_invalid"])
        ax_c.text(median + 0.12, 1, f"{median:.3f}%", va="center", fontsize=6.8)
        ax_c.text(maximum - 0.12, 0, f"{maximum:.3f}%", ha="right", va="center", fontsize=6.8)
        ax_c.set_yticks([1, 0], ["Median", "Maximum"])
        ax_c.set_xlim(0, 5.25)
        ax_c.set_xlabel("61-to-121-grid reference gap (%)")
        clean_axis(ax_c, grid_axis="x")
        ax_c.set_title("References remain below\nthe 5% limit", fontsize=7.8)
        panel_label(ax_c, "c", x=-0.15, y=1.06)
        fig.subplots_adjust(left=0.15, right=0.99, bottom=0.16, top=0.86)
        return fig

    save_pair(build, PAPER / "fig_external_validation", NATURE / "fig_external_validation")


def grouped_agent_forest(ax, rows, *, value="coefficient", low="ci95_low", high="ci95_high", agent_key="agent", xlabel="Coefficient"):
    lookup = {(row["axis"], row[agent_key]): row for row in rows}
    y = np.arange(6)[::-1]
    for offset, agent, color, marker in [(-0.10, "Codex GPT-5.5", PALETTE["codex"], "o"), (0.10, "Claude Opus 4.7", PALETTE["claude"], "s")]:
        est = np.array([float(lookup[(axis, agent)][value]) for axis in AXES])
        lo = np.array([float(lookup[(axis, agent)][low]) for axis in AXES])
        hi = np.array([float(lookup[(axis, agent)][high]) for axis in AXES])
        ax.errorbar(est, y + offset, xerr=np.vstack([est - lo, hi - est]), fmt=marker, ms=3.4, color=color, ecolor=color, elinewidth=0.8, capsize=1.6, label=agent.split()[0])
    ax.axvline(0, color=PALETTE["spine"], lw=0.7)
    ax.set_yticks(y, AXES)
    ax.set_xlabel(xlabel)
    clean_axis(ax, grid_axis="x")


def render_ed4() -> None:
    source = ROOT / "paper/source_data/edfig9_codex_claude_axis1024"
    agents = read_csv(source / "agent_summary.csv")
    conditional = read_csv(source / "conditional_normalized_coefficients.csv")
    paired = read_csv(source / "paired_conditional_delta.csv")
    failure = read_csv(source / "failure_fraction_coefficients.csv")
    order = read_csv(source / "execution_order_diagnostics.csv")
    stage = read_csv(ROOT / "paper/source_data/fig3_outcome_stage_route_conditioning/three_estimand_main_effects.csv")
    assert [int(row["scheduled_records"]) for row in agents] == [3072, 3072]

    def build():
        fig = plt.figure(figsize=figure_size(166))
        grid = fig.add_gridspec(3, 2, hspace=0.58, wspace=0.42)
        axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
        ax = axes[0]
        y = np.arange(2)[::-1]
        left = np.zeros(2)
        classes = [("outcome_pass", "Endpoint pass", PALETTE["route_valid"]), ("outcome_F-Acc", "Accuracy failure", "#9AA7B1"), ("outcome_F-Time", "Timeout", "#7E8992"), ("outcome_HOST_EVAL_DOMAIN_MASK_MISMATCH", "Host mask", PALETTE["neutral"])]
        for key, label, color in classes:
            vals = np.array([int(row[key]) for row in agents])
            ax.barh(y, vals, left=left, height=0.56, color=color, label=label)
            left += vals
        ax.set_yticks(y, ["Codex", "Claude Code"])
        ax.set_ylim(-0.5, 2.05)
        ax.set_xlabel("Calls (n=3,072 per agent)")
        ax.legend(loc="upper center", ncol=4, fontsize=6.5, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, 1.02))
        clean_axis(ax, grid_axis="x")
        ax.set_title("Outcome composition")
        panel_label(ax, "a", x=-0.12, y=1.06)

        ax = axes[1]
        grouped_agent_forest(ax, conditional, xlabel="Conditional coefficient")
        ax.set_title("Coefficients among common passes (n=512)")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Codex", "Claude Code"], loc="lower right", fontsize=6.5)
        panel_label(ax, "b", x=-0.12, y=1.06)

        ax = axes[2]
        forest(ax, paired, estimate_key="coefficient_delta", color=PALETTE["route_valid"], labels=AXES, xlabel="Codex − Claude Code coefficient")
        ax.set_title("Paired Codex − Claude Code differences")
        panel_label(ax, "c", x=-0.12, y=1.06)

        ax = axes[3]
        grouped_agent_forest(ax, failure, xlabel="Failure-fraction coefficient")
        ax.set_title("Failure-fraction coefficients, all cases")
        panel_label(ax, "d", x=-0.12, y=1.06)

        ax = axes[4]
        lookup = {(row["agent"], row["estimand"], row["axis"]): float(row["coefficient"]) for row in stage}
        yv = np.arange(6)[::-1]
        for offset, agent, color, marker in [(-0.10, "Codex GPT-5.5", PALETTE["codex"], "o"), (0.10, "Claude Opus 4.7", PALETTE["claude"], "s")]:
            delta = np.array([lookup[(agent, "route_valid_passes", axis)] - lookup[(agent, "endpoint_passes", axis)] for axis in AXES])
            ax.scatter(delta, yv + offset, s=18, color=color, marker=marker)
        ax.axvline(0, color=PALETTE["spine"], lw=0.7)
        ax.set_yticks(yv, AXES)
        ax.set_xlabel("Route-valid − endpoint-pass coefficient")
        clean_axis(ax, grid_axis="x")
        ax.set_title("Route-valid minus endpoint-pass coefficients")
        panel_label(ax, "e", x=-0.12, y=1.06)

        ax = axes[5]
        lookup_order = defaultdict(dict)
        for row in order:
            lookup_order[row["agent"]][row["axis"]] = float(row["abs_pearson_r_axis_vs_schedule_order"])
        x = np.arange(6)
        for agent, color, marker in [("Codex GPT-5.5", PALETTE["codex"], "o"), ("Claude Opus 4.7", PALETTE["claude"], "s")]:
            vals = [max(lookup_order[agent][axis], 1e-5) for axis in AXES]
            ax.plot(x, vals, color=color, marker=marker, ms=3.5, label="Codex" if agent.startswith("Codex") else "Claude Code")
        ax.set_yscale("log")
        ax.set_xticks(x, AXES)
        ax.set_ylabel("|r| with submission order")
        ax.legend(loc="best", fontsize=6.5)
        clean_axis(ax, grid_axis="y")
        ax.set_title("Axis correlation with submission order")
        panel_label(ax, "f", x=-0.12, y=1.06)
        fig.subplots_adjust(left=0.10, right=0.99, bottom=0.07, top=0.95)
        return fig

    save_pair(build, PAPER / "edfig9_codex_claude_axis1024", NATURE / "edfig9_codex_claude_axis1024")


def outcome_counts(rows, arm_keys):
    return {label: Counter(row[column] for row in rows) for label, column in arm_keys}


def draw_outcome_composition(ax, counts, arms, route_rows=None):
    order = ["pass", "F-Acc", "F-Time", "HOST_EVAL_DOMAIN_MASK_MISMATCH", "F-Exec"]
    y = np.arange(len(arms))[::-1]
    left = np.zeros(len(arms))
    segment_starts = {}
    for cls in order:
        vals = np.array([counts[arm].get(cls, 0) for arm in arms])
        if vals.sum() == 0:
            continue
        segment_starts[cls] = left.copy()
        ax.barh(y, vals, left=left, height=0.56, color=OUTCOME_COLORS.get(cls, "#E5E8EB"), label=cls)
        left += vals
    for cls in order:
        if cls not in segment_starts:
            continue
        value = counts[arms[0]].get(cls, 0)
        if value >= 12:
            start = segment_starts[cls][0]
            label = "host mask" if cls == "HOST_EVAL_DOMAIN_MASK_MISMATCH" else cls
            ax.text(start + value / 2, y[0], label, ha="center", va="center", fontsize=6.5, color="white" if cls == "pass" else PALETTE["text"], fontweight="semibold")
    if route_rows:
        route_lookup = {row["arm"]: row for row in route_rows}
        mapping = {"Full": "full", "Prescription": "prescription_only", "No-code": "choice_free_no_code", "Full guide": "full_guide", "Minimal guide": "minimal_guide"}
        for yy, arm in zip(y, arms, strict=True):
            row = route_lookup[mapping[arm]]
            ax.scatter([int(row["route_valid_composite_pass"])], [yy], marker="|", s=160, linewidth=1.6, color=PALETTE["route_valid"], zorder=4)
            if int(row["confirmed_direct"]):
                ax.text(int(row["route_valid_composite_pass"]) - 1, yy - 0.27, f"{int(row['confirmed_direct'])} direct", ha="right", va="top", fontsize=6.5, color=PALETTE["claim_invalid"])
    ax.set_yticks(y, arms)
    ax.set_xlabel("Calls per arm")
    clean_axis(ax, grid_axis="x")


def render_ed5() -> None:
    source = ROOT / "paper/source_data/edfig_prescription_code_separation"
    rows = read_csv(source / "paired_outcomes.csv")
    route = read_csv(source / "postreview_route_readout.csv")
    summary = read_json(source / "summary.json")
    counts = outcome_counts(rows, [("Full", "full_class"), ("Prescription", "prescription_class"), ("No-code", "no_code_class")])
    tests = summary["ordered_confirmatory_tests"]
    mechanism = summary["program_mechanism"]["registered_contrasts"]

    def build():
        fig = plt.figure(figsize=figure_size(145))
        grid = fig.add_gridspec(2, 3, height_ratios=[0.88, 1.15], hspace=0.50, wspace=0.54)
        ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]
        ax_d = fig.add_subplot(grid[1, :])
        finding_title(fig, "Guide wording changes resource calibration and route mixture")
        draw_outcome_composition(ax_a, counts, ["Full", "Prescription", "No-code"], route)
        ax_a.set_title("Output and route-valid outcomes", fontsize=7.8)
        panel_label(ax_a, "a", x=-0.16, y=1.06)

        formatted = []
        names = ["Prescription − full", "Prescription − no-code"]
        for test, name in zip(tests, names, strict=True):
            ci = test["one_sided_95_interval"]
            formatted.append({"estimate": 100 * ci["estimate"], "ci95_low": 100 * ci["lower"], "ci95_high": 100 * ci["upper"], "label": name})
        forest(ax_b, formatted, labels=names, color=PALETTE["reference_hidden"], xlabel="Absolute pass difference (pp)")
        ax_b.axvline(-5, color=PALETTE["claim_invalid"], lw=0.8, ls=":")
        ax_b.set_title("Registered ordered tests", fontsize=7.8)
        panel_label(ax_b, "b", x=-0.15, y=1.06)

        table = tests[0]["paired_table"]
        matrix = np.array([[table["both_pass"], table["prescription_only_only"]], [table["full_guide_only"], table["both_fail"]]])
        ax_c.imshow(np.array([[1, 0], [0, 0]]), cmap=ListedColormap(["#EEF1F3", PALETTE["route_valid"]]), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax_c.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7.5, fontweight="semibold", color="white" if (i, j) == (0, 0) else PALETTE["text"])
        ax_c.set_xticks([0, 1], ["Full pass", "Full fail"])
        ax_c.set_yticks([0, 1], ["Prescription pass", "Prescription fail"])
        ax_c.tick_params(length=0)
        for spine in ax_c.spines.values(): spine.set_visible(False)
        ax_c.set_title("Full and prescription agree", fontsize=7.8)
        panel_label(ax_c, "c", x=-0.15, y=1.06)

        ax_d.set_axis_off()
        panel_label(ax_d, "d", x=-0.025, y=1.06)
        sub = ax_d.get_subplotspec().subgridspec(1, 3, wspace=0.48)
        subaxes = [fig.add_subplot(sub[0, i]) for i in range(3)]
        contrasts = [("prescription_only_minus_full_guide", "Prescription − full", PALETTE["route_valid"]), ("prescription_only_minus_choice_free_no_code", "Prescription − no-code", PALETTE["reference_hidden"])]
        metric_rows = []
        for key, label, color in contrasts:
            for metric, short in [("log10_grid_points_ratio", "grid"), ("log10_time_steps_ratio", "steps")]:
                d = mechanism[key]["paired_effects"][metric]
                compact = "P–full" if key == "prescription_only_minus_full_guide" else "P–no-code"
                metric_rows.append((compact + ": " + short, d, color))
        yy = np.arange(len(metric_rows))[::-1]
        subaxes[0].axvline(0, color=PALETTE["spine"], lw=0.7)
        for yv, (label, d, color) in zip(yy, metric_rows, strict=True):
            est=float(d["estimate"]); lo,hi=map(float,d["ci95"])
            subaxes[0].errorbar(est,yv,xerr=[[est-lo],[hi-est]],fmt="o",ms=3.5,color=color,capsize=1.8)
        subaxes[0].set_yticks(yy,[row[0] for row in metric_rows]); subaxes[0].set_xlabel(r"Mean paired $\Delta\log_{10}$"); subaxes[0].set_title("Program scale",fontsize=7.5); clean_axis(subaxes[0],grid_axis="x")
        error_rows=[]
        for key,label,color in contrasts:
            d=summary["paired_secondary_deltas"][key]["log10_rel_l2"]
            error_rows.append((label,d,color))
        subaxes[1].axvline(0,color=PALETTE["spine"],lw=0.7)
        for yv,(label,d,color) in zip([1,0],error_rows,strict=True):
            est=float(d["estimate"]); lo,hi=map(float,d["ci95"])
            subaxes[1].errorbar(est,yv,xerr=[[est-lo],[hi-est]],fmt="o",ms=3.5,color=color,capsize=1.8)
        subaxes[1].set_yticks([1,0],["P–full", "P–no-code"]); subaxes[1].set_xlabel(r"Mean $\Delta\log_{10}$ relative error"); subaxes[1].set_title("Numerical fidelity",fontsize=7.5); clean_axis(subaxes[1],grid_axis="x")
        vals=[]; labs=[]
        for key,label,color in contrasts:
            trace=mechanism[key]["runtime_calibration_trace"]; vals.append(trace["traceable_fraction"]); labs.append(label)
        subaxes[2].barh([1,0],vals,color=[PALETTE["route_valid"],PALETTE["reference_hidden"]],height=0.52)
        for yv,v,key in zip([1,0],vals,[row[0] for row in contrasts],strict=True):
            trace=mechanism[key]["runtime_calibration_trace"]; subaxes[2].text(v+0.02,yv,f"{trace['traceable_by_program_choice']}/{trace['runtime_comparable_pairs']}",va="center",fontsize=6.5)
        subaxes[2].set_yticks([1,0],["P–full", "P–no-code"]); subaxes[2].set_xlim(0,1.18); subaxes[2].set_xlabel("Traceable fraction"); subaxes[2].set_title("Runtime differences",fontsize=7.5); clean_axis(subaxes[2],grid_axis="x")
        fig.subplots_adjust(left=0.13,right=0.99,bottom=0.09,top=0.90)
        return fig

    save_pair(build, PAPER / "edfig_prescription_code_separation", NATURE / "edfig_prescription_code_separation")


def render_ed6() -> None:
    source = ROOT / "paper/source_data/edfig_scaffold_ablation"
    rows = read_csv(source / "paired_outcomes.csv")
    route = read_csv(source / "postreview_route_readout.csv")
    frontier = read_csv(source / "accuracy_runtime_frontier.csv")
    summary = read_json(source / "summary.json")
    counts = outcome_counts(rows, [("Full guide", "full_class"), ("Minimal guide", "minimal_class")])
    contrast = summary["paired_pass_contrast"]

    def build():
        fig = plt.figure(figsize=figure_size(152))
        grid = fig.add_gridspec(2, 3, height_ratios=[0.86, 1.12], width_ratios=[1.12, 0.82, 0.88], hspace=0.50, wspace=0.46)
        ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]
        ax_d = fig.add_subplot(grid[1, :2])
        ax_e = fig.add_subplot(grid[1, 2])
        finding_title(fig, "Minimal scaffolding changes both route mixture and resource calibration")
        draw_outcome_composition(ax_a, counts, ["Full guide", "Minimal guide"], route)
        ax_a.set_title("Output and route-valid outcomes", fontsize=7.8)
        panel_label(ax_a, "a", x=-0.16, y=1.06)

        matrix=np.array([[contrast["full_pass_minimal_pass"],contrast["full_pass_minimal_fail"]],[contrast["full_fail_minimal_pass"],contrast["full_fail_minimal_fail"]]])
        ax_b.imshow(np.array([[1,0],[0,0]]),cmap=ListedColormap(["#EEF1F3",PALETTE["route_valid"]]),vmin=0,vmax=1)
        for i in range(2):
            for j in range(2): ax_b.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=7.5,fontweight="semibold",color="white" if (i,j)==(0,0) else PALETTE["text"])
        ax_b.set_xticks([0,1],["Minimal pass","Minimal fail"]); ax_b.set_yticks([0,1],["Full pass","Full fail"]); ax_b.tick_params(length=0)
        for spine in ax_b.spines.values(): spine.set_visible(False)
        ax_b.set_title("19 full passes fail under\nminimal guidance", fontsize=7.8)
        panel_label(ax_b,"b",x=-0.16,y=1.06)

        effect=contrast["minimal_minus_full_absolute_pass_difference"]; est=100*float(effect["estimate"]); lo,hi=[100*float(v) for v in effect["ci95"]]
        ax_c.axvline(0,color=PALETTE["spine"],lw=0.7); ax_c.errorbar(est,0,xerr=[[est-lo],[hi-est]],fmt="o",ms=5,color=PALETTE["claim_invalid"],capsize=2.5)
        ax_c.set_yticks([0],["Minimal − full"]); ax_c.set_xlabel("Pass difference (pp)"); clean_axis(ax_c,grid_axis="x"); ax_c.set_title(f"Paired effect: {est:+.1f} pp", fontsize=7.8); panel_label(ax_c,"c",x=-0.16,y=1.06)

        xs=[]; ys=[]; crossing=[]
        for row in frontier:
            try: x=float(row["log10_solver_wall_seconds_ratio"]); y=float(row["log10_normalized_error_difference"])
            except (TypeError,ValueError): continue
            xs.append(x); ys.append(y); crossing.append(row.get("right_class")=="pass" and row.get("left_class")=="F-Time")
        xs=np.array(xs); ys=np.array(ys); crossing=np.array(crossing,dtype=bool)
        ax_d.scatter(xs[~crossing],ys[~crossing],s=13,color=PALETTE["route_valid"],alpha=0.50,edgecolor="none",label="Other paired outcomes")
        if crossing.any(): ax_d.scatter(xs[crossing],ys[crossing],s=30,marker="^",color=PALETTE["claim_invalid"],label="Full pass → minimal F-Time")
        ax_d.axhline(0,color=PALETTE["spine"],lw=0.7); ax_d.axvline(0,color=PALETTE["spine"],lw=0.7); ax_d.set_xlabel(r"$\Delta\log_{10}$ solver runtime (minimal/full)"); ax_d.set_ylabel(r"$\Delta\log_{10}$ normalised error"); ax_d.legend(loc="best",fontsize=6.5); clean_axis(ax_d); ax_d.set_title(f"Accuracy–runtime frontier (n={len(xs)})"); panel_label(ax_d,"d",x=-0.10,y=1.06)

        ax_e.set_axis_off(); panel_label(ax_e,"e",x=-0.10,y=1.06)
        sub=ax_e.get_subplotspec().subgridspec(2,1,hspace=0.65); top=fig.add_subplot(sub[0,0]); bottom=fig.add_subplot(sub[1,0])
        effects=summary["paired_program_effects"]; metrics=[("log10_normalized_error_difference","Error"),("log10_solver_wall_seconds_ratio","Runtime"),("log10_time_steps_ratio","Time steps"),("log10_grid_points_ratio","Grid points")]
        yy=np.arange(4)[::-1]; top.axvline(0,color=PALETTE["spine"],lw=0.7)
        for yv,(key,label) in zip(yy,metrics,strict=True):
            d=effects[key]; e=float(d["estimate"]); l,h=map(float,d["ci95"]); top.errorbar(e,yv,xerr=[[e-l],[h-e]],fmt="o",ms=3.3,color=PALETTE["route_valid"] if key.startswith("log10_normalized") else PALETTE["reference_hidden"],capsize=1.5)
        top.set_yticks(yy,[x[1] for x in metrics]); top.set_xlabel(r"Mean paired $\Delta\log_{10}$"); top.set_title("Submitted-program shifts",fontsize=7.5); clean_axis(top,grid_axis="x")
        d=summary["paired_log10_normalized_error"]; e=float(d["estimate"]); l,h=map(float,d["ci95"]); bottom.axvline(0,color=PALETTE["spine"],lw=0.7); bottom.errorbar(e,0,xerr=[[e-l],[h-e]],fmt="o",ms=3.5,color=PALETTE["route_valid"],capsize=1.8); bottom.set_yticks([0],["Minimal − full"]); bottom.set_xlabel(r"Mean $\Delta\log_{10}$ error"); bottom.set_title("Conditional numerical fidelity",fontsize=7.5); clean_axis(bottom,grid_axis="x")
        fig.subplots_adjust(left=0.13,right=0.99,bottom=0.10,top=0.90)
        return fig

    save_pair(build, PAPER / "edfig_scaffold_ablation", NATURE / "edfig_scaffold_ablation")


def render_ed7() -> None:
    source=ROOT/"paper/source_data/edfig8_gpt54_gpt55_sensitivity"
    agreement=read_csv(source/"case_pass_count_confusion.csv"); paired=read_csv(source/"paired_pass_summary.csv")[0]; failure=read_csv(source/"failure_fraction_coefficients.csv"); conditional=read_csv(source/"conditional_normalized_coefficients.csv")
    matrix=np.zeros((4,4),dtype=int)
    for row in agreement: matrix[int(row["gpt55_pass_count"]),int(row["gpt54_pass_count"])]=int(row["n_cases"])
    assert matrix.sum()==256 and int(paired["exact_pass_count_agreement_cases"])==240
    def backbone_forest(ax,rows):
        lookup={(r["axis"],r["backbone"]):r for r in rows}; y=np.arange(6)[::-1]
        for offset,b,color,marker,label in [(-0.10,"gpt55",PALETTE["route_valid"],"o","GPT-5.5"),(0.10,"gpt54",PALETTE["reference_hidden"],"s","GPT-5.4")]:
            e=np.array([float(lookup[(a,b)]["coefficient"]) for a in AXES]); l=np.array([float(lookup[(a,b)]["ci95_low"]) for a in AXES]); h=np.array([float(lookup[(a,b)]["ci95_high"]) for a in AXES]); ax.errorbar(e,y+offset,xerr=np.vstack([e-l,h-e]),fmt=marker,ms=3.5,color=color,capsize=1.6,elinewidth=0.8,label=label)
        ax.axvline(0,color=PALETTE["spine"],lw=0.7); ax.set_yticks(y,AXES); clean_axis(ax,grid_axis="x")
    def build():
        fig=plt.figure(figsize=figure_size(108)); grid=fig.add_gridspec(1,3,width_ratios=[0.92,1.08,1.18],wspace=0.45); ax_a,ax_b,ax_c=[fig.add_subplot(grid[0,i]) for i in range(3)]
        finding_title(fig,"Backbones preserve pass topology but alter conditional coefficients")
        region=np.fromfunction(lambda i,j:(i==j).astype(int),(4,4)); ax_a.imshow(region,cmap=ListedColormap(["#F1F3F5",PALETTE["route_valid"]]),vmin=0,vmax=1)
        for i in range(4):
            for j in range(4): ax_a.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=6.8,fontweight="semibold",color="white" if i==j else PALETTE["text"])
        ax_a.set_xticks(range(4),range(4)); ax_a.set_yticks(range(4),range(4)); ax_a.set_xlabel("GPT-5.4 passes per case"); ax_a.set_ylabel("GPT-5.5 passes per case"); ax_a.tick_params(length=0)
        for s in ax_a.spines.values(): s.set_visible(False)
        delta=100*float(paired["mean_pass_rate_delta_gpt55_minus_gpt54"]); lo=100*float(paired["delta_ci95_low"]); hi=100*float(paired["delta_ci95_high"])
        ax_a.set_title("240/256 cases agree exactly"); ax_a.text(0.5,-0.22,f"r = 0.982  ·  pass-rate Δ {delta:+.2f} pp\n95% CI {lo:+.2f} to {hi:+.2f} pp",transform=ax_a.transAxes,ha="center",va="top",fontsize=6.5,color=PALETTE["secondary"]); panel_label(ax_a,"a",x=-0.18,y=1.06)
        backbone_forest(ax_b,failure); ax_b.set_xlabel("Failure-fraction coefficient"); ax_b.set_title("All scheduled runs"); panel_label(ax_b,"b",x=-0.14,y=1.06)
        backbone_forest(ax_c,conditional); ax_c.set_xlabel("Conditional coefficient"); ax_c.set_title("Common passing cases (n=136)"); panel_label(ax_c,"c",x=-0.14,y=1.06)
        handles, labels = ax_b.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.27, 0.14), frameon=False, fontsize=6.5)
        fig.subplots_adjust(left=0.09,right=0.99,bottom=0.19,top=0.86); return fig
    save_pair(build,PAPER/"edfig8_gpt54_gpt55_sensitivity",NATURE/"edfig8_gpt54_gpt55_sensitivity")


def render_ed8() -> None:
    experiments=[
        ("Codex · clean SciPy","phase2_wave1b_codex_scipy_no_exact_20260523T224200Z"),
        ("Claude · clean SciPy","phase2_wave1b_claude_cli_scipy_no_exact_20260526T110547Z"),
        ("Codex · DOLFINx control","phase2_confirm72_codex_dolfinx_no_exact_20260526T022418Z"),
        ("Codex · DOLFINx v0.9 hints","phase2_confirm72_codex_dolfinx_v09hints_20260527T015929Z"),
        ("Codex · formula-blind Poisson","phase2_formula_blind_poisson48_codex_scipy_20260526T072300Z"),
    ]
    records=[]
    for label,exp in experiments:
        d=read_json(ROOT/"analysis/phase2"/exp/"wave1_report.json"); s=d.get("summary",d); fc=s.get("failure_code_counts",{}) or {}; nv=s.get("no_verdict_classification_counts",{}) or {}; total=int(s.get("n_records_total",0) or s.get("expected_total_calls",0)); values={"pass":int(fc.get("NONE",0)),"F-Acc":int(fc.get("F-Acc",0)),"F-Exec":int(fc.get("F-Exec",0)),"host-mask no-verdict":int(nv.get("HOST_EVAL_DOMAIN_MASK_MISMATCH",0))}; values["agent-error / timeout / other"]=max(0,total-sum(values.values())); records.append((label,total,values))
    assert [r[1] for r in records]==[1450,1450,216,216,144]
    def build():
        fig,ax=plt.subplots(figsize=figure_size(82)); finding_title(fig,"Execution outcomes differ across backend and formula-blind regimes")
        y=np.arange(5)[::-1]; left=np.zeros(5); order=["pass","F-Acc","F-Exec","host-mask no-verdict","agent-error / timeout / other"]
        for cls in order:
            vals=np.array([row[2][cls]/row[1]*100 for row in records]); ax.barh(y,vals,left=left,height=0.58,color=OUTCOME_COLORS[cls],label=cls)
            for yy,start,val,row in zip(y,left,vals,records,strict=True):
                count=row[2][cls]
                if val>=6: ax.text(start+val/2,yy,str(count),ha="center",va="center",fontsize=6.5,color="white" if cls in ("pass","F-Exec") else PALETTE["text"],fontweight="semibold")
            left+=vals
        ax.set_yticks(y,[r[0] for r in records]); ax.set_xlim(0,100); ax.set_xlabel("Records (%)"); clean_axis(ax,grid_axis="x")
        for yy,(_,total,_) in zip(y,records,strict=True): ax.text(100.1,yy,f"n={total:,}",va="center",fontsize=6.5,color=PALETTE["secondary"])
        ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.37),ncol=3,fontsize=6.5); panel_label(ax,"a",x=-0.10,y=1.04)
        fig.subplots_adjust(left=0.25,right=0.95,bottom=0.25,top=0.82); return fig
    save_pair(build,PAPER/"fig5_failure_taxonomy",NATURE/"fig5_taxonomy")


RENDERERS = {1: render_ed1, 2: render_ed2, 3: render_ed3, 4: render_ed4, 5: render_ed5, 6: render_ed6, 7: render_ed7, 8: render_ed8}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--figures",nargs="*",type=int,choices=range(1,9),default=list(range(1,9))); args=parser.parse_args()
    apply_editorial_style(extended_data=True)
    for number in args.figures:
        RENDERERS[number]()
        print(f"rendered Extended Data Fig. {number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
