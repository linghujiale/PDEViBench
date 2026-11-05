#!/usr/bin/env python3
"""Render Fig. 1 from its source data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.cm
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from _nmi_editorial_style import (
    PALETTE,
    apply_editorial_style,
    clean_axis,
    figure_size,
    log_tick_labels,
    panel_label,
    power_text,
    save_figure,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "paper" / "source_data" / "fig1_endpoint_nonidentifiability"
SCORE_DIR = ROOT / "paper" / "source_data" / "fig_program_aware_score_and_robustness"
AUDIT_DIR = ROOT / "paper" / "source_data" / "fig3_exact_formula_audit"
PAPER_STEM = ROOT / "paper" / "figures" / "fig_endpoint_nonidentifiability"
NATURE_STEM = ROOT / "Nature" / "figs" / "fig1_schematic"
CANDIDATE_DIR = ROOT / "paper" / "figures" / "candidates" / "main"

CASE_ID = "O0G1B1D0S0C0_s50205_196"
CASE_RUNS = {"numerical": 1, "direct": 0, "mixed": 2}
MIXED_TINT = "#E4B9A9"

ROUTE_STYLE = {
    "numerical": ("Numerical\nsolve", PALETTE["route_valid"], "Route-valid"),
    "direct": ("Direct\nreference\nreconstruction", PALETTE["claim_invalid"], "Direct reference\nreconstruction"),
    "mixed": ("Mixed\nsolver/reference\nreturn", PALETTE["claim_invalid"], "Mixed solver/\nreference return"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_values() -> tuple[dict[str, int], dict, dict[str, str], list[dict[str, str]]]:
    rows = read_csv(SOURCE_DIR / "route_decomposition.csv")
    counts = {row["component"]: int(row["count"]) for row in rows}
    headline = json.loads((SOURCE_DIR / "headline_numbers.json").read_text(encoding="utf-8"))
    assert counts == {
        "confirmed route-valid": 1037,
        "confirmed route-invalid": 125,
        "route-unresolved": 0,
    }
    assert sum(counts.values()) == 1162
    panel_a = read_csv(SCORE_DIR / "fig2_panel_a.csv")[0]
    families = read_csv(SCORE_DIR / "fig2_panel_c_family.csv")
    assert int(panel_a["direct"]) + int(panel_a["mixed"]) == counts["confirmed route-invalid"]
    assert sum(int(row["confirmed_invalid"]) for row in families) == 125
    return counts, headline, panel_a, families


def load_case() -> dict:
    audit = json.loads((AUDIT_DIR / "near_zero_audit.json").read_text(encoding="utf-8"))
    experiment = audit["experiment_id"]
    labels = {
        (row["case_id"], row["run_idx"]): row["classification"]["label"]
        for row in audit["near_zero_runs"]
    }
    assert labels[(CASE_ID, CASE_RUNS["direct"])] == "DIRECT_EXACT_FORMULA"
    assert labels[(CASE_ID, CASE_RUNS["mixed"])] == "MIXED_SOLVER_WITH_EXACT_HINTS"
    assert (CASE_ID, CASE_RUNS["numerical"]) not in labels
    runs_path = ROOT / "analysis" / "phase2" / experiment / "runs.jsonl"
    verdicts = {}
    with runs_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["case_id"] == CASE_ID:
                verdicts[record["run_idx"]] = record["verdict_summary"]
    rel_l2 = {}
    for route, run_idx in CASE_RUNS.items():
        verdict = verdicts[run_idx]
        assert verdict["case_pass"] is True and verdict["failure_code"] == "NONE"
        rel_l2[route] = float(verdict["rel_l2"])
    case = None
    with (ROOT / "benchmark" / "cases" / Path(audit["cases_path"]).name).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["id"] == CASE_ID:
                case = record
                break
    assert case is not None
    grid = case["case_spec"]["eval_grid"]
    assert grid["type"] == "structured_polar"
    spec = grid["spec"]
    reference = np.load(ROOT / "benchmark" / "reference" / "cases" / f"{CASE_ID}.npz")["solution"]
    field = reference.reshape(int(spec["n_radial"]), int(spec["n_angular"]))
    returned_npz = np.load(SOURCE_DIR / f"returned_fields_{CASE_ID}.npz")
    returned = {
        "numerical": returned_npz["numerical_run1"],
        "direct": returned_npz["direct_run0"],
        "mixed": returned_npz["mixed_run2"],
    }
    for route, values in returned.items():
        check = float(np.linalg.norm(values - reference) / np.linalg.norm(reference))
        assert abs(check - rel_l2[route]) <= 1e-6 * max(rel_l2[route], 1e-12), (route, check, rel_l2[route])
    returned = {k: v.reshape(int(spec["n_radial"]), int(spec["n_angular"])) for k, v in returned.items()}
    return {
        "returned": returned,
        "family": case["case_spec"]["pde"]["operator_family"],
        "geometry": case["case_spec"]["domain"]["spec"]["shape"],
        "bc": case["case_spec"]["bc"]["kind"],
        "tau_acc": float(case["thresholds"]["tau_acc"]),
        "r": np.linspace(float(spec["r_in"]), float(spec["r_out"]), int(spec["n_radial"])),
        "theta": np.linspace(0.0, 2.0 * np.pi, int(spec["n_angular"]), endpoint=False),
        "field": field,
        "rel_l2": rel_l2,
        "n_points": int(reference.size),
    }


def box(ax, xy, width, height, text, *, face, edge, text_color=None, weight="normal", size=7.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=size,
        color=text_color or PALETTE["text"],
        fontweight=weight,
        linespacing=1.15,
    )


def arrow(ax, start, end, *, color=None, width=0.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=width,
            color=color or PALETTE["secondary"],
            transform=ax.transAxes,
            shrinkA=1.5,
            shrinkB=1.5,
            clip_on=False,
        )
    )


def sci(value: float) -> tuple[str, str]:
    """Split a value into ('m.m×10', 'e') parts for power_text."""
    assert value > 0.0
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10**exponent
    return f"{mantissa:.1f}×10", f"{exponent}".replace("-", "\u2212")


def panel_routes(ax, fig, case):
    """Fig. 1a: reference field on the left; one row per route with returned field, pointwise error and relative L2 error.
    """
    from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize
    ax.set_axis_off()
    ink, muted, rule = PALETTE["text"], PALETTE["secondary"], PALETTE["grid"]
    ax.set_title("Three passing programs on one task", loc="left", pad=4, x=0.0)
    pos = ax.get_position()
    W, H = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()

    def sub(x0, y0, w, h, **kw):
        return fig.add_axes([pos.x0 + pos.width * x0, pos.y0 + pos.height * y0, pos.width * w, pos.height * h], **kw)

    theta = np.append(case["theta"], case["theta"][0] + 2 * np.pi)
    vmax = float(np.max(np.abs(case["field"])))
    fnorm = Normalize(-vmax, vmax)
    enorm = LogNorm(vmin=1e-18, vmax=1e-3)
    ecmap = LinearSegmentedColormap.from_list("err", ["#E4E8EC", "#8E9AA6", "#2F3B46"])
    d_in = 0.53
    rw, rh = d_in / W, d_in / H

    def ring(xc, yc, values, **kw):
        a_ = sub(xc - rw / 2, yc - rh / 2, rw, rh, projection="polar")
        vals = np.concatenate([values, values[:, :1]], axis=1)
        a_.pcolormesh(theta, case["r"], vals, shading="gouraud", rasterized=True, **kw)
        a_.set_xticks([]); a_.set_yticks([]); a_.set_ylim(0, case["r"][-1]); a_.spines["polar"].set_visible(False); a_.set_facecolor("none")

    def key(x0, y0, w, title_, cmap, norm, ticks, labels):
        cax = sub(x0, y0, w, 0.018)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks(ticks); cb.set_ticklabels(labels)
        cb.ax.minorticks_off(); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.0, length=0, pad=2)
        tl = cb.ax.get_xticklabels(); tl[0].set_horizontalalignment("left"); tl[-1].set_horizontalalignment("right")
        ax.text(x0, y0 + 0.03, title_, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=muted)

    x_ref, x_div, x_lab = 0.08, 0.18, 0.215
    x_field = 0.455 + rw / 2
    x_err = x_field + rw + 0.055
    x_val = x_err + rw / 2 + 0.045
    w_bar = 0.995 - x_val
    rows = {"numerical": 0.715, "direct": 0.44, "mixed": 0.165}

    for x, lab in ((x_field, "Returned field"), (x_err, "Pointwise error")):
        ax.text(x, 0.885, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=muted)
    ax.text(x_val, 0.885, "Relative $\\mathregular{L^2}$ error", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=muted)

    ring(x_ref, rows["numerical"], case["field"], cmap="RdBu_r", norm=fnorm)
    ax.text(x_ref, rows["numerical"] + rh / 2 + 0.03, "Reference", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=ink)
    key(0.005, 0.405, 0.15, "Field $\\mathregular{u}$", "RdBu_r", fnorm, [-vmax, vmax], [f"{-vmax:.2f}".replace("-", "\u2212"), f"{vmax:.2f}"])
    key(0.005, 0.13, 0.15, "Error $\\mathregular{|u_h-u|}$", ecmap, enorm, [1e-17, 1e-4], [r"$\mathregular{10^{-17}}$", r"$\mathregular{10^{-4}}$"])
    ax.plot([x_div, x_div], [0.03, 0.86], transform=ax.transAxes, color=rule, lw=0.6, clip_on=False)
    for y in (0.5775, 0.3025):
        ax.plot([x_lab, 1.0], [y, y], transform=ax.transAxes, color=rule, lw=0.6, clip_on=False)

    names = {"numerical": "Numerical solve", "direct": "Direct\nreconstruction", "mixed": "Mixed return"}
    verdicts = {"numerical": "route-valid", "direct": "route-invalid", "mixed": "route-invalid"}
    for route, yc in rows.items():
        color = ROUTE_STYLE[route][1]
        ax.text(x_lab, yc + 0.012, names[route], transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=color, linespacing=1.1)
        ax.text(x_lab, yc - 0.012, verdicts[route], transform=ax.transAxes, ha="left", va="top", fontsize=7.0, color=muted)
        ret = case["returned"][route]
        ring(x_field, yc, ret, cmap="RdBu_r", norm=fnorm)
        ring(x_err, yc, np.clip(np.abs(ret - case["field"]), 1e-18, None), cmap=ecmap, norm=enorm)
        value = case["rel_l2"][route]
        e_ = int(np.floor(np.log10(value))); m_ = value / 10**e_
        ax.text(x_val, yc + 0.012, rf"$\mathregular{{{m_:.1f}\times10^{{{e_}}}}}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=color)
        bar = sub(x_val, yc - 0.045, w_bar, 0.03)
        bar.set_xscale("log"); bar.set_xlim(1e-17, 1.0); bar.set_ylim(0, 1)
        bar.axvspan(1e-17, case["tau_acc"], color="#ECEFF1", lw=0)
        bar.axvline(case["tau_acc"], color=ink, lw=0.6)
        bar.scatter([value], [0.5], s=12, color=color, zorder=3, edgecolor="white", linewidth=0.3)
        bar.set_axis_off()
    tx = x_val + w_bar * (np.log10(case["tau_acc"]) + 17) / 17
    ax.text(tx, rows["mixed"] - 0.055, r"$\tau_{\mathrm{acc}}$", transform=ax.transAxes, ha="center", va="top", fontsize=7.0, color=ink)


def panel_decomposition(ax, counts, panel_a):
    valid = counts["confirmed route-valid"]
    direct = int(panel_a["direct"])
    mixed = int(panel_a["mixed"])
    total = valid + direct + mixed
    assert total == 1162
    ax.set_xlim(0, total)
    ax.set_ylim(-0.9, 1.0)
    ax.set_title(f"Routes of the {total:,} endpoint passes", loc="left", pad=4)
    ax.barh([0], [valid], height=0.42, color=PALETTE["route_valid"], edgecolor="none")
    ax.barh([0], [direct], left=[valid], height=0.42, color=PALETTE["claim_invalid"], edgecolor="none")
    ax.barh([0], [mixed], left=[valid + direct], height=0.42, color=MIXED_TINT, edgecolor="none")
    ax.text(0, 0.36, f"Numerical {valid:,}", ha="left", va="bottom", fontsize=7.0, color=PALETTE["route_valid"])
    ax.text(total, 0.36, f"Direct {direct}   Mixed {mixed}", ha="right", va="bottom", fontsize=7.0, color=PALETTE["claim_invalid"])
    ax.set_axis_off()


def panel_family(ax, families):
    display = {
        "wave": "Wave",
        "Burgers": "Burgers",
        "heat": "Heat",
        "Helmholtz_low_k": "Helmholtz",
        "Poisson": "Poisson",
        "linear_elasticity": "Elasticity",
    }
    ordered = sorted(families, key=lambda row: float(row["fraction_endpoint_passes_invalid"]), reverse=True)
    y = np.arange(len(ordered))[::-1]
    direct = np.array([100 * int(row["direct"]) / int(row["endpoint_pass"]) for row in ordered])
    mixed = np.array([100 * int(row["mixed"]) / int(row["endpoint_pass"]) for row in ordered])
    total = np.array([100 * float(row["fraction_endpoint_passes_invalid"]) for row in ordered])
    assert np.allclose(direct + mixed, total)
    ax.barh(y, direct, height=0.5, color=PALETTE["claim_invalid"], edgecolor="none", label="Direct")
    ax.barh(y, mixed, left=direct, height=0.5, color=MIXED_TINT, edgecolor="none", label="Mixed")
    for yy, value, row in zip(y, total, ordered, strict=True):
        ax.text(value + 0.8, yy, f"{int(row['confirmed_invalid'])}/{int(row['endpoint_pass'])}", ha="left", va="center", fontsize=7.0, color=PALETTE["text"])
    ax.set_yticks(y, [display[row["family"]] for row in ordered])
    ax.set_xlim(0, 44)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Direct or mixed share of endpoint passes (%)")
    ax.set_title("Share by PDE family", loc="left", pad=4)
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=7.0, handlelength=1.0, handleheight=0.7, labelspacing=0.3)
    ax.tick_params(axis="y", length=0)


def build_figure(layout: str = "A"):
    counts, _, panel_a, families = load_values()
    case = load_case()
    fig = plt.figure(figsize=figure_size(95))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[0.16, 1.0], wspace=0.34, hspace=0.30, left=0.035, right=0.985, bottom=0.10, top=0.94)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    panel_routes(ax_a, fig, case)
    panel_decomposition(ax_b, counts, panel_a)
    panel_family(ax_c, families)
    for letter, (x, y) in {"a": (0.008, 0.985), "b": (0.618, 0.985), "c": (0.618, 0.775)}.items():
        fig.text(x, y, letter, fontsize=8.0, fontweight="bold", color=PALETTE["text"], va="top")
    return fig


def render(*, candidates: bool = True) -> None:
    apply_editorial_style()
    if candidates:
        CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
        fig = build_figure("A")
        fig.savefig(CANDIDATE_DIR / "fig1_candidate_A.png", dpi=110)
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
