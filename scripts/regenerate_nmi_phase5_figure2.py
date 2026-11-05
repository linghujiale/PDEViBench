#!/usr/bin/env python3
"""Route census of the primary panel and the macros used in the Results."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_EXPERIMENT = "phase2_wave1_codex_scipy_20260522T030958Z"
DEFAULT_LEDGER = ROOT / "analysis/nmi_phase5_20260813/final_postreview_route_ledger.csv"
DEFAULT_TRIGGER = ROOT / f"analysis/phase2/{PRIMARY_EXPERIMENT}/near_zero_audit.json"
DEFAULT_OLD = ROOT / "paper/source_data/fig_program_aware_score_and_robustness/program_aware_score_impact.json"
DEFAULT_OUT = ROOT / "analysis/nmi_phase5_20260813/figure2"
DEFAULT_SOURCE = ROOT / "paper/source_data/fig_program_aware_score_and_robustness"
DEFAULT_FIGURE = ROOT / "paper/figures/fig_program_aware_score_impact"
DEFAULT_NATURE = ROOT / "Nature/figs/fig_program_aware_score_impact"
DEFAULT_MACROS = ROOT / "Nature/generated/nmi_phase5_figure2_macros.tex"
INVALID_ROUTES = {"direct_recovery", "mixed_route"}
PREDICTORS = ["O", "G", "B", "D", "S"]

PALETTE = {
    "endpoint": "#7884B4",
    "route_valid": "#484878",
    "direct": "#E9A6A1",
    "mixed": "#E4CCD8",
    "solver": "#B4C0E4",
    "accent": "#B64342",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
    }
)


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def fit_case_models(case_rows: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, Any]:
    x = np.column_stack(
        [np.ones(len(case_rows))]
        + [case_rows[name].to_numpy(dtype=float) for name in PREDICTORS]
    )
    endpoint = case_rows["endpoint_pass_fraction"].to_numpy(dtype=float)
    route_valid = case_rows["route_valid_pass_fraction"].to_numpy(dtype=float)
    endpoint_coef = np.linalg.lstsq(x, endpoint, rcond=None)[0]
    route_coef = np.linalg.lstsq(x, route_valid, rcond=None)[0]

    rng = np.random.default_rng(seed)
    boot_endpoint = np.empty((n_bootstrap, x.shape[1]))
    boot_route = np.empty_like(boot_endpoint)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(case_rows), size=len(case_rows))
        boot_endpoint[index] = np.linalg.lstsq(x[sample], endpoint[sample], rcond=None)[0]
        boot_route[index] = np.linalg.lstsq(x[sample], route_valid[sample], rcond=None)[0]

    names = ["intercept", *PREDICTORS]
    return {
        "estimand": "per-case pass fraction; ordinary least squares; 10,000 case bootstraps",
        "n_cases": int(len(case_rows)),
        "n_bootstrap": int(n_bootstrap),
        "resampling_unit": "base case",
        "endpoint": {
            name: {
                "estimate": float(endpoint_coef[pos]),
                "ci95": percentile_interval(boot_endpoint[:, pos]),
            }
            for pos, name in enumerate(names)
        },
        "route_valid": {
            name: {
                "estimate": float(route_coef[pos]),
                "ci95": percentile_interval(boot_route[:, pos]),
            }
            for pos, name in enumerate(names)
        },
        "route_valid_minus_endpoint": {
            name: {
                "estimate": float(route_coef[pos] - endpoint_coef[pos]),
                "ci95": percentile_interval(boot_route[:, pos] - boot_endpoint[:, pos]),
            }
            for pos, name in enumerate(names)
        },
    }


def grouped_summary(primary: pd.DataFrame, column_names: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouper: str | list[str] = column_names[0] if len(column_names) == 1 else column_names
    for values, group in primary.groupby(grouper, sort=True, dropna=False):
        values_tuple = values if isinstance(values, tuple) else (values,)
        endpoint = group[group["endpoint_pass"].eq(1)]
        invalid = endpoint[endpoint["route_status"].isin(INVALID_ROUTES)]
        valid = endpoint[endpoint["route_status"].eq("route_valid")]
        row: dict[str, Any] = {
            name: value for name, value in zip(column_names, values_tuple, strict=True)
        }
        row.update(
            {
                "scheduled": int(len(group)),
                "endpoint_pass": int(len(endpoint)),
                "route_valid_pass": int(len(valid)),
                "confirmed_invalid": int(len(invalid)),
                "direct": int(invalid["route_status"].eq("direct_recovery").sum()),
                "mixed": int(invalid["route_status"].eq("mixed_route").sum()),
                "endpoint_pass_rate": float(len(endpoint) / len(group)),
                "route_valid_pass_rate": float(len(valid) / len(group)),
                "absolute_inflation": float(len(invalid) / len(group)),
                "fraction_endpoint_passes_invalid": float(len(invalid) / len(endpoint)) if len(endpoint) else 0.0,
            }
        )
        records.append(row)
    return records


def analyse(args: argparse.Namespace) -> dict[str, Any]:
    ledger = pd.read_csv(args.ledger)
    primary = ledger[ledger["experiment_id"].eq(PRIMARY_EXPERIMENT)].copy()
    if len(primary) != 1450:
        raise ValueError(f"expected 1,450 primary records, found {len(primary)}")
    if primary.duplicated(["case_id", "run_idx"]).any():
        raise ValueError("primary ledger contains duplicate case/run records")
    if set(primary.loc[primary["endpoint_pass"].eq(1), "route_status"]) - ({"route_valid"} | INVALID_ROUTES):
        raise ValueError("final primary endpoint passes contain unresolved or unknown route labels")

    endpoint = primary[primary["endpoint_pass"].eq(1)]
    route_valid = endpoint[endpoint["route_status"].eq("route_valid")]
    invalid = endpoint[endpoint["route_status"].isin(INVALID_ROUTES)]
    if (len(endpoint), len(route_valid), len(invalid)) != (1162, 1037, 125):
        raise ValueError(
            "final primary conservation differs from expected 1,162/1,037/125: "
            f"{len(endpoint)}/{len(route_valid)}/{len(invalid)}"
        )
    if len(endpoint) != len(route_valid) + len(invalid):
        raise ValueError("primary endpoint-pass route classes do not conserve")

    trigger = json.loads(args.trigger.read_text(encoding="utf-8"))
    trigger_counts = Counter(
        str(row["classification"]["label"]) for row in trigger["near_zero_runs"]
    )
    expected_trigger = {
        "DIRECT_EXACT_FORMULA": 120,
        "MIXED_SOLVER_WITH_EXACT_HINTS": 5,
    }
    if len(trigger["near_zero_runs"]) != 125 or dict(trigger_counts) != expected_trigger:
        raise ValueError(f"trigger bank differs from 120/5/0: {dict(trigger_counts)}")

    family = grouped_summary(primary, ["family"])
    ogb = grouped_summary(primary, ["O", "G", "B"])
    if sum(row["confirmed_invalid"] for row in family) != len(invalid):
        raise ValueError("PDE-family invalid counts do not sum to the primary invalid count")
    if sum(row["confirmed_invalid"] for row in ogb) != len(invalid):
        raise ValueError("O/G/B invalid counts do not sum to the primary invalid count")

    grouped = primary.groupby("case_id", sort=True)
    case_rows = grouped.agg(
        scheduled=("case_id", "size"),
        endpoint_pass=("endpoint_pass", "sum"),
        O=("O", "first"),
        G=("G", "first"),
        B=("B", "first"),
        D=("D", "first"),
        S=("S", "first"),
    ).reset_index()
    valid_by_case = (
        primary.assign(route_valid_pass=primary["route_status"].eq("route_valid").astype(int))
        .groupby("case_id", sort=True)["route_valid_pass"]
        .sum()
    )
    case_rows["route_valid_pass"] = case_rows["case_id"].map(valid_by_case).astype(int)
    case_rows["endpoint_pass_fraction"] = case_rows["endpoint_pass"] / case_rows["scheduled"]
    case_rows["route_valid_pass_fraction"] = case_rows["route_valid_pass"] / case_rows["scheduled"]
    if len(case_rows) != 290 or not case_rows["scheduled"].eq(5).all():
        raise ValueError("primary panel is not the expected 290 cases x 5 repeats")

    models = fit_case_models(case_rows, args.n_bootstrap, args.seed)
    old = json.loads(args.old.read_text(encoding="utf-8"))
    old_models = old["pass_fraction_models"]
    old_operator = {
        "endpoint": float(old_models["endpoint_only"]["O"]["estimate"]),
        "route_corrected": float(old_models["program_aware"]["O"]["estimate"]),
        "difference": float(old_models["program_aware_minus_endpoint"]["O"]["estimate"]),
        "difference_ci95": [float(value) for value in old_models["program_aware_minus_endpoint"]["O"]["ci95"]],
    }

    headline = {
        "scheduled": int(len(primary)),
        "endpoint_passes": int(len(endpoint)),
        "route_valid_passes": int(len(route_valid)),
        "confirmed_invalid": int(len(invalid)),
        "direct": int(invalid["route_status"].eq("direct_recovery").sum()),
        "mixed": int(invalid["route_status"].eq("mixed_route").sum()),
        "unresolved": 0,
        "endpoint_pass_rate": float(len(endpoint) / len(primary)),
        "route_valid_pass_rate": float(len(route_valid) / len(primary)),
        "absolute_difference": float(len(invalid) / len(primary)),
    }
    return {
        "schema_version": 2,
        "experiment_id": PRIMARY_EXPERIMENT,
        "authority": str(args.ledger.relative_to(ROOT)),
        "estimand_note": (
            "Panels a, c and d and the factor-effect models use the final postreview census of all "
            "1,450 primary records. Panel b alone reports the independently selected 125-item "
            f"near-zero trigger-selected bank; it is not a census of the final {len(invalid)} claim-invalid routes."
        ),
        "headline": headline,
        "trigger_selected_near_zero_bank": {
            "selection_label": "Trigger-selected near-zero output-level passes (n = 125)",
            "n": 125,
            "counts": dict(trigger_counts),
        },
        "by_family": family,
        "by_ogb_cell": ogb,
        "per_case": case_rows.to_dict(orient="records"),
        "factor_models": models,
        "pre_expanded_review_operator_comparison": old_operator,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def macro_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    args.source.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n"
    (args.output / "route_valid_score_impact.json").write_text(serialized, encoding="utf-8")
    (args.source / "route_valid_score_impact.json").write_text(serialized, encoding="utf-8")

    h = payload["headline"]
    family = payload["by_family"]
    ogb = payload["by_ogb_cell"]
    model = payload["factor_models"]
    op_endpoint = model["endpoint"]["O"]
    op_route = model["route_valid"]["O"]
    op_diff = model["route_valid_minus_endpoint"]["O"]
    old = payload["pre_expanded_review_operator_comparison"]
    tb = payload["trigger_selected_near_zero_bank"]["counts"]


    write_csv(args.source / "fig2_panel_a.csv", [h])
    write_csv(
        args.source / "fig2_panel_b_trigger_bank.csv",
        [
            {"route_class": "direct recovery", "count": tb.get("DIRECT_EXACT_FORMULA", 0)},
            {"route_class": "mixed solver/reference", "count": tb.get("MIXED_SOLVER_WITH_EXACT_HINTS", 0)},
            {"route_class": "numerical solver", "count": tb.get("NUMERICAL_SOLVER_NEAR_ZERO", 0)},
        ],
    )
    write_csv(args.source / "fig2_panel_c_family.csv", family)
    write_csv(args.source / "fig2_panel_d_ogb.csv", ogb)
    factor_rows = []
    for estimand in ("endpoint", "route_valid", "route_valid_minus_endpoint"):
        for term, stats in model[estimand].items():
            factor_rows.append(
                {
                    "estimand": estimand,
                    "term": term,
                    "estimate": stats["estimate"],
                    "ci95_low": stats["ci95"][0],
                    "ci95_high": stats["ci95"][1],
                    "n_cases": model["n_cases"],
                    "n_bootstrap": model["n_bootstrap"],
                    "resampling_unit": model["resampling_unit"],
                }
            )
    write_csv(args.source / "fig2_factor_effects.csv", factor_rows)

    family_map = {row["family"]: row for row in family}
    macros = [
        "% Auto-generated by scripts/regenerate_nmi_phase5_figure2.py",
        f"\\newcommand{{\\NMIFigTwoEndpointOperatorCoefficient}}{{{macro_number(op_endpoint['estimate'])}}}",
        f"\\newcommand{{\\NMIFigTwoRouteValidOperatorCoefficient}}{{{macro_number(op_route['estimate'])}}}",
        f"\\newcommand{{\\NMIFigTwoOperatorDifference}}{{{macro_number(op_diff['estimate'])}}}",
        f"\\newcommand{{\\NMIFigTwoOperatorDifferenceCILow}}{{{macro_number(op_diff['ci95'][0])}}}",
        f"\\newcommand{{\\NMIFigTwoOperatorDifferenceCIHigh}}{{{macro_number(op_diff['ci95'][1])}}}",
        f"\\newcommand{{\\NMIFigTwoWaveInvalidCount}}{{{family_map['wave']['confirmed_invalid']}}}",
        f"\\newcommand{{\\NMIFigTwoWaveEndpointPassCount}}{{{family_map['wave']['endpoint_pass']}}}",
        f"\\newcommand{{\\NMIFigTwoBurgersInvalidCount}}{{{family_map['Burgers']['confirmed_invalid']}}}",
        f"\\newcommand{{\\NMIFigTwoBurgersEndpointPassCount}}{{{family_map['Burgers']['endpoint_pass']}}}",
        f"\\newcommand{{\\NMIFigTwoElasticityInvalidCount}}{{{family_map['linear_elasticity']['confirmed_invalid']}}}",
        f"\\newcommand{{\\NMIFigTwoElasticityEndpointPassCount}}{{{family_map['linear_elasticity']['endpoint_pass']}}}",
    ]
    args.macros.parent.mkdir(parents=True, exist_ok=True)
    args.macros.write_text("\n".join(macros) + "\n", encoding="utf-8")


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")


def plot(payload: dict[str, Any], args: argparse.Namespace) -> None:
    h = payload["headline"]
    fig = plt.figure(figsize=(183 / 25.4, 118 / 25.4))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.25, 1.05, 1.2],
        height_ratios=[1.0, 1.0],
        wspace=0.58,
        hspace=0.66,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 1:])

    rates = [h["endpoint_pass_rate"], h["route_valid_pass_rate"]]
    counts = [h["endpoint_passes"], h["route_valid_passes"]]
    bars = ax_a.barh([1, 0], rates, color=[PALETTE["endpoint"], PALETTE["route_valid"]], height=0.52)
    for bar, count, rate in zip(bars, counts, rates, strict=True):
        ax_a.text(
            rate - 0.014,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}/1,450\n{100*rate:.2f}%",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=7,
        )
    ax_a.annotate(
        f"−{h['confirmed_invalid']} confirmed invalid\n−{100*h['absolute_difference']:.1f} points",
        xy=(h["route_valid_pass_rate"], 0.24),
        xytext=(0.37, 0.58),
        arrowprops={"arrowstyle": "->", "color": PALETTE["accent"], "lw": 1.0},
        color=PALETTE["accent"],
        fontsize=7,
        ha="center",
    )
    ax_a.set_yticks([1, 0], ["Output-level\ncomposite", "Route-valid\ncomposite"])
    ax_a.set_xlim(0, 0.9)
    ax_a.set_xlabel("Scheduled-call rate")
    ax_a.set_title("Route check changes the score", loc="left", fontweight="bold", fontsize=8)
    panel_label(ax_a, "a")

    labels = ["Direct\nrecovery", "Mixed\nroute", "Numerical\nsolver"]
    tb = payload["trigger_selected_near_zero_bank"]["counts"]
    values = [tb.get("DIRECT_EXACT_FORMULA", 0), tb.get("MIXED_SOLVER_WITH_EXACT_HINTS", 0), tb.get("NUMERICAL_SOLVER_NEAR_ZERO", 0)]
    bars = ax_b.bar(labels, values, color=[PALETTE["direct"], PALETTE["mixed"], PALETTE["solver"]], edgecolor="#4D4D4D", linewidth=0.5)
    for bar, value in zip(bars, values, strict=True):
        ax_b.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_b.set_ylim(0, 132)
    ax_b.set_ylabel("Programs")
    ax_b.set_title("Trigger-selected near-zero\noutput-level passes (n = 125)", loc="left", fontweight="bold", fontsize=8)
    panel_label(ax_b, "b")

    family_rows = sorted(payload["by_family"], key=lambda row: row["fraction_endpoint_passes_invalid"])
    values = [100 * row["fraction_endpoint_passes_invalid"] for row in family_rows]
    ypos = np.arange(len(family_rows))
    ax_c.barh(ypos, values, color=PALETTE["direct"], edgecolor="#4D4D4D", linewidth=0.4)
    names = {"Helmholtz_low_k": "Helmholtz", "linear_elasticity": "elasticity"}
    ax_c.set_yticks(ypos, [names.get(row["family"], row["family"]) for row in family_rows])
    for pos, row, value in zip(ypos, family_rows, values, strict=True):
        ax_c.text(value + 0.5, pos, f"{row['confirmed_invalid']}/{row['endpoint_pass']}", va="center", fontsize=6.3)
    ax_c.set_xlim(0, max(values) * 1.35)
    ax_c.set_xlabel("Output-level passes invalid (%)")
    ax_c.set_title("Route mixture differs by PDE family", loc="left", fontweight="bold", fontsize=8)
    panel_label(ax_c, "c")

    cells = {(int(row["O"]), int(row["G"]), int(row["B"])): row for row in payload["by_ogb_cell"]}
    matrix = np.zeros((2, 4))
    counts_matrix = np.zeros((2, 4), dtype=int)
    ordered_gb = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for oi in range(2):
        for pos, (gi, bi) in enumerate(ordered_gb):
            matrix[oi, pos] = 100 * cells[(oi, gi, bi)]["absolute_inflation"]
            counts_matrix[oi, pos] = cells[(oi, gi, bi)]["confirmed_invalid"]
    image = ax_d.imshow(matrix, cmap="Reds", vmin=0, vmax=max(1.0, float(matrix.max())), aspect="auto")
    for row in range(2):
        for col in range(4):
            color = "white" if matrix[row, col] > 0.55 * matrix.max() else "#272727"
            ax_d.text(col, row, f"{matrix[row,col]:.1f}\n(n={counts_matrix[row,col]})", ha="center", va="center", color=color, fontsize=6.5, fontweight="bold")
    ax_d.set_xticks(range(4), ["G0 B0", "G0 B1", "G1 B0", "G1 B1"])
    ax_d.set_yticks(range(2), ["O0", "O1"])
    ax_d.set_title("Confirmed inflation by O/G/B cell (percentage points)", loc="left", fontweight="bold", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.035, pad=0.03)
    colorbar.ax.tick_params(labelsize=6)
    panel_label(ax_d, "d")

    fig.suptitle("Route mixture inflates output-level scores non-uniformly", x=0.01, y=0.995, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.115, right=0.965, bottom=0.12, top=0.89)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.nature.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None}
    fig.savefig(args.figure.with_suffix(".svg"), bbox_inches="tight", metadata=metadata)
    fig.savefig(
        args.figure.with_suffix(".pdf"),
        dpi=300,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(args.figure.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(args.figure.with_suffix(".png"), dpi=300, bbox_inches="tight")
    svg = args.figure.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        shutil.copy2(args.figure.with_suffix(suffix), args.nature.with_suffix(suffix))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--trigger", type=Path, default=DEFAULT_TRIGGER)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--nature", type=Path, default=DEFAULT_NATURE)
    parser.add_argument("--macros", type=Path, default=DEFAULT_MACROS)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    result = analyse(cli_args)
    write_outputs(result, cli_args)
    plot(result, cli_args)
    print(json.dumps(result["headline"], indent=2, sort_keys=True))
