"""Data loaders and plotting helpers for the representative-task figure."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge


ROOT = Path(__file__).resolve().parents[3]
NATURE_ASSETS = ROOT / "Nature" / "atlas_assets"
SOURCE_DIR = ROOT / "paper" / "source_data" / "supp_note_representative_case_atlas"
SOURCE_ASSETS = SOURCE_DIR / "assets"
SUMMARY_CSV = SOURCE_DIR / "case_atlas_summary.csv"
SELECTION_MANIFEST = ROOT / "NMI_SUPP_ATLAS_SELECTION_MANIFEST.csv"
NATURE_TEX = ROOT / "Nature" / "representative_case_atlas.tex"
SOURCE_TEX = SOURCE_DIR / "representative_case_atlas.tex"
CASES = ROOT / "benchmark" / "cases" / "pilot_phase1_8.jsonl"
ROUTE_CENSUS = ROOT / "analysis" / "nmi_phase4b_20260811" / "blinded_review_completion" / "final_results_20260812" / "frozen" / "record_route_census_postreview.csv"
AGENTS = {
    "Codex": ROOT / "analysis" / "phase2" / "phase2_wave1b_codex_scipy_no_exact_20260523T224200Z",
    "Claude": ROOT / "analysis" / "phase2" / "phase2_wave1b_claude_cli_scipy_no_exact_20260526T110547Z",
}
AGENT_EXPERIMENT = {agent: directory.name for agent, directory in AGENTS.items()}

ROLE = {
    1: "baseline elliptic discretisation",
    2: "boundary-subset assignment",
    3: "polar-coordinate representation",
    4: "angular-boundary representation",
    5: "non-convex mask interface",
    6: "multiply connected mask interface",
    7: "time stepping on a simple domain",
    8: "time stepping in sector coordinates",
    9: "nonlinear operator with mixed boundary",
    10: "nonlinear polar representation",
    11: "accuracy failure without mask ambiguity",
    12: "vector-valued output structure",
}

PDE_LABEL = {
    "Poisson": "Poisson",
    "heat": "Heat",
    "wave": "Wave",
    "Burgers": "Burgers",
    "Helmholtz_low_k": "Low-wavenumber Helmholtz",
    "linear_elasticity": "Linear elasticity",
}
GEOMETRY_LABEL = {
    "unit_square": "unit square",
    "annulus": "annulus",
    "sector": "sector",
    "L_shape_2d": "L-shaped domain",
    "square_with_hole": "square with hole",
}
BC_LABEL = {
    "dirichlet_homogeneous": "homogeneous Dirichlet",
    "dirichlet_neumann_mixed": "mixed boundary",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 7.6,
        "svg.fonttype": "none",
        "svg.hashsalt": "pdevibench-supp-atlas",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases() -> dict[str, dict]:
    return {case["id"]: case for case in read_jsonl(CASES)}


def shape_of(case: dict) -> str:
    domain = case["case_spec"]["domain"]
    return domain.get("spec", {}).get("shape", domain.get("type", "unknown"))


def bc_of(case: dict) -> str:
    return case["case_spec"]["bc"].get("kind", "unknown")


def finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def select_record(records: list[dict], report_entry: dict) -> dict:
    passes = [
        record
        for record in records
        if record["verdict_summary"].get("case_pass")
        and finite_number(record["verdict_summary"].get("rel_l2")) is not None
    ]
    evaluated = [
        record
        for record in records
        if finite_number(record["verdict_summary"].get("rel_l2")) is not None
    ]
    candidates = passes or evaluated
    if not candidates:
        diagnostic_counts = report_entry.get("no_verdict_classifications", {})
        dominant = (
            max(sorted(diagnostic_counts), key=lambda code: diagnostic_counts[code])
            if diagnostic_counts
            else "NO_VERDICT"
        )
        return {
            "selection_rule": "registered diagnostic state; no evaluated field",
            "candidate_set": "diagnostic records",
            "candidate_count": len(records),
            "median_rel_l2": "",
            "selected_run_idx": "",
            "selected_record_id": "aggregate diagnostic state",
            "selected_verdict": dominant,
            "selected_rel_l2": "",
            "tie_count": "",
        }
    errors = [float(record["verdict_summary"]["rel_l2"]) for record in candidates]
    median = statistics.median(errors)
    distances = [abs(error - median) for error in errors]
    minimum = min(distances)
    tied = [
        record
        for record, distance in zip(candidates, distances)
        if math.isclose(distance, minimum, rel_tol=0.0, abs_tol=1e-18)
    ]
    selected = min(tied, key=lambda record: int(record["run_idx"]))
    verdict = selected["verdict_summary"]
    return {
        "selection_rule": "passing error nearest agent-specific median" if passes else "evaluated error nearest agent-specific median",
        "candidate_set": "output-level passes" if passes else "evaluated non-passes",
        "candidate_count": len(candidates),
        "median_rel_l2": f"{median:.17g}",
        "selected_run_idx": int(selected["run_idx"]),
        "selected_record_id": f"{selected['case_id']}:run{selected['run_idx']}",
        "selected_verdict": "pass" if verdict.get("case_pass") else verdict.get("failure_code", "non-pass"),
        "selected_rel_l2": f"{float(verdict['rel_l2']):.17g}",
        "tie_count": len(tied),
    }


def reshape_reference(array: np.ndarray, case: dict) -> np.ndarray:
    output_shape = case["case_spec"]["output"]["shape"]
    if len(output_shape) == 2 and int(output_shape[1]) in (2, 3):
        values = np.linalg.norm(array.reshape(int(output_shape[0]), int(output_shape[1])), axis=1)
    else:
        values = array.reshape(-1)
    grid = case["case_spec"]["eval_grid"]
    grid_type = grid["type"]
    spec = grid["spec"]
    if grid_type == "structured_polar":
        return values.reshape(int(spec["n_radial"]), int(spec["n_angular"]))
    if grid_type == "structured_polar_sector":
        return values.reshape(int(spec["n_radial"]), int(spec["n_angular"]))
    return values.reshape(int(spec.get("ny", 16)), int(spec.get("nx", 16)))


def load_reference(case: dict) -> np.ndarray:
    with np.load(ROOT / case["reference"]["path"]) as archive:
        return reshape_reference(np.asarray(archive["solution"], dtype=float), case)


def reference_limits(field: np.ndarray) -> tuple[float, float]:
    finite = field[np.isfinite(field)]
    low, high = float(np.min(finite)), float(np.max(finite))
    if math.isclose(low, high):
        delta = abs(low) * 0.05 or 1.0
        return low - delta, high + delta
    return low, high


def plot_outline(ax: plt.Axes, case: dict) -> None:
    shape = shape_of(case)
    spec = case["case_spec"]["domain"].get("spec", {})
    if shape == "annulus":
        for radius in (float(spec["r_in"]), float(spec["r_out"])):
            ax.add_patch(Circle(spec.get("center", [0, 0]), radius, fill=False, ec="#2F3B46", lw=0.55))
    elif shape == "sector":
        ax.add_patch(Wedge(spec.get("center", [0, 0]), float(spec["r_max"]), 0, math.degrees(float(spec["theta_max"])), fill=False, ec="#2F3B46", lw=0.55))
    elif shape == "L_shape_2d":
        ax.add_patch(Polygon(spec["outer"], closed=True, fill=False, ec="#2F3B46", lw=0.65))
    elif shape == "square_with_hole":
        ax.add_patch(Polygon(spec["outer"], closed=True, fill=False, ec="#2F3B46", lw=0.65))
        ax.add_patch(Circle(spec["hole_center"], float(spec["hole_radius"]), fill=False, ec="#2F3B46", lw=0.65))
    else:
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="#2F3B46", lw=0.65))


def plot_reference(ax: plt.Axes, field: np.ndarray, case: dict):
    vmin, vmax = reference_limits(field)
    grid = case["case_spec"]["eval_grid"]
    spec = grid["spec"]
    if grid["type"] in {"structured_polar", "structured_polar_sector"}:
        nr, nt = field.shape
        r0 = float(spec.get("r_in", 0.0))
        r1 = float(spec.get("r_out", spec.get("r_max", 1.0)))
        theta1 = float(spec.get("theta_max", 2 * math.pi))
        r_edges = np.linspace(r0, r1, nr + 1)
        theta_edges = np.linspace(0.0, theta1, nt + 1)
        rr, tt = np.meshgrid(r_edges, theta_edges, indexing="ij")
        xx, yy = rr * np.cos(tt), rr * np.sin(tt)
        image = ax.pcolormesh(xx, yy, np.ma.masked_invalid(field), cmap="viridis", vmin=vmin, vmax=vmax, shading="flat", rasterized=False)
        ax.set_aspect("equal")
        ax.autoscale_view()
    else:
        extent = [float(spec.get("x0", 0.0)), float(spec.get("x1", 1.0)), float(spec.get("y0", 0.0)), float(spec.get("y1", 1.0))]
        image = ax.imshow(np.ma.masked_invalid(field), origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_aspect("equal")
    plot_outline(ax, case)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def card(ax: plt.Axes, title: str, lines: list[str], accent: str = "#747B86") -> None:
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0.02, 0.03),
            0.96,
            0.94,
            transform=ax.transAxes,
            facecolor="#F4F5F7",
            edgecolor="#C7CDD3",
            linewidth=0.7,
        )
    )
    title_lines = title.split("\n")
    for offset, title_line in enumerate(title_lines):
        ax.text(0.07, 0.88 - 0.10 * offset, title_line, transform=ax.transAxes, ha="left", va="top", fontsize=7.6, fontweight="bold", color=accent)
    body_y = 0.61 if len(title_lines) > 1 else 0.64
    ax.text(0.07, body_y, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=7.6, color="#2F3B46", linespacing=1.22)


def fmt_error(value: str) -> str:
    if value == "":
        return "not evaluated"
    number = float(value)
    if number == 0:
        return "0"
    exponent = math.floor(math.log10(abs(number)))
    mantissa = number / (10**exponent)
    superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa:.2f}×10{superscript}"


def compact_route(value: str) -> str:
    labels = {
        "route_valid": "route-valid",
        "claim_invalid": "claim-invalid",
        "deterministic_static_census": "static census",
        "expanded_independent_human_consensus": "human consensus",
        "independent_human_consensus": "human consensus",
        "sparse_fd_linear_solve": "sparse FD solve",
        "coordinate_transform_or_polar_solver": "coordinate/polar solve",
        "time_stepping_solver": "time-stepping solve",
        "not_endpoint_pass": "endpoint not passed",
    }
    return labels.get(value, value.replace("_", " "))


def colourbar_ticks(field: np.ndarray) -> tuple[np.ndarray, list[str]]:
    low, high = reference_limits(field)
    ticks = np.linspace(low, high, 3)
    labels = [f"{value:.2g}".replace("e-", "e−").replace("e+", "e+") for value in ticks]
    return ticks, labels


def verdict_counts(records: list[dict], report_entry: dict) -> str:
    if report_entry.get("n_no_verdict") and not report_entry.get("n_verdicts"):
        diagnostic_counts = report_entry.get("no_verdict_classifications", {})
        if diagnostic_counts:
            code = max(sorted(diagnostic_counts), key=lambda item: diagnostic_counts[item])
            label = "host-mask mismatch" if code == "HOST_EVAL_DOMAIN_MASK_MISMATCH" else code
            return f"{diagnostic_counts[code]}/{report_entry['n_records']} {label}"
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        verdict = record["verdict_summary"]
        key = "pass" if verdict.get("case_pass") else verdict.get("failure_code", "no verdict")
        counts[key] += 1
    if len(counts) == 1:
        key, count = next(iter(counts.items()))
        label = "no verdict" if key == "HOST_EVAL_DOMAIN_MASK_MISMATCH" else key
        return f"{count}/{len(records)} {label}"
    return " · ".join(f"{value} {key}" for key, value in sorted(counts.items()))


def draw_standard(case: dict, field: np.ndarray, selections: dict[str, dict], aggregate: dict[str, str], out_base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.50), gridspec_kw={"wspace": 0.18, "width_ratios": [1.05, 1.0, 1.0]})
    case_index = int(out_base.name.split("_")[-1])
    panel = "abc"[(case_index - 1) % 3]
    header = (
        f"{PDE_LABEL[case['case_spec']['pde']['operator_family']]} · {GEOMETRY_LABEL[shape_of(case)]} · {BC_LABEL[bc_of(case)]}\n"
        "Cell " + "/".join(f"{axis}{case['axes'][axis]}" for axis in ["O", "G", "B", "D", "S", "C"])
    )
    fig.text(0.015, 0.98, panel, ha="left", va="top", fontsize=9.8, fontweight="bold", color="#2F3B46")
    fig.text(0.045, 0.98, header, ha="left", va="top", multialignment="left", fontsize=8.2, linespacing=1.18, color="#2F3B46")
    image = plot_reference(axes[0], field, case)
    operator_family = case["case_spec"]["pde"]["operator_family"]
    reference_title = "Reference displacement magnitude" if operator_family == "linear_elasticity" else "Reference field"
    axes[0].set_title(reference_title, fontsize=7.6, pad=2.5)
    for agent, ax_card, accent in (
        ("Codex", axes[1], "#496A8D"),
        ("Claude", axes[2], "#D39A58"),
    ):
        selected = selections[agent]
        record = selected["selected_record_id"]
        run_label = record.split(":")[-1] if ":" in record else record
        evidence_lines = [
            f"{run_label} · {selected['selected_verdict']}",
            f"rel. L2 {fmt_error(selected['selected_rel_l2'])}",
            compact_route(selected["route_status"]),
        ]
        if selected["route_status"] == "route_valid" and selected["static_route"] != "not_applicable":
            evidence_lines.append(compact_route(selected["static_route"]))
        card(
            ax_card,
            f"{agent}\nrepresentative record",
            evidence_lines,
            accent,
        )
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.048, pad=0.025)
    ticks, labels = colourbar_ticks(field)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=7.6, length=2)
    fig.subplots_adjust(left=0.015, right=0.992, top=0.75, bottom=0.06)
    save_figure(fig, out_base)


def draw_mask(case: dict, field: np.ndarray, selections: dict[str, dict], aggregate: dict[str, str], out_base: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.09, 2.50), gridspec_kw={"wspace": 0.30, "width_ratios": [0.92, 0.92, 1.22, 1.22]})
    case_index = int(out_base.name.split("_")[-1])
    panel = "abc"[(case_index - 1) % 3]
    header = (
        f"{PDE_LABEL[case['case_spec']['pde']['operator_family']]} · {GEOMETRY_LABEL[shape_of(case)]} · {BC_LABEL[bc_of(case)]}\n"
        "Cell " + "/".join(f"{axis}{case['axes'][axis]}" for axis in ["O", "G", "B", "D", "S", "C"])
    )
    fig.text(0.015, 0.98, panel, ha="left", va="top", fontsize=9.8, fontweight="bold", color="#2F3B46")
    fig.text(0.045, 0.98, header, ha="left", va="top", multialignment="left", fontsize=8.2, linespacing=1.18, color="#2F3B46")
    image = plot_reference(axes[0], field, case)
    axes[0].set_title("Reference field", fontsize=7.6, pad=2.5)
    mask = np.isfinite(field).astype(float)
    mask_map = ListedColormap(["#E7EAED", "#69747E"])
    mask_spec = case["case_spec"]["eval_grid"]["spec"]
    mask_extent = [float(mask_spec.get("x0", 0.0)), float(mask_spec.get("x1", 1.0)), float(mask_spec.get("y0", 0.0)), float(mask_spec.get("y1", 1.0))]
    axes[1].imshow(mask, origin="lower", extent=mask_extent, cmap=mask_map, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_aspect("equal")
    plot_outline(axes[1], case)
    axes[1].set_title("Reference-valid mask", fontsize=7.6, pad=2.5)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    for agent, ax_card, accent in (("Codex", axes[2], "#496A8D"), ("Claude", axes[3], "#D39A58")):
        selected = selections[agent]
        card(
            ax_card,
            f"{agent}\nrepresentative record",
            ["endpoint not passed", "host-mask outcome"],
            accent,
        )
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.048, pad=0.025)
    ticks, labels = colourbar_ticks(field)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=7.6, length=2)
    fig.subplots_adjust(left=0.015, right=0.992, top=0.75, bottom=0.06)
    save_figure(fig, out_base)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg")
    fig.savefig(svg_path, facecolor="white", metadata={"Date": None})
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        out_base.with_suffix(".pdf"),
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_base.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def build_tex(summary: list[dict]) -> str:
    captions = [
        r"\textbf{Elliptic boundary and coordinate representations.} (a) Poisson equation on a unit square with Dirichlet data. (b) Poisson equation on a unit square with mixed boundary data. (c) Poisson equation on an annulus with Dirichlet data.",
        r"\textbf{Elliptic domain representations.} (a) Poisson equation on a sector. (b) Poisson equation on an L-shaped domain. (c) Poisson equation on a square-with-hole domain. Panels b and c also show the reference-valid mask.",
        r"\textbf{Time-dependent and nonlinear tasks.} (a) Unit-square heat equation. (b) Sector wave equation. (c) Unit-square Burgers equation with mixed boundary data.",
        r"\textbf{Nonlinear, oscillatory and vector-valued tasks.} (a) Burgers equation on an annulus with mixed boundary data. (b) Helmholtz equation on an annulus. (c) Linear elasticity on a unit square; the reference shows the displacement magnitude.",
    ]
    blocks = [
        r"\clearpage",
        r"\section*{Supplementary Note 7: Representative PDE tasks and program records}",
        "Supplementary Figs. 2--5 show twelve primary-panel tasks, each with one program from the prohibition reruns by Codex (GPT-5.2) and by Claude Code. Each panel gives the PDE, geometry, boundary condition and design cell, the numerical reference and, for each agent, the passing program closest to the agent's median error, or the closest available record when no program passed, with its run index, verdict, relative $L^2$ error, route and solver class. All passing programs shown are numerical solvers: sparse finite-difference solves on the square, polar-coordinate solves on the annulus and sector, and time stepping for the heat, wave and Burgers equations. On the L-shaped and square-with-hole domains both agents end at the mask check (Supplementary Fig. 3b,c), and on the annular Helmholtz task with the sign error of Supplementary Note 8 both fail with relative error 2.0 (Supplementary Fig. 5b).",
    ]
    for group_index, caption in enumerate(captions, 1):
        if group_index > 1:
            blocks.append(r"\clearpage")
        first_case = (group_index - 1) * 3 + 1
        blocks.extend(
            [
                r"\begin{center}",
                rf"\supplementaryfigureheading{{{caption}}}",
                *[
                    rf"\includegraphics[width=\linewidth]{{atlas_assets/atlas_{case_index:02d}.pdf}}"
                    for case_index in range(first_case, first_case + 3)
                ],
                r"\end{center}",
            ]
        )
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    NATURE_ASSETS.mkdir(parents=True, exist_ok=True)
    SOURCE_ASSETS.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    summary = list(csv.DictReader(SUMMARY_CSV.open(encoding="utf-8")))
    runs_by_agent: dict[str, dict[str, list[dict]]] = {}
    report_by_agent: dict[str, dict[str, dict]] = {}
    route_index: dict[tuple[str, str, str], dict] = {}
    with ROUTE_CENSUS.open(newline="", encoding="utf-8") as handle:
        for route_row in csv.DictReader(handle):
            route_index[(route_row["experiment_id"], route_row["case_id"], route_row["run_idx"])] = route_row
    for agent, directory in AGENTS.items():
        path = directory / "runs.jsonl"
        indexed: dict[str, dict[int, dict]] = defaultdict(dict)
        for record in read_jsonl(path):
            indexed[record["case_id"]][int(record["run_idx"])] = record
        runs_by_agent[agent] = {case_id: list(by_run.values()) for case_id, by_run in indexed.items()}
        report = json.loads((directory / "wave1_report.json").read_text(encoding="utf-8"))
        report_by_agent[agent] = {entry["case_id"]: entry for entry in report["per_case"]}

    manifest_rows: list[dict] = []
    for row in summary:
        index = int(row["index"])
        supplementary_figure = (index - 1) // 3 + 1
        panel = "abc"[(index - 1) % 3]
        case_id = row["case_id"]
        case = cases[case_id]
        field = load_reference(case)
        selections: dict[str, dict] = {}
        aggregate: dict[str, str] = {}
        for agent in AGENTS:
            records = sorted(runs_by_agent[agent][case_id], key=lambda record: int(record["run_idx"]))
            report_entry = report_by_agent[agent][case_id]
            selected = select_record(records, report_entry)
            if selected["selected_run_idx"] == "":
                selected.update(route_status="not_endpoint_pass", route_source="not_applicable", static_route="not_applicable")
            else:
                route = route_index[(AGENT_EXPERIMENT[agent], case_id, str(selected["selected_run_idx"]))]
                selected.update(
                    route_status=route["route_status"],
                    route_source=route["route_source"],
                    static_route=route["static_route"] or "not_applicable",
                )
            selections[agent] = selected
            aggregate[agent] = verdict_counts(records, report_entry)
            source_ledger = AGENTS[agent] / "runs.jsonl"
            source_report = AGENTS[agent] / "wave1_report.json"
            manifest_rows.append(
                {
                    "supplementary_figure": supplementary_figure,
                    "panel": panel,
                    "atlas_case_index": index,
                    "case_id": case_id,
                    "agent": agent,
                    **selected,
                    "record_count": len(records),
                    "field_available": "no",
                    "display_state": "registered diagnostic card; selected output array not retained",
                    "source_ledger": str(source_ledger.relative_to(ROOT)),
                    "source_ledger_sha256": sha256(source_ledger),
                    "source_report": str(source_report.relative_to(ROOT)),
                    "source_report_sha256": sha256(source_report),
                    "source_report_n_records": report_entry["n_records"],
                }
            )
        out_base = NATURE_ASSETS / f"atlas_{index:02d}"
        if shape_of(case) in {"L_shape_2d", "square_with_hole"}:
            draw_mask(case, field, selections, aggregate, out_base)
        else:
            draw_standard(case, field, selections, aggregate, out_base)
        for suffix in (".svg", ".pdf", ".png"):
            shutil.copy2(out_base.with_suffix(suffix), SOURCE_ASSETS / out_base.with_suffix(suffix).name)

    with SELECTION_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    shutil.copy2(SELECTION_MANIFEST, SOURCE_DIR / SELECTION_MANIFEST.name)
    nature_tex = build_tex(summary)
    NATURE_TEX.write_text(nature_tex, encoding="utf-8")
    SOURCE_TEX.write_text(nature_tex.replace("atlas_assets/", "assets/"), encoding="utf-8")
    print(f"wrote {len(summary)} atlas figures as SVG/PDF/PNG")
    print(f"wrote {SELECTION_MANIFEST}")
    print(f"wrote {NATURE_TEX}")


if __name__ == "__main__":
    main()
