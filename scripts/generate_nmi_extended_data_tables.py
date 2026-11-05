#!/usr/bin/env python3
"""Write the tables and macros of the naturalistic panel, the guide experiments and the route test.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "paper" / "source_data" / "fig_external_naturalistic_validation"
PRESCRIPTION_DIR = ROOT / "paper" / "source_data" / "edfig_prescription_code_separation"
OUTPUT = ROOT / "Nature" / "generated" / "nmi_extended_data_tables.tex"

AGENT_LABELS = {
    "Codex GPT-5.5": "Codex",
    "Claude Opus 4.7": "Claude Code",
}
FAMILY_LABELS = {
    "heterogeneous_diffusion": "Heterogeneous diffusion",
    "advection_diffusion_reaction": r"Advection--diffusion--reaction",
    "lshape_poisson": "L-shaped Poisson",
    "coupled_elliptic": "Coupled elliptic",
}
ARM_LABELS = {
    "full": "Full guide",
    "prescription_only": "Prescription-only",
    "choice_free_no_code": "Choice-free",
}
CONTRAST_KEYS = (
    "prescription_only_minus_full_guide",
    "prescription_only_minus_choice_free_no_code",
)
REGIME_EXPERIMENTS = (
    ("Codex GPT-5.2, RH SciPy", "phase2_wave1b_codex_scipy_no_exact_20260523T224200Z"),
    ("Claude Opus 4.7, RH SciPy", "phase2_wave1b_claude_cli_scipy_no_exact_20260526T110547Z"),
    ("Codex GPT-5.2, RH DOLFINx", "phase2_confirm72_codex_dolfinx_no_exact_20260526T022418Z"),
    ("Codex GPT-5.2, RH DOLFINx + v0.9", "phase2_confirm72_codex_dolfinx_v09hints_20260527T015929Z"),
    ("Codex GPT-5.2, formula-blind Poisson", "phase2_formula_blind_poisson48_codex_scipy_20260526T072300Z"),
)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def three(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.3f}"


def interval(metric: dict[str, object]) -> str:
    estimate = float(metric["estimate"])
    lower, upper = (float(value) for value in metric["ci95"])
    return f"{three(estimate)} [{three(lower)}, {three(upper)}]"


def main() -> int:
    external_rows_path = EXTERNAL_DIR / "run_level_source_data.csv"
    external_summary_path = EXTERNAL_DIR / "summary.json"
    prescription_route_path = PRESCRIPTION_DIR / "postreview_route_readout.csv"
    prescription_summary_path = PRESCRIPTION_DIR / "summary.json"
    regime_paths = [
        ROOT / "analysis" / "phase2" / experiment / "wave1_report.json"
        for _, experiment in REGIME_EXPERIMENTS
    ]

    external_rows = csv_rows(external_rows_path)
    external_summary = json.loads(external_summary_path.read_text(encoding="utf-8"))
    route_rows = csv_rows(prescription_route_path)
    prescription = json.loads(prescription_summary_path.read_text(encoding="utf-8"))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in external_rows:
        grouped[(row["agent"], row["family"])].append(row)

    external_lines: list[str] = []
    for agent in AGENT_LABELS:
        for family in FAMILY_LABELS:
            rows = grouped[(agent, family)]
            assert len(rows) == 16
            output_pass = sum(row["endpoint_pass"] == "True" for row in rows)
            route_valid = sum(row["method_pass"] == "True" for row in rows)
            values = [float(row["normalized_rel_l2"]) for row in rows]
            external_lines.append(
                f"{AGENT_LABELS[agent]} & {FAMILY_LABELS[family]} & "
                f"{output_pass}/{len(rows)} & {route_valid}/{len(rows)} & "
                f"{median(values):.4f} & {max(values):.4f} \\\\"
            )
    assert sum(len(rows) for rows in grouped.values()) == 128
    assert all("16/16" in line for line in external_lines)

    convergence = external_summary["reference_convergence"]
    assert convergence["n_cases"] == 32
    assert convergence["all_below_acceptance_limit"] is True
    convergence_line = (
        f"61-to-121-grid reference gap & {int(convergence['n_cases'])} & "
        f"{100 * float(convergence['median_gap_relative_l2']):.3f}\\% & "
        f"{100 * float(convergence['maximum_gap_relative_l2']):.3f}\\% & "
        f"{100 * float(convergence['acceptance_limit_relative_l2']):.1f}\\% \\\\"
    )

    import csv as _csv
    _rt = list(_csv.DictReader((ROOT / "analysis/reaudit_20261030/perturbation_route_test_90_numbers.csv").open()))
    _key = {"full": "pcsep128_fullguide", "prescription": "pcsep128_prescriptionguide", "choice_free": "pcsep128_nocodeguide"}
    arm_lines: list[str] = []
    for row in route_rows:
        k = next((v for a, v in _key.items() if a in row["arm"]), None)
        rs = [r for r in _rt if k and r["experiment_id"].startswith(k)]
        hum_invalid = sum(r["human_consensus"] != "NUMERICAL_SOLVER" for r in rs)
        inst = sum(r["route_test_class_final"] != "follows" for r in rs)
        reviewed = f"{inst}/{len(rs)}" if rs else "--"
        arm_lines.append(
            f"{ARM_LABELS[row['arm']]} & {row['output_level_pass']} & "
            f"{row['route_valid_composite_pass']} & {int(row['route_valid_composite_pass']) - hum_invalid} & {reviewed} \\\\"
        )
    assert sum(int(row["scheduled"]) for row in route_rows) == 384
    assert [int(row["route_valid_composite_pass"]) for row in route_rows] == [80, 80, 77]
    assert all(int(row["confirmed_mixed"]) == 0 for row in route_rows)
    assert all(int(row["scheduled"]) == 128 for row in route_rows)

    test_lines: list[str] = []
    test_labels = (
        r"Prescription-only $-$ full (non-inferiority)",
        r"Prescription-only $-$ choice-free (superiority)",
    )
    for label, test in zip(test_labels, prescription["ordered_confirmatory_tests"], strict=True):
        bounds = test["one_sided_95_interval"]
        margin = -100 * float(test["margin"]) if "noninferiority" in test["name"] else 0.0
        test_lines.append(
            f"{label} & {100 * float(bounds['estimate']):.2f} & "
            f"[{100 * float(bounds['lower']):.2f}, {100 * float(bounds['upper']):.2f}] & "
            f"{margin:.1f} \\\\"
        )
    agreement = prescription["ordered_confirmatory_tests"][0]["paired_table"]
    assert agreement == {
        "both_pass": 80,
        "prescription_only_only": 0,
        "full_guide_only": 0,
        "both_fail": 48,
    }
    agreement_line = (
        "Full versus prescription-only endpoint-pass agreement & "
        r"\multicolumn{3}{l}{80 both pass; 48 both fail; 0 discordant} \\"
    )

    mechanism = prescription["program_mechanism"]["registered_contrasts"]
    secondary = prescription["paired_secondary_deltas"]
    metric_specs = (
        ("log10_grid_points_ratio", r"$\Delta\log_{10}$ grid points", "program"),
        ("log10_time_steps_ratio", r"$\Delta\log_{10}$ time steps", "program"),
        ("log10_rel_l2", r"$\Delta\log_{10}$ relative $L^2$ error", "secondary"),
    )
    secondary_lines: list[str] = []
    for metric_key, label, source in metric_specs:
        cells: list[str] = []
        for contrast in CONTRAST_KEYS:
            metric = (
                mechanism[contrast]["paired_effects"][metric_key]
                if source == "program"
                else secondary[contrast][metric_key]
            )
            cells.append(f"{interval(metric)}; $n={int(metric['n_pairs'])}$")
        secondary_lines.append(f"{label} & {cells[0]} & {cells[1]} \\\\ ")

    traces: list[str] = []
    for contrast in CONTRAST_KEYS:
        trace = mechanism[contrast]["runtime_calibration_trace"]
        numerator = int(trace["traceable_by_program_choice"])
        denominator = int(trace["runtime_comparable_pairs"])
        traces.append(f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}\\%)")
    secondary_lines.append(
        f"Runtime differences with identified program change & {traces[0]} & {traces[1]} \\\\"
    )

    regime_lines: list[str] = []
    expected_totals = [1450, 1450, 216, 216, 144]
    for (label, _), path, expected_total in zip(
        REGIME_EXPERIMENTS, regime_paths, expected_totals, strict=True
    ):
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report.get("summary", report)
        failure_counts = summary.get("failure_code_counts", {}) or {}
        no_verdict_counts = summary.get("no_verdict_classification_counts", {}) or {}
        scheduled = int(
            summary.get("n_records_total", 0)
            or summary.get("expected_total_calls", 0)
        )
        passed = int(failure_counts.get("NONE", 0))
        accuracy = int(failure_counts.get("F-Acc", 0))
        execution = int(failure_counts.get("F-Exec", 0))
        host_mask = int(no_verdict_counts.get("HOST_EVAL_DOMAIN_MASK_MISMATCH", 0))
        other = scheduled - passed - accuracy - execution - host_mask
        assert scheduled == expected_total
        assert other >= 0
        assert passed + accuracy + execution + host_mask + other == scheduled
        regime_lines.append(
            f"{label} & {scheduled:,} & {passed:,} & {accuracy:,} & "
            f"{execution:,} & {host_mask:,} & {other:,} \\\\"
        )

    source_hashes = ", ".join(
        f"{path.name}={digest(path)}"
        for path in (
            external_rows_path,
            external_summary_path,
            prescription_route_path,
            prescription_summary_path,
            *regime_paths,
        )
    )
    content = "\n".join(
        [
            "% Generated by scripts/generate_nmi_extended_data_tables.py; do not edit manually.",
            f"% Frozen source SHA-256: {source_hashes}",
            r"\newcommand{\NMIExternalValidationTableRows}{%",
            *external_lines,
            r"}",
            r"\newcommand{\NMIExternalConvergenceTableRow}{%",
            convergence_line,
            r"}",
            r"\newcommand{\NMIPrescriptionArmTableRows}{%",
            *arm_lines,
            r"}",
            r"\newcommand{\NMIPrescriptionTestTableRows}{%",
            *test_lines,
            r"}",
            r"\newcommand{\NMIPrescriptionAgreementTableRow}{%",
            agreement_line,
            r"}",
            r"\newcommand{\NMIPrescriptionSecondaryTableRows}{%",
            *secondary_lines,
            r"}",
            r"\newcommand{\NMIRegimeOutcomeTableRows}{%",
            *regime_lines,
            r"}",
            "",
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
