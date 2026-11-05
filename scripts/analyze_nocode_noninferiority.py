#!/usr/bin/env python3
"""Completion gate and paired non-inferiority analysis for the no-code guide."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.axis_models import (  # noqa: E402
    design_matrix,
    fit_clustered_logistic,
    fit_clustered_ols,
    term_names,
)
from scripts.analyze_failure_inclusive_frontier import (  # noqa: E402
    load_cases,
    load_latest_runs,
    record_class,
    usable_for_pass_model,
)
from scripts.run_phase2_wave1 import is_infra_failure  # noqa: E402

DEFAULT_MANIFEST = ROOT / "benchmark/cases/nocode_noninferiority_128_manifest.json"
DEFAULT_CASES = ROOT / "benchmark/cases/axis_balanced_1024.jsonl"
DEFAULT_OUT = ROOT / "analysis/phase2/nocode_noninferiority_20260801"
MODEL_AXES = ("A", "O", "G", "B", "D", "S", "C")
MODEL_INTERACTIONS = (("A", "O"), ("A", "G"), ("A", "B"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["case_id"]), int(row["run_idx"])): row for row in load_latest_runs(path)}


def wilson_score_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def newcombe_method10_interval(
    *, both_pass: int, no_code_only: int, full_only: int, both_fail: int, z: float
) -> dict[str, float]:
    """Newcombe (1998) paired score interval, method 10.

    The difference is no-code pass probability minus full-guide pass
    probability.  Cell orientation is therefore e=both pass, f=no-code only,
    g=full only, h=both fail in Newcombe's notation.
    """
    e = int(both_pass)
    f = int(no_code_only)
    g = int(full_only)
    h = int(both_fail)
    total = e + f + g + h
    if total <= 0 or min(e, f, g, h) < 0:
        raise ValueError("paired cell counts must be nonnegative with positive total")

    no_code = (e + f) / total
    full = (e + g) / total
    estimate = (f - g) / total
    no_code_lower, no_code_upper = wilson_score_interval(e + f, total, z)
    full_lower, full_upper = wilson_score_interval(e + g, total, z)
    no_code_down = no_code - no_code_lower
    no_code_up = no_code_upper - no_code
    full_down = full - full_lower
    full_up = full_upper - full

    denominator = math.sqrt((e + f) * (g + h) * (e + g) * (f + h))
    numerator = e * h - f * g
    if denominator == 0.0:
        correlation = 0.0
    elif numerator > 0:
        correlation = max(numerator - total / 2.0, 0.0) / denominator
    else:
        correlation = numerator / denominator
    correlation = min(1.0, max(-1.0, correlation))

    lower_distance = math.sqrt(
        max(
            0.0,
            no_code_down**2 - 2.0 * correlation * no_code_down * full_up + full_up**2,
        )
    )
    upper_distance = math.sqrt(
        max(
            0.0,
            no_code_up**2 - 2.0 * correlation * no_code_up * full_down + full_down**2,
        )
    )
    return {
        "estimate": estimate,
        "lower": max(-1.0, estimate - lower_distance),
        "upper": min(1.0, estimate + upper_distance),
        "z": float(z),
        "continuity_corrected_phi": correlation,
    }


def clustered_mean_ci(
    values_by_case: dict[str, list[float]], *, seed: int = 20260801, n_boot: int = 5000
) -> dict[str, Any]:
    case_means = np.asarray(
        [float(np.mean(values)) for _, values in sorted(values_by_case.items())],
        dtype=float,
    )
    if not len(case_means):
        return {"n_cases": 0, "estimate": None, "ci95": None}
    rng = np.random.default_rng(seed)
    samples = rng.choice(case_means, size=(n_boot, len(case_means)), replace=True)
    means = np.mean(samples, axis=1)
    return {
        "n_cases": len(case_means),
        "estimate": float(np.mean(case_means)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_replicates": n_boot,
        "bootstrap_seed": seed,
    }


def paired_numeric_delta(
    pairs: list[dict[str, Any]], extractor: Callable[[dict[str, Any]], float | None]
) -> dict[str, Any]:
    by_case: dict[str, list[float]] = defaultdict(list)
    n_pairs = 0
    for pair in pairs:
        full_value = extractor(pair["full"])
        no_code_value = extractor(pair["no_code"])
        if full_value is None or no_code_value is None:
            continue
        by_case[pair["case_id"]].append(float(no_code_value) - float(full_value))
        n_pairs += 1
    return {"n_pairs": n_pairs, **clustered_mean_ci(by_case)}


def finite_field(record: dict[str, Any], section: str, field: str) -> float | None:
    value = (record.get(section) or {}).get(field)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def analyze(*, manifest_path: Path, cases_path: Path, analysis_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = load_cases(cases_path)
    case_ids = [str(case_id) for case_id in manifest["case_ids"]]
    run_indices = [int(value) for value in manifest["execution"]["run_indices"]]
    expected_keys = {(case_id, run_idx) for case_id in case_ids for run_idx in run_indices}
    arm_specs = manifest["arms"]
    ledgers = {
        arm: analysis_root / str(spec["experiment_id"]) / "runs.jsonl"
        for arm, spec in arm_specs.items()
    }
    records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    arm_status: dict[str, Any] = {}
    for arm, path in ledgers.items():
        latest = latest_map(path) if path.is_file() else {}
        selected = {key: value for key, value in latest.items() if key in expected_keys}
        records[arm] = selected
        arm_status[arm] = {
            "ledger_path": str(path),
            "ledger_present": path.is_file(),
            "ledger_sha256": sha256(path) if path.is_file() else None,
            "latest_selected_records": len(selected),
            "expected_records": len(expected_keys),
            "missing_keys": [list(key) for key in sorted(expected_keys - set(selected))],
            "unresolved_retriable_infrastructure": sum(
                is_infra_failure(record) for record in selected.values()
            ),
            "scientific_or_deterministic_records": sum(
                usable_for_pass_model(record) for record in selected.values()
            ),
            "taxonomy": dict(Counter(record_class(record) for record in selected.values())),
        }

    pairs: list[dict[str, Any]] = []
    for case_id, run_idx in sorted(expected_keys):
        full = records["full_guide"].get((case_id, run_idx))
        no_code = records["no_code_guide"].get((case_id, run_idx))
        if (
            full is None
            or no_code is None
            or not usable_for_pass_model(full)
            or not usable_for_pass_model(no_code)
        ):
            continue
        pairs.append(
            {
                "case_id": case_id,
                "run_idx": run_idx,
                "full": full,
                "no_code": no_code,
                "full_class": record_class(full),
                "no_code_class": record_class(no_code),
                "case_order": case_ids.index(case_id),
            }
        )

    discordance = Counter()
    transition_taxonomy = Counter()
    for pair in pairs:
        full_pass = pair["full_class"] == "pass"
        no_code_pass = pair["no_code_class"] == "pass"
        discordance[(full_pass, no_code_pass)] += 1
        transition_taxonomy[f"{pair['full_class']} -> {pair['no_code_class']}"] += 1

    both_pass = discordance[(True, True)]
    full_only = discordance[(True, False)]
    no_code_only = discordance[(False, True)]
    both_fail = discordance[(False, False)]
    discordant = full_only + no_code_only
    mcnemar_p = (
        float(binomtest(min(full_only, no_code_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    primary_interval = (
        newcombe_method10_interval(
            both_pass=both_pass,
            no_code_only=no_code_only,
            full_only=full_only,
            both_fail=both_fail,
            z=float(norm.ppf(0.95)),
        )
        if pairs
        else None
    )
    descriptive_interval = (
        newcombe_method10_interval(
            both_pass=both_pass,
            no_code_only=no_code_only,
            full_only=full_only,
            both_fail=both_fail,
            z=float(norm.ppf(0.975)),
        )
        if pairs
        else None
    )

    model_cases: list[dict[str, Any]] = []
    model_pass: list[float] = []
    model_clusters: list[str] = []
    error_cases: list[dict[str, Any]] = []
    error_values: list[float] = []
    error_clusters: list[str] = []
    for pair in pairs:
        base_case = cases[pair["case_id"]]
        tau = float(base_case["thresholds"]["tau_acc"])
        for arm_value, arm in ((0, "full"), (1, "no_code")):
            record = pair[arm]
            augmented = {**base_case, "axes": {**base_case["axes"], "A": arm_value}}
            model_cases.append(augmented)
            model_pass.append(1.0 if record_class(record) == "pass" else 0.0)
            model_clusters.append(pair["case_id"])
            rel_l2 = (record.get("verdict_summary") or {}).get("rel_l2")
            if isinstance(rel_l2, (int, float)) and math.isfinite(float(rel_l2)):
                error_cases.append(augmented)
                error_values.append(math.log10(max(float(rel_l2) / tau, 1e-15)))
                error_clusters.append(pair["case_id"])
    names = term_names(axes=MODEL_AXES, interactions=MODEL_INTERACTIONS)
    pass_model = (
        fit_clustered_logistic(
            design_matrix(model_cases, axes=MODEL_AXES, interactions=MODEL_INTERACTIONS),
            np.asarray(model_pass),
            model_clusters,
            names,
        )
        if model_cases
        else {}
    )
    error_model = (
        fit_clustered_ols(
            design_matrix(error_cases, axes=MODEL_AXES, interactions=MODEL_INTERACTIONS),
            np.asarray(error_values),
            error_clusters,
            names,
        )
        if error_cases
        else {}
    )

    def normalized_error(record: dict[str, Any]) -> float | None:
        rel_l2 = (record.get("verdict_summary") or {}).get("rel_l2")
        if not isinstance(rel_l2, (int, float)):
            return None
        case_id = str(record["case_id"])
        return math.log10(
            max(float(rel_l2) / float(cases[case_id]["thresholds"]["tau_acc"]), 1e-15)
        )

    resource_deltas = {
        "log10_normalized_error": paired_numeric_delta(pairs, normalized_error),
        "agent_wall_seconds": paired_numeric_delta(
            pairs, lambda record: finite_field(record, "agent_call", "wall_seconds_total")
        ),
        "input_tokens": paired_numeric_delta(
            pairs, lambda record: finite_field(record, "agent_call", "token_total_in")
        ),
        "output_tokens": paired_numeric_delta(
            pairs, lambda record: finite_field(record, "agent_call", "token_total_out")
        ),
        "tool_calls": paired_numeric_delta(
            pairs, lambda record: finite_field(record, "agent_call", "n_tool_calls")
        ),
    }

    cell_outcomes: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"full": [], "no_code": []}
    )
    for pair in pairs:
        case = cases[pair["case_id"]]
        cell = "".join(f"{axis}{int(case['axes'][axis])}" for axis in "OGBDSC")
        cell_outcomes[cell]["full"].append(pair["full_class"] == "pass")
        cell_outcomes[cell]["no_code"].append(pair["no_code_class"] == "pass")
    cell_summary = {
        cell: {
            "full_pass_fraction": float(np.mean(values["full"])),
            "no_code_pass_fraction": float(np.mean(values["no_code"])),
            "full_majority": bool(float(np.mean(values["full"])) >= 0.5),
            "no_code_majority": bool(float(np.mean(values["no_code"])) >= 0.5),
        }
        for cell, values in sorted(cell_outcomes.items())
    }
    topology_flips = sum(
        row["full_majority"] != row["no_code_majority"] for row in cell_summary.values()
    )

    order_diagnostics = {
        "discordant_case_positions_zero_based": [
            pair["case_order"]
            for pair in pairs
            if (pair["full_class"] == "pass") != (pair["no_code_class"] == "pass")
        ],
        "first_half_transitions": dict(
            Counter(
                f"{pair['full_class']} -> {pair['no_code_class']}"
                for pair in pairs
                if pair["case_order"] < len(case_ids) / 2
            )
        ),
        "second_half_transitions": dict(
            Counter(
                f"{pair['full_class']} -> {pair['no_code_class']}"
                for pair in pairs
                if pair["case_order"] >= len(case_ids) / 2
            )
        ),
    }

    complete = bool(
        len(pairs) == len(expected_keys)
        and all(
            row["latest_selected_records"] == len(expected_keys)
            and row["unresolved_retriable_infrastructure"] == 0
            and row["scientific_or_deterministic_records"] == len(expected_keys)
            for row in arm_status.values()
        )
    )
    margin = float(manifest["primary_estimand"]["noninferiority_margin"])
    lower_bound = primary_interval["lower"] if primary_interval is not None else None
    noninferiority_passed = bool(complete and lower_bound is not None and lower_bound > -margin)
    if not complete:
        decision = "incomplete_no_inference"
    elif noninferiority_passed:
        decision = "noninferiority_confirmed"
    else:
        decision = "noninferiority_not_confirmed"

    return {
        "schema_version": 1,
        "analysis": "paired_no_code_guide_noninferiority",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "completion_gate": {
            "passed": complete,
            "expected_paired_keys": len(expected_keys),
            "analyzable_paired_keys": len(pairs),
        },
        "arm_status": arm_status,
        "primary_noninferiority": {
            "estimand": "no_code_minus_full_absolute_pass_probability",
            "margin": margin,
            "null": f"delta <= {-margin:+.3f}",
            "alternative": f"delta > {-margin:+.3f}",
            "confidence_method": "Newcombe paired score interval method 10",
            "one_sided_confidence": 0.95,
            "paired_table": {
                "both_pass": both_pass,
                "full_pass_no_code_fail": full_only,
                "full_fail_no_code_pass": no_code_only,
                "both_fail": both_fail,
            },
            "estimate_and_one_sided_95_lower": primary_interval,
            "passed": noninferiority_passed,
            "decision": decision,
        },
        "descriptive_paired_inference": {
            "newcombe_method10_two_sided_95": descriptive_interval,
            "mcnemar_exact_two_sided_p": mcnemar_p,
        },
        "transition_taxonomy": dict(transition_taxonomy),
        "binary_cell_summary": cell_summary,
        "binary_cell_majority_topology_flips": topology_flips,
        "case_clustered_pass_model": pass_model,
        "case_clustered_normalized_error_model": error_model,
        "paired_secondary_deltas_no_code_minus_full": resource_deltas,
        "case_order_diagnostics": order_diagnostics,
        "interpretation_boundary": (
            "The registered non-inferiority decision is valid only after the "
            "completion gate passes. Partial results are operational diagnostics. "
            "A passed result applies only to this panel, model, runner, and "
            "guide intervention."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    gate = payload["completion_gate"]
    primary = payload["primary_noninferiority"]
    table = primary["paired_table"]
    interval = primary["estimate_and_one_sided_95_lower"]
    lines = [
        "# No-code-guide paired non-inferiority ablation",
        "",
        f"Completion gate: **{gate['passed']}** "
        f"({gate['analyzable_paired_keys']}/{gate['expected_paired_keys']} pairs).",
        "",
        f"Decision: **{primary['decision']}**.",
        "",
        "| Full guide | No-code guide | Count |",
        "|---|---|---:|",
        f"| pass | pass | {table['both_pass']} |",
        f"| pass | fail | {table['full_pass_no_code_fail']} |",
        f"| fail | pass | {table['full_fail_no_code_pass']} |",
        f"| fail | fail | {table['both_fail']} |",
        "",
    ]
    if interval is not None:
        lines.extend(
            [
                f"No-code-minus-full pass difference: **{interval['estimate']:+.3f}**.",
                "",
                f"Registered one-sided 95% lower limit: **{interval['lower']:+.3f}**; "
                f"non-inferiority margin: **{-primary['margin']:+.3f}**.",
                "",
            ]
        )
    lines.extend(
        [
            f"Binary-cell majority-topology flips: "
            f"**{payload['binary_cell_majority_topology_flips']}**.",
            "",
            "## Arm status",
            "",
            "| Arm | Records | Scientific/deterministic | Retriable infra |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, row in payload["arm_status"].items():
        lines.append(
            f"| `{arm}` | {row['latest_selected_records']}/{row['expected_records']} | "
            f"{row['scientific_or_deterministic_records']} | "
            f"{row['unresolved_retriable_infrastructure']} |"
        )
    lines.extend(["", f"> {payload['interpretation_boundary']}", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--analysis-root", type=Path, default=ROOT / "analysis/phase2")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT / "paired_analysis.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT / "paired_analysis.md")
    args = parser.parse_args(argv)
    payload = analyze(
        manifest_path=args.manifest,
        cases_path=args.cases,
        analysis_root=args.analysis_root,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "completion_gate": payload["completion_gate"],
                "decision": payload["primary_noninferiority"]["decision"],
                "arm_status": {
                    arm: {
                        "records": row["latest_selected_records"],
                        "retriable_infra": row["unresolved_retriable_infrastructure"],
                    }
                    for arm, row in payload["arm_status"].items()
                },
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0 if payload["completion_gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
