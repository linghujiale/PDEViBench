#!/usr/bin/env python3
"""Benchmark-integrity checks and statistical reanalysis of the primary panel."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_nocode_noninferiority import newcombe_method10_interval

OUT_DIR = ROOT / "analysis/nmi_next_round_20260826"
SOURCE_DIR = ROOT / "paper/source_data/nmi_next_round"
PRIMARY_EXPERIMENT = "phase2_wave1_codex_scipy_20260522T030958Z"
PRIMARY_LEDGER = ROOT / "analysis/nmi_phase5_20260813/final_postreview_route_ledger.csv"
PRIMARY_RAW_RUNS = ROOT / f"analysis/phase2/{PRIMARY_EXPERIMENT}/runs.jsonl"
INTEGRITY_REPORT = ROOT / "analysis/benchmark_integrity_report.csv"
NATURALISTIC = ROOT / "benchmark/cases/external_naturalistic_32.jsonl"
INITIAL_REVIEW = ROOT / "analysis/independent_program_validation_20260811/empirical_results_author_attested.json"
EXPANDED_REVIEW = ROOT / "analysis/nmi_phase4b_20260811/blinded_review_completion/final_results_20260812/frozen/completion_summary.json"
INTEGRITY_SUMMARY = ROOT / "analysis/benchmark_integrity_summary.json"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_interval(left: list[bool], right: list[bool], z: float = 1.959963984540054) -> dict[str, float]:
    both = sum(a and b for a, b in zip(left, right, strict=True))
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(not a and b for a, b in zip(left, right, strict=True))
    neither = len(left) - both - left_only - right_only
    result = newcombe_method10_interval(
        both_pass=both,
        no_code_only=left_only,
        full_only=right_only,
        both_fail=neither,
        z=z,
    )
    return {**result, "both": both, "left_only": left_only, "right_only": right_only, "neither": neither}


def primary_statistics() -> dict[str, Any]:
    rows = [row for row in csv_rows(PRIMARY_LEDGER) if row["experiment_id"] == PRIMARY_EXPERIMENT]
    assert len(rows) == 1450
    endpoint = [row for row in rows if row["endpoint_pass"] == "True"]
    invalid = [row for row in endpoint if row["route_status"] in {"direct_recovery", "mixed_route"}]
    route_valid = [row for row in endpoint if row["route_status"] == "route_valid"]
    assert len(endpoint) == 1162 and len(route_valid) == 1037 and len(invalid) == 125
    observed_route_valid = len(route_valid)
    classifier_retained = sum(row["route_source"] == "deterministic_static_census" for row in route_valid)
    assert classifier_retained == 907
    by_case: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(int(row["endpoint_pass"] == "True" and row["route_status"] in {"direct_recovery", "mixed_route"}))
    values = np.asarray([sum(by_case[case_id]) / 5.0 for case_id in sorted(by_case)], dtype=float)
    rng = np.random.default_rng(20260826)
    boot = np.empty(20000, dtype=float)
    for i in range(len(boot)):
        boot[i] = rng.choice(values, size=len(values), replace=True).mean()
    estimate = float(values.mean())
    ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    scenarios = []
    for label, p in (("observed", 0.0), ("one_percent", 0.01), ("sampled_wilson_upper", 0.0291), ("five_percent", 0.05)):
        count = float(observed_route_valid) - float(classifier_retained) * p
        scenarios.append({"scenario": label, "false_negative_rate_p": p, "route_valid_count_equivalent": count, "route_valid_success_rate": count / 1450.0})
    integrity = json.loads(INTEGRITY_SUMMARY.read_text(encoding="utf-8"))
    defective = set(integrity["known_specification_reference_inconsistencies"])
    defective_rows = [row for row in rows if row["case_id"] in defective]
    defective_endpoint = [row for row in defective_rows if row["endpoint_pass"] == "True"]
    defective_route_valid = [row for row in defective_endpoint if row["route_status"] == "route_valid"]
    claim_supporting_count = observed_route_valid - len(defective_route_valid)
    return {
        "scheduled": 1450,
        "endpoint_passes": 1162,
        "observed_route_valid": observed_route_valid,
        "observed_direct_or_mixed": len(invalid),
        "confirmed_score_inflation": estimate,
        "case_cluster_bootstrap_95ci": ci,
        "bootstrap_replicates": 20000,
        "bootstrap_seed": 20260826,
        "interpretation": "Observed direct and mixed routes account for the confirmed score inflation.",
        "specification_fidelity": {
            "affected_primary_cases": len({row["case_id"] for row in defective_rows}),
            "affected_scheduled_calls": len(defective_rows),
            "endpoint_passes_on_affected_cases": len(defective_endpoint),
            "route_valid_endpoint_passes_excluded_by_fidelity": len(defective_route_valid),
            "claim_supporting_success_count_C_equals_R_times_F": claim_supporting_count,
            "claim_supporting_success_rate": claim_supporting_count / 1450.0,
        },
        "classifier_false_negative_scenarios": scenarios,
    }


def unaffected_primary_sensitivity() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct the affected-call ledger from raw runs and final route labels.

    ``runs.jsonl`` contains 29 duplicated write events from resumptions.  Every
    duplicate agrees on the endpoint verdict and artifact directory, so this
    function collapses exact case--repeat keys before joining the final route
    labels.  No scheduled call is removed from the prespecified primary panel.
    """

    def raw_endpoint_pass(row: dict[str, Any]) -> bool:
        verdict = row.get("verdict_summary", {})
        if "case_pass" in verdict:
            return bool(verdict["case_pass"])
        assert int(row["subprocess_returncode"]) != 0 and not verdict
        return False

    raw_rows = jsonl(PRIMARY_RAW_RUNS)
    raw_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in raw_rows:
        key = (str(row["case_id"]), int(row["run_idx"]))
        if key in raw_by_key:
            previous = raw_by_key[key]
            assert previous["agent_call"]["artifact_dir"] == row["agent_call"]["artifact_dir"]
            if raw_endpoint_pass(previous) != raw_endpoint_pass(row):
                assert bool(row.get("rerun_of_infra_failure"))
            if "case_pass" in previous.get("verdict_summary", {}) and "case_pass" not in row.get("verdict_summary", {}):
                continue
        raw_by_key[key] = row
    assert len(raw_by_key) == 1450

    ledger_rows = [row for row in csv_rows(PRIMARY_LEDGER) if row["experiment_id"] == PRIMARY_EXPERIMENT]
    ledger_by_key = {(row["case_id"], int(row["run_idx"])): row for row in ledger_rows}
    assert len(ledger_by_key) == 1450 and set(ledger_by_key) == set(raw_by_key)

    integrity = json.loads(INTEGRITY_SUMMARY.read_text(encoding="utf-8"))
    affected_all = set(integrity["known_specification_reference_inconsistencies"])
    primary_cases = {case_id for case_id, _ in raw_by_key}
    affected_cases = affected_all & primary_cases
    assert len(affected_cases) == 10

    audit_rows = csv_rows(INTEGRITY_REPORT)
    audit_reason_by_case: dict[str, str] = {}
    for row in audit_rows:
        if row["case_id"] not in affected_cases or row["overall_status"] != "fail":
            continue
        failed = set(row["failed_checks"].split(";"))
        if {"agent_facing_pde_residual", "forcing_representation"} <= failed:
            audit_reason_by_case[row["case_id"]] = (
                "agent-facing forcing is inconsistent with the stated operator and stored reference"
            )
    assert set(audit_reason_by_case) == affected_cases

    affected_rows: list[dict[str, Any]] = []
    unaffected_keys: list[tuple[str, int]] = []
    for key in sorted(raw_by_key):
        case_id, repeat = key
        raw = raw_by_key[key]
        ledger = ledger_by_key[key]
        endpoint_pass = raw_endpoint_pass(raw)
        assert endpoint_pass == (ledger["endpoint_pass"] == "True")
        if case_id in affected_cases:
            affected_rows.append({
                "case_id": case_id,
                "call_id": f"{case_id}::repeat_{repeat}",
                "repeat": repeat,
                "endpoint_pass": endpoint_pass,
                "route_label": ledger["route_status"],
                "fidelity_status": "affected_specification_reference_inconsistency",
                "inclusion_in_primary": True,
                "affected_reason": audit_reason_by_case[case_id],
            })
        else:
            unaffected_keys.append(key)

    assert len(affected_rows) == 50 and len(unaffected_keys) == 1400
    affected_passes = [row for row in affected_rows if row["endpoint_pass"]]
    assert len(affected_passes) == 2
    assert sorted(row["route_label"] for row in affected_passes) == ["direct_recovery", "route_valid"]

    endpoint_passes = sum(raw_endpoint_pass(raw_by_key[key]) for key in unaffected_keys)
    route_valid = sum(
        raw_endpoint_pass(raw_by_key[key])
        and ledger_by_key[key]["route_status"] == "route_valid"
        for key in unaffected_keys
    )
    direct_or_mixed = sum(
        raw_endpoint_pass(raw_by_key[key])
        and ledger_by_key[key]["route_status"] in {"direct_recovery", "mixed_route"}
        for key in unaffected_keys
    )
    assert endpoint_passes == 1160 and route_valid == 1036 and direct_or_mixed == 124
    summary = {
        "raw_run_records": len(raw_rows),
        "unique_primary_calls": len(raw_by_key),
        "incomplete_or_duplicate_resume_records_collapsed": len(raw_rows) - len(raw_by_key),
        "affected_cases": len(affected_cases),
        "affected_calls": len(affected_rows),
        "affected_endpoint_passes": len(affected_passes),
        "affected_route_valid_passes": sum(row["route_label"] == "route_valid" for row in affected_passes),
        "affected_direct_passes": sum(row["route_label"] == "direct_recovery" for row in affected_passes),
        "unaffected_calls": len(unaffected_keys),
        "unaffected_endpoint_passes": endpoint_passes,
        "unaffected_endpoint_success_rate": endpoint_passes / len(unaffected_keys),
        "unaffected_route_valid_passes": route_valid,
        "unaffected_route_valid_success_rate": route_valid / len(unaffected_keys),
        "unaffected_direct_or_mixed_passes": direct_or_mixed,
        "unaffected_score_gap_percentage_points": 100.0 * (endpoint_passes - route_valid) / len(unaffected_keys),
        "primary_inclusion_policy": "All 1,450 prespecified calls remain in the primary analysis; this is a diagnostic exclusion sensitivity.",
    }
    return affected_rows, summary


def naturalistic_margins() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in jsonl(NATURALISTIC):
        e_ref = float(case["reference"]["convergence_check"]["gap_relative_l2"])
        tau = float(case["thresholds"]["tau_acc"])
        rows.append({
            "case_id": case["id"],
            "family": case["case_spec"]["pde"]["operator_family"],
            "e_ref": e_ref,
            "tau_acc": tau,
            "rho": e_ref / tau,
            "requires_refinement": e_ref / tau >= 1.0,
        })
    family: dict[str, Any] = {}
    for name in sorted({row["family"] for row in rows}):
        values = [row["rho"] for row in rows if row["family"] == name]
        family[name] = {"cases": len(values), "median_rho": median(values), "maximum_rho": max(values), "all_below_one": all(value < 1 for value in values)}
    summary = {
        "cases": len(rows),
        "median_rho": median(row["rho"] for row in rows),
        "maximum_rho": max(row["rho"] for row in rows),
        "cases_requiring_refinement": sum(row["requires_refinement"] for row in rows),
        "family_summary": family,
    }
    return rows, summary


def review_reliability() -> dict[str, Any]:
    initial = json.loads(INITIAL_REVIEW.read_text(encoding="utf-8"))
    expanded = json.loads(EXPANDED_REVIEW.read_text(encoding="utf-8"))
    return {
        "initial_250_bank": {
            "sampling": {"trigger_selected": 125, "stratified_controls": 125},
            "four_class": initial["four_class"],
            "collapsed_binary": initial["binary"],
            "automated_vs_consensus": initial["automated_rule_agreement"],
            "evidence_status": initial["claim_status"],
        },
        "expanded_218_bank": {
            "sampling": {"formerly_unresolved": 90, "classifier_controls": 128},
            "four_class": expanded["agreement"]["four_class"],
            "collapsed_binary": expanded["agreement"]["collapsed_valid_invalid"],
            "control_result": "128/128 adjudicated numerical; no direct or mixed route observed",
            "scope": "The 2.91% Wilson upper bound applies to this sampled classifier-control bank, not automatically to all 907 classifier-retained primary passes.",
        },
    }


def scaffold_intervals() -> dict[str, Any]:
    rows = csv_rows(PRIMARY_LEDGER)
    experiments = {
        "full": "pcsep128_fullguide_codex_gpt55_scipy_n1_20260802",
        "prescription_only": "pcsep128_prescriptionguide_codex_gpt55_scipy_n1_20260802",
        "choice_free": "pcsep128_nocodeguide_codex_gpt55_scipy_n1_20260802",
    }
    maps: dict[str, dict[str, dict[str, str]]] = {}
    for arm, experiment in experiments.items():
        arm_rows = [row for row in rows if row["experiment_id"] == experiment]
        assert len(arm_rows) == 128
        maps[arm] = {row["case_id"]: row for row in arm_rows}
    case_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    assert len(case_ids) == 128
    def series(arm: str, kind: str) -> list[bool]:
        if kind == "route_valid":
            return [maps[arm][case_id]["endpoint_pass"] == "True" and maps[arm][case_id]["route_status"] == "route_valid" for case_id in case_ids]
        return [maps[arm][case_id]["endpoint_pass"] == "True" and maps[arm][case_id]["route_status"] == "direct_recovery" for case_id in case_ids]
    contrasts: dict[str, Any] = {}
    for left, right in (("prescription_only", "full"), ("prescription_only", "choice_free")):
        contrasts[f"{left}_minus_{right}"] = {
            kind: score_interval(series(left, kind), series(right, kind))
            for kind in ("route_valid", "direct_return")
        }
    counts = {
        arm: {
            "calls": 128,
            "output_pass": sum(maps[arm][case_id]["endpoint_pass"] == "True" for case_id in case_ids),
            "route_valid": sum(series(arm, "route_valid")),
            "direct_return": sum(series(arm, "direct_return")),
        }
        for arm in experiments
    }
    return {"arm_counts": counts, "paired_two_sided_95_intervals": contrasts, "interpretation": "Paired route readouts; no scaffold-by-visibility interaction is tested."}


def primary_metadata() -> dict[str, Any]:
    return {
        "experiment_id": PRIMARY_EXPERIMENT,
        "agent": "Codex CLI batch runner",
        "provider_model": "openai-gpt-5.2",
        "reasoning_effort": "unavailable (driver command and summary record no explicit setting)",
        "temperature": 0.2,
        "temperature_source": "agents/codex_runner.py runner default",
        "rng_seed_rule": "70000 + 1000 * run_idx; run_idx=0,...,4",
        "tool_environment": "Codex CLI shell/file tools in an isolated working directory; plugins disabled; agent prompt prohibited network access",
        "library_scaffold": "canonical SciPy/NumPy guide",
        "agent_wall_timeout_seconds": 600,
        "recorded_max_turns": 30,
        "evaluator": "host evaluator; three fresh executions; accuracy then 3*t_base resource gate",
        "provenance": [
            "analysis/phase2/phase2_wave1_codex_scipy_20260522T030958Z/driver.cmd",
            "analysis/phase2/phase2_wave1_codex_scipy_20260522T030958Z/driver.log",
            "analysis/phase2/phase2_wave1_codex_scipy_20260522T030958Z/runs.jsonl",
            "agents/codex_runner.py",
            "scripts/run_phase2_wave1.py",
        ],
    }


def main_primary_only() -> int:
    """Refresh only the primary-panel blocks after a route-label change."""
    affected_rows, unaffected_summary = unaffected_primary_sensitivity()
    primary = primary_statistics()
    scaffold = scaffold_intervals()
    for path in (OUT_DIR / "statistical_reanalysis.json", SOURCE_DIR / "statistical_reanalysis.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["primary_score_inflation"] = primary
        payload["unaffected_primary_sensitivity"] = unaffected_summary
        payload["scaffold_route_readouts"] = scaffold
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (SOURCE_DIR / "classifier_false_negative_scenarios.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = primary["classifier_false_negative_scenarios"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    affected_fields = list(affected_rows[0])
    for path in (
        ROOT / "analysis/affected_primary_calls.csv",
        OUT_DIR / "affected_primary_calls.csv",
        SOURCE_DIR / "affected_primary_calls.csv",
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=affected_fields)
            writer.writeheader(); writer.writerows(affected_rows)
    for path in (OUT_DIR / "unaffected_primary_sensitivity.json", SOURCE_DIR / "unaffected_primary_sensitivity.json"):
        path.write_text(json.dumps(unaffected_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"primary_score_inflation": primary, "unaffected_primary_sensitivity": unaffected_summary}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    if "--primary-only" in sys.argv[1:]:
        return main_primary_only()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    margins, margin_summary = naturalistic_margins()
    affected_rows, unaffected_summary = unaffected_primary_sensitivity()
    payload = {
        "schema_version": 1,
        "primary_score_inflation": primary_statistics(),
        "naturalistic_reference_margins": margin_summary,
        "route_review_reliability": review_reliability(),
        "scaffold_route_readouts": scaffold_intervals(),
        "primary_panel_metadata": primary_metadata(),
        "unaffected_primary_sensitivity": unaffected_summary,
    }
    (OUT_DIR / "statistical_reanalysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT_DIR / "naturalistic_reference_margins.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(margins[0]))
        writer.writeheader(); writer.writerows(margins)
    with (SOURCE_DIR / "classifier_false_negative_scenarios.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = payload["primary_score_inflation"]["classifier_false_negative_scenarios"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (SOURCE_DIR / "naturalistic_reference_margins.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(margins[0])); writer.writeheader(); writer.writerows(margins)
    (SOURCE_DIR / "statistical_reanalysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    affected_fields = list(affected_rows[0])
    for path in (
        ROOT / "analysis/affected_primary_calls.csv",
        OUT_DIR / "affected_primary_calls.csv",
        SOURCE_DIR / "affected_primary_calls.csv",
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=affected_fields)
            writer.writeheader(); writer.writerows(affected_rows)
    for path in (OUT_DIR / "unaffected_primary_sensitivity.json", SOURCE_DIR / "unaffected_primary_sensitivity.json"):
        path.write_text(json.dumps(unaffected_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
