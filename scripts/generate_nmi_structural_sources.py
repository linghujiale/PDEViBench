#!/usr/bin/env python3
"""Write source data and macros for the route attribution and the panel summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROUTE_DIR = ROOT / "analysis/nmi_route_mixture_20260811"
OUT_DIR = ROOT / "analysis/nmi_structural_rebuild_20260811"
MACRO_PATH = ROOT / "Nature/generated/nmi_structural_rebuild_macros.tex"
FIG1_DIR = ROOT / "paper/source_data/fig1_endpoint_nonidentifiability"
PRIMARY_ID = "phase2_wave1_codex_scipy_20260522T030958Z"
CODEX_BREADTH = "axis1024_codex_scipy_n3_20260710T000000Z"
CLAUDE_BREADTH = "axis1024_claude_opus47_scipy_n3_20260716T044105Z"
SCAFFOLD_FULL = "scaffold64_fullguide_codex_gpt55_scipy_n2_20260801"
SCAFFOLD_MINIMAL = "scaffold64_minimalguide_codex_gpt55_scipy_n2_20260801"
PCSEP_FULL = "pcsep128_fullguide_codex_gpt55_scipy_n1_20260802"
PCSEP_PRESCRIPTION = "pcsep128_prescriptionguide_codex_gpt55_scipy_n1_20260802"
PCSEP_CHOICE_FREE = "pcsep128_nocodeguide_codex_gpt55_scipy_n1_20260802"
EXPANDED_REVIEW_FINAL = (
    ROOT
    / "analysis/nmi_phase4b_20260811/blinded_review_completion/final_results_20260812/frozen"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator)


def rounded(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def macro(name: str, value: str | int) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> int:
    completed_review_path = EXPANDED_REVIEW_FINAL / "completion_summary.json"
    review_complete = completed_review_path.is_file()
    decomp_path = (
        EXPANDED_REVIEW_FINAL / "route_decomposition_by_experiment_id_postreview.csv"
        if review_complete
        else ROUTE_DIR / "route_decomposition_by_experiment_id.csv"
    )
    census_path = (
        EXPANDED_REVIEW_FINAL / "record_route_census_postreview.csv"
        if review_complete
        else ROUTE_DIR / "record_route_census.csv"
    )
    route_summary_path = ROUTE_DIR / "route_mixture_summary.json"
    order_path = ROUTE_DIR / "execution_order_sensitivity.json"
    human_path = ROOT / "analysis/independent_program_validation_20260811/empirical_results_author_attested.json"
    human_consensus_path = ROOT / "analysis/independent_program_validation_20260811/consensus_labels_author_attested.jsonl"
    agreement_path = ROOT / (
        "analysis/phase2/axis1024_codex_scipy_n3_20260710T000000Z__vs__"
        "axis1024_claude_opus47_scipy_n3_20260716T044105Z/case_agreement_summary.json"
    )
    prescription_path = ROOT / "analysis/phase2/prescription_code_separation_20260802/paired_analysis.json"
    external_path = ROOT / "analysis/external_validation_20260811/results/summary.json"
    inputs = [
        decomp_path,
        census_path,
        route_summary_path,
        order_path,
        human_path,
        human_consensus_path,
        agreement_path,
        prescription_path,
        external_path,
    ]
    if review_complete:
        inputs.append(completed_review_path)

    decomp = {row["group"]: row for row in read_csv(decomp_path)}
    route_summary = read_json(route_summary_path)
    order = read_json(order_path)
    human = read_json(human_path)
    human_consensus = [
        json.loads(line)
        for line in human_consensus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    human_consensus_counts = Counter(
        str(row["consensus_label"]) for row in human_consensus
    )
    expanded_completion = read_json(completed_review_path) if review_complete else None
    census = read_csv(census_path)
    endpoint_census = [row for row in census if row["endpoint_pass"].lower() == "true"]
    census_route_counts = Counter(row["route_status"] for row in endpoint_census)
    agreement = read_json(agreement_path)
    prescription = read_json(prescription_path)
    external = read_json(external_path)

    primary = decomp[PRIMARY_ID]
    scheduled = 1450
    endpoint = int(primary["endpoint_passes"])
    valid = int(primary["route_valid"])
    direct = int(primary["direct_recovery"])
    mixed = int(primary["mixed_route"])
    unresolved = int(primary["unresolved"])
    invalid = direct + mixed
    known_invalid_corrected = endpoint - invalid
    if endpoint != valid + invalid + unresolved:
        raise ValueError("primary route decomposition does not close")
    if known_invalid_corrected != valid + unresolved:
        raise ValueError("known-invalid-corrected count is inconsistent")

    codex_breadth_decomp = decomp[CODEX_BREADTH]
    claude_breadth_decomp = decomp[CLAUDE_BREADTH]
    codex_order = order[CODEX_BREADTH]
    claude_order = order[CLAUDE_BREADTH]
    prescription_arms = prescription["arm_status"]
    external_agents = external["agents"]

    payload = {
        "schema_version": 1,
        "freeze_id": "nmi_structural_headline_numbers_20260811_v1",
        "status": (
            "completed_expanded_blinded_review"
            if review_complete
            else "interval_pending_218_record_blinded_review"
        ),
        "primary_route_semantics": {
            "scheduled": scheduled,
            "endpoint_pass": endpoint,
            "endpoint_rate_pct": rounded(pct(endpoint, scheduled)),
            "confirmed_route_valid": valid,
            "strict_route_valid_lower_rate_pct": rounded(pct(valid, scheduled)),
            "audit_conditioned_established_route_valid_lower_count": valid,
            "audit_conditioned_established_route_valid_lower_rate_pct": rounded(pct(valid, scheduled)),
            "confirmed_direct": direct,
            "confirmed_mixed": mixed,
            "confirmed_route_invalid": invalid,
            "confirmed_invalid_rate_pct": rounded(pct(invalid, scheduled)),
            "route_unresolved": unresolved,
            "route_unresolved_rate_pct": rounded(pct(unresolved, scheduled)),
            "known_invalid_corrected_upper_count": known_invalid_corrected,
            "known_invalid_corrected_upper_rate_pct": rounded(pct(known_invalid_corrected, scheduled)),
            "current_route_valid_interval_pct": [
                rounded(pct(valid, scheduled)),
                rounded(pct(known_invalid_corrected, scheduled)),
            ],
            "audit_conditioned_partial_identification_interval_pct": [
                rounded(pct(valid, scheduled)),
                rounded(pct(known_invalid_corrected, scheduled)),
            ],
            "confirmed_invalid_endpoint_inflation_points": rounded(pct(invalid, scheduled)),
            "endpoint_minus_strict_lower_bound_points": rounded(pct(endpoint - valid, scheduled)),
            "confirmed_invalid_share_of_endpoint_pct": rounded(pct(invalid, endpoint)),
            "route_unresolved_share_of_endpoint_pct": rounded(pct(unresolved, endpoint)),
        },
        "heterogeneous_audit_corpus": {
            "records": len(census),
            "endpoint_pass": len(endpoint_census),
            "route_valid": int(census_route_counts["route_valid"]),
            "direct": int(census_route_counts["direct_recovery"]),
            "mixed": int(census_route_counts["mixed_route"]),
            "unresolved": int(census_route_counts["unresolved"]),
            "pooled_capability_rate_permitted": False,
        },
        "completed_blinded_validation": {
            "double_reviewed": int(human["n_double_reviewed"]),
            "trigger_selected": int(human["consensus_by_sampling_group"]["TRIGGER_SELECTED"]["n"]),
            "passing_controls": int(human["consensus_by_sampling_group"]["PASSING_CONTROL"]["n"]),
            "four_class_agreement_pct": rounded(100.0 * human["four_class"]["raw_agreement"], 1),
            "four_class_kappa": rounded(human["four_class"]["cohen_kappa"], 3),
            "binary_agreement_pct": rounded(100.0 * human["binary"]["raw_agreement"], 1),
            "binary_kappa": rounded(human["binary"]["cohen_kappa"], 3),
            "adjudicated_disagreements": int(human["n_disagreements"]),
            "consensus_numerical": int(human_consensus_counts["NUMERICAL_SOLVER"]),
            "consensus_direct": int(human_consensus_counts["DIRECT_REFERENCE_RECOVERY"]),
            "consensus_mixed": int(human_consensus_counts["MIXED_SOLVER_REFERENCE_RECOVERY"]),
            "consensus_route_invalid": int(
                human_consensus_counts["DIRECT_REFERENCE_RECOVERY"]
                + human_consensus_counts["MIXED_SOLVER_REFERENCE_RECOVERY"]
            ),
            "rule_matches": int(human["automated_rule_agreement"]["matches"]),
            "rule_match_pct": rounded(100.0 * human["automated_rule_agreement"]["fraction"], 1),
            "rule_match_wilson_pct": [
                rounded(100.0 * human["automated_rule_agreement"]["wilson_95_interval"][0], 1),
                rounded(100.0 * human["automated_rule_agreement"]["wilson_95_interval"][1], 1),
            ],
        },
        "expanded_blinded_review": (
            {
                **expanded_completion,
                "bank_input_unresolved_records": int(
                    route_summary["review_extension"]["n_unresolved"]
                ),
                "bank_route_valid_controls": int(
                    route_summary["review_extension"]["n_stratified_route_valid_controls"]
                ),
            }
            if review_complete
            else {
                "items": int(route_summary["review_extension"]["n_review_items"]),
                "unresolved": int(route_summary["review_extension"]["n_unresolved"]),
                "route_valid_controls": int(
                    route_summary["review_extension"]["n_stratified_route_valid_controls"]
                ),
                "status": route_summary["review_extension"]["status"],
            }
        ),
        "breadth": {
            "cases": int(agreement["n_cases"]),
            "records_per_agent": int(agreement["expected_verdicts_per_agent"]),
            "codex_endpoint_pass": int(agreement["left_total_passes"]),
            "claude_endpoint_pass": int(agreement["right_total_passes"]),
            "codex_route_valid_pass": int(codex_breadth_decomp["route_valid"]),
            "claude_route_valid_pass": int(claude_breadth_decomp["route_valid"]),
            "claude_direct": int(claude_breadth_decomp["direct_recovery"]),
            "claude_mixed": int(claude_breadth_decomp["mixed_route"]),
            "claude_unresolved_endpoint_pass": int(claude_breadth_decomp["unresolved"]),
            "same_pass_count_cases": int(agreement["same_pass_count_cases"]),
            "pass_count_pearson_r": rounded(agreement["pearson_r_pass_counts"], 3),
            "codex_max_axis_order_correlation": rounded(codex_order["max_abs_axis_order_correlation"], 4),
            "claude_max_axis_order_correlation": rounded(claude_order["max_abs_axis_order_correlation"], 4),
            "codex_rerun_for_current_descriptive_claims": order["recommendation"]["codex_breadth_rerun_required_for_current_descriptive_claims"],
            "codex_rerun_for_causal_axis_claims": order["recommendation"]["codex_breadth_rerun_required_for_codex_specific_causal_axis_claims"],
        },
        "prescription_intervention": {
            "cases_per_arm": int(prescription["completion_gate"]["expected_paired_keys"]),
            "full_pass": int(prescription_arms["full_guide"]["taxonomy"]["pass"]),
            "prescription_only_pass": int(prescription_arms["prescription_only"]["taxonomy"]["pass"]),
            "choice_free_pass": int(prescription_arms["choice_free_no_code"]["taxonomy"]["pass"]),
            "prescription_only_minus_full_lower_points": rounded(
                100.0 * prescription["ordered_confirmatory_tests"][0]["one_sided_95_interval"]["lower"], 1
            ),
            "prescription_only_minus_choice_free_points": rounded(
                100.0 * prescription["ordered_confirmatory_tests"][1]["one_sided_95_interval"]["estimate"], 1
            ),
            "prescription_only_minus_choice_free_lower_points": rounded(
                100.0 * prescription["ordered_confirmatory_tests"][1]["one_sided_95_interval"]["lower"], 1
            ),
            "full_route_valid": int(decomp[PCSEP_FULL]["route_valid"]),
            "prescription_only_route_valid": int(decomp[PCSEP_PRESCRIPTION]["route_valid"]),
            "choice_free_route_valid": int(decomp[PCSEP_CHOICE_FREE]["route_valid"]),
            "choice_free_confirmed_invalid": int(decomp[PCSEP_CHOICE_FREE]["direct_recovery"])
            + int(decomp[PCSEP_CHOICE_FREE]["mixed_route"]),
            "unresolved": sum(
                int(decomp[experiment_id]["unresolved"])
                for experiment_id in (PCSEP_FULL, PCSEP_PRESCRIPTION, PCSEP_CHOICE_FREE)
            ),
        },
        "full_minimal_intervention": {
            "full_output_pass": int(decomp[SCAFFOLD_FULL]["endpoint_passes"]),
            "minimal_output_pass": int(decomp[SCAFFOLD_MINIMAL]["endpoint_passes"]),
            "full_route_valid": int(decomp[SCAFFOLD_FULL]["route_valid"]),
            "minimal_route_valid": int(decomp[SCAFFOLD_MINIMAL]["route_valid"]),
            "minimal_confirmed_invalid": int(decomp[SCAFFOLD_MINIMAL]["direct_recovery"])
            + int(decomp[SCAFFOLD_MINIMAL]["mixed_route"]),
            "unresolved": int(decomp[SCAFFOLD_FULL]["unresolved"])
            + int(decomp[SCAFFOLD_MINIMAL]["unresolved"]),
        },
        "formula_blind_external_control": {
            "cases": int(external["reference_convergence"]["n_cases"]),
            "records": int(external["observed_records"]),
            "codex_pass": int(external_agents["Codex GPT-5.5"]["case_pass"]),
            "claude_pass": int(external_agents["Claude Opus 4.7"]["case_pass"]),
            "route_invalid_detected": sum(
                int(agent["endpoint_pass_reclassified_by_method"])
                for agent in external_agents.values()
            ),
            "median_reference_gap_pct": rounded(100.0 * external["reference_convergence"]["median_gap_relative_l2"], 2),
            "maximum_reference_gap_pct": rounded(100.0 * external["reference_convergence"]["maximum_gap_relative_l2"], 2),
        },
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headline_path = OUT_DIR / "headline_numbers.json"
    headline_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    p = payload["primary_route_semantics"]
    h = payload["completed_blinded_validation"]
    x = payload["expanded_blinded_review"]
    b = payload["breadth"]
    s = payload["prescription_intervention"]
    sm = payload["full_minimal_intervention"]
    e = payload["formula_blind_external_control"]
    a = payload["heterogeneous_audit_corpus"]
    expanded_input_unresolved = x.get("bank_input_unresolved_records", x.get("unresolved"))
    expanded_control_count = x.get("bank_route_valid_controls")
    if expanded_control_count is None:
        control_block = x.get("route_valid_controls")
        expanded_control_count = (
            control_block.get("n") if isinstance(control_block, dict) else control_block
        )
    lines = [
        "% Auto-generated by scripts/generate_nmi_structural_sources.py; do not edit.",
        macro("NMIPrimaryScheduledCount", p["scheduled"]),
        macro("NMIEndpointPassCount", p["endpoint_pass"]),
        macro("NMIEndpointPassRate", f"{p['endpoint_rate_pct']:.2f}\\%"),
        macro("NMIConfirmedRouteValidCount", p["confirmed_route_valid"]),
        macro("NMIStrictRouteValidLowerRate", f"{p['strict_route_valid_lower_rate_pct']:.2f}\\%"),
        macro("NMIAuditConditionedEstablishedRouteValidCount", p["confirmed_route_valid"]),
        macro("NMIAuditConditionedEstablishedRouteValidRate", f"{p['strict_route_valid_lower_rate_pct']:.2f}\\%"),
        macro("NMIConfirmedDirectCount", p["confirmed_direct"]),
        macro("NMIConfirmedMixedCount", p["confirmed_mixed"]),
        macro("NMIConfirmedRouteInvalidCount", p["confirmed_route_invalid"]),
        macro("NMIConfirmedInvalidRate", f"{p['confirmed_invalid_rate_pct']:.2f}\\%"),
        macro("NMIRouteUnresolvedCount", p["route_unresolved"]),
        macro("NMIRouteUnresolvedRate", f"{p['route_unresolved_rate_pct']:.2f}\\%"),
        macro("NMIKnownInvalidCorrectedCount", p["known_invalid_corrected_upper_count"]),
        macro("NMIKnownInvalidCorrectedUpperRate", f"{p['known_invalid_corrected_upper_rate_pct']:.2f}\\%"),
        macro("NMIRouteValidInterval", f"{p['current_route_valid_interval_pct'][0]:.2f}--{p['current_route_valid_interval_pct'][1]:.2f}\\%"),
        macro("NMIAuditConditionedPartialIdentificationInterval", f"{p['current_route_valid_interval_pct'][0]:.2f}--{p['current_route_valid_interval_pct'][1]:.2f}\\%"),
        macro("NMIConfirmedInvalidInflationPoints", f"{p['confirmed_invalid_endpoint_inflation_points']:.2f}"),
        macro("NMIEndpointStrictGapPoints", f"{p['endpoint_minus_strict_lower_bound_points']:.2f}"),
        macro("NMIConfirmedInvalidEndpointShare", f"{p['confirmed_invalid_share_of_endpoint_pct']:.2f}\\%"),
        macro("NMIAuditCorpusRecords", a["records"]),
        macro("NMIAuditCorpusEndpointPasses", a["endpoint_pass"]),
        macro("NMIAuditCorpusRouteValid", a["route_valid"]),
        macro("NMIAuditCorpusDirect", a["direct"]),
        macro("NMIAuditCorpusMixed", a["mixed"]),
        macro("NMIAuditCorpusClaimInvalid", a["direct"] + a["mixed"]),
        macro("NMIAuditCorpusUnresolved", a["unresolved"]),
        macro("NMICompletedDoubleReviewCount", h["double_reviewed"]),
        macro("NMICompletedTriggerReviewCount", h["trigger_selected"]),
        macro("NMICompletedControlReviewCount", h["passing_controls"]),
        macro("NMIFourClassAgreement", f"{h['four_class_agreement_pct']:.1f}\\%"),
        macro("NMIFourClassKappa", f"{h['four_class_kappa']:.3f}"),
        macro("NMIBinaryAgreement", f"{h['binary_agreement_pct']:.1f}\\%"),
        macro("NMIBinaryKappa", f"{h['binary_kappa']:.3f}"),
        macro("NMIAdjudicatedDisagreementCount", h["adjudicated_disagreements"]),
        macro("NMICompletedConsensusNumericalCount", h["consensus_numerical"]),
        macro("NMICompletedConsensusDirectCount", h["consensus_direct"]),
        macro("NMICompletedConsensusMixedCount", h["consensus_mixed"]),
        macro("NMICompletedConsensusRouteInvalidCount", h["consensus_route_invalid"]),
        macro("NMIExpandedReviewItemCount", x["items"]),
        macro("NMIExpandedReviewUnresolvedCount", expanded_input_unresolved),
        macro("NMIExpandedReviewControlCount", expanded_control_count),
        macro("NMIExpandedFourClassAgreement", f"{x['agreement']['four_class']['agreement_pct']:.2f}\\%"),
        macro("NMIExpandedFourClassKappa", f"{x['agreement']['four_class']['cohen_kappa']:.3f}"),
        macro("NMIExpandedBinaryAgreement", f"{x['agreement']['collapsed_valid_invalid']['agreement_pct']:.2f}\\%"),
        macro("NMIExpandedBinaryKappa", f"{x['agreement']['collapsed_valid_invalid']['cohen_kappa']:.3f}"),
        macro("NMIExpandedAdjudicatedDisagreementCount", x["adjudicated_disagreements"]),
        macro("NMIExpandedConsensusNumericalCount", x["label_counts"]["NUMERICAL_SOLVER"]),
        macro("NMIExpandedConsensusDirectCount", x["label_counts"]["DIRECT_REFERENCE_RECOVERY"]),
        macro("NMIExpandedConsensusMixedCount", x["label_counts"]["MIXED_SOLVER_REFERENCE_RECOVERY"]),
        macro("NMIExpandedControlInvalidCount", x["off_trigger_invalid_route_uncertainty"]["invalid_controls"]),
        macro("NMIExpandedOffTriggerWilsonUpper", f"{x['off_trigger_invalid_route_uncertainty']['wilson_95_pct'][1]:.2f}\\%"),
        macro("NMIBreadthCaseCount", b["cases"]),
        macro("NMIBreadthRecordsPerAgent", b["records_per_agent"]),
        macro("NMICodexBreadthEndpointPassCount", b["codex_endpoint_pass"]),
        macro("NMIClaudeBreadthEndpointPassCount", b["claude_endpoint_pass"]),
        macro("NMICodexBreadthRouteValidCount", b["codex_route_valid_pass"]),
        macro("NMIClaudeBreadthRouteValidCount", b["claude_route_valid_pass"]),
        macro("NMIClaudeBreadthDirectCount", b["claude_direct"]),
        macro("NMIClaudeBreadthMixedCount", b["claude_mixed"]),
        macro("NMIClaudeBreadthUnresolvedCount", b["claude_unresolved_endpoint_pass"]),
        macro("NMIBreadthSamePassCountCases", b["same_pass_count_cases"]),
        macro("NMIBreadthPassCountCorrelation", f"{b['pass_count_pearson_r']:.3f}"),
        macro("NMIPrescriptionCasesPerArm", s["cases_per_arm"]),
        macro("NMIFullGuidePassCount", s["full_pass"]),
        macro("NMIPrescriptionOnlyPassCount", s["prescription_only_pass"]),
        macro("NMIChoiceFreePassCount", s["choice_free_pass"]),
        macro("NMIFullGuideRouteValidPassCount", s["full_route_valid"]),
        macro("NMIPrescriptionOnlyRouteValidPassCount", s["prescription_only_route_valid"]),
        macro("NMIChoiceFreeRouteValidPassCount", s["choice_free_route_valid"]),
        macro("NMIChoiceFreeConfirmedInvalidCount", s["choice_free_confirmed_invalid"]),
        macro("NMIPrescriptionInterventionUnresolvedCount", s["unresolved"]),
        macro("NMIPrescriptionNoninferiorityLowerPoints", f"{s['prescription_only_minus_full_lower_points']:.1f}"),
        macro("NMIPrescriptionChoiceFreeDifferencePoints", f"{s['prescription_only_minus_choice_free_points']:.1f}"),
        macro("NMIPrescriptionChoiceFreeLowerPoints", f"{s['prescription_only_minus_choice_free_lower_points']:.1f}"),
        macro("NMIFullMinimalFullPassCount", sm["full_output_pass"]),
        macro("NMIFullMinimalMinimalPassCount", sm["minimal_output_pass"]),
        macro("NMIFullMinimalFullRouteValidPassCount", sm["full_route_valid"]),
        macro("NMIFullMinimalMinimalRouteValidPassCount", sm["minimal_route_valid"]),
        macro("NMIMinimalGuideConfirmedInvalidCount", sm["minimal_confirmed_invalid"]),
        macro("NMIFullMinimalUnresolvedCount", sm["unresolved"]),
        macro("NMIExternalCaseCount", e["cases"]),
        macro("NMIExternalRecordCount", e["records"]),
        macro("NMIExternalCodexPassCount", e["codex_pass"]),
        macro("NMIExternalClaudePassCount", e["claude_pass"]),
        macro("NMIExternalRouteInvalidDetectedCount", e["route_invalid_detected"]),
        macro("NMIExternalMedianReferenceGap", f"{e['median_reference_gap_pct']:.2f}\\%"),
        macro("NMIExternalMaximumReferenceGap", f"{e['maximum_reference_gap_pct']:.2f}\\%"),
    ]
    MACRO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MACRO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    FIG1_DIR.mkdir(parents=True, exist_ok=True)
    fig_rows = [
        {
            "component": "confirmed route-valid",
            "count": valid,
            "scheduled_rate_pct": f"{pct(valid, scheduled):.6f}",
            "endpoint_share_pct": f"{pct(valid, endpoint):.6f}",
            "evidence_status": "established admissible route",
        },
        {
            "component": "confirmed route-invalid",
            "count": invalid,
            "scheduled_rate_pct": f"{pct(invalid, scheduled):.6f}",
            "endpoint_share_pct": f"{pct(invalid, endpoint):.6f}",
            "evidence_status": "independent human consensus",
        },
        {
            "component": "route-unresolved",
            "count": unresolved,
            "scheduled_rate_pct": f"{pct(unresolved, scheduled):.6f}",
            "endpoint_share_pct": f"{pct(unresolved, endpoint):.6f}",
            "evidence_status": (
                "no postreview unresolved records"
                if review_complete and unresolved == 0
                else "pending expanded blinded review"
            ),
        },
    ]
    with (FIG1_DIR / "route_decomposition.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fig_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(fig_rows)
    (FIG1_DIR / "headline_numbers.json").write_text(
        json.dumps(payload["primary_route_semantics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"headline": str(headline_path), "macros": str(MACRO_PATH), "fig1_source": str(FIG1_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
