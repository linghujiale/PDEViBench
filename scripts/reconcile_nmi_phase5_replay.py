#!/usr/bin/env python3
"""Collect the unchanged-program replay results and write their source data and macros."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "analysis/nmi_phase4b_20260811/frozen_program_replay"
FREEZE = REPLAY / "program_freeze_manifest.json"
RESULTS = REPLAY / "agent_evaluation/program_results.jsonl"
ANALYSIS = REPLAY / "analysis/replay_analysis.json"
REPLAY_CASES = REPLAY / "case_bank.jsonl"
NATURALISTIC_CASES = ROOT / "benchmark/cases/external_naturalistic_32.jsonl"
NATURALISTIC_RESULTS = ROOT / "analysis/external_validation_20260811/results/run_level_source_data.csv"
OUT = ROOT / "analysis/nmi_phase5_20260813/replay_reconciliation"
MACROS = ROOT / "Nature/generated/nmi_replay_macros.tex"
SOURCE = ROOT / "paper/source_data/fig4_frozen_program_scientific_replay"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def one_sided_exact_upper_zero(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        raise ValueError("exact zero-event bound requires a positive denominator")
    return 1.0 - alpha ** (1.0 / n)


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not 0 <= k <= n or n <= 0:
        raise ValueError("invalid binomial count")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyse() -> dict[str, Any]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    frozen_analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    results = read_jsonl(RESULTS)
    replay_cases = read_jsonl(REPLAY_CASES)
    naturalistic_cases = read_jsonl(NATURALISTIC_CASES)

    if freeze.get("status") != "PROGRAMS_FROZEN" or len(freeze["programs"]) != 64:
        raise ValueError("generation manifest is not the expected 64-call record")
    if len(results) != 64 or len(replay_cases) != 32 or len(naturalistic_cases) != 32:
        raise ValueError("replay or naturalistic cardinality drift")
    frozen_by_id = {row["program_id"]: row for row in freeze["programs"]}
    if set(frozen_by_id) != {row["program_id"] for row in results}:
        raise ValueError("program result IDs do not match the generation manifest")

    submitted_programs = sum(bool(row["submitted_program"]) for row in freeze["programs"])
    generation_failures = sum(not bool(row["submitted_program"]) for row in freeze["programs"])
    if (submitted_programs, generation_failures) != (59, 5):
        raise ValueError("submitted/failure split differs from 59/5")

    accounting = Counter()
    placeholder_rows: list[dict[str, Any]] = []
    for row in results:
        frozen = frozen_by_id[row["program_id"]]
        kind = "submitted_program" if frozen["submitted_program"] else "evaluator_failure_placeholder"
        if len(row["executions"]) != 4:
            raise ValueError(f"scheduled call does not have four evaluator records: {row['program_id']}")
        accounting[(kind, "scheduled_call_records")] += 1
        for execution in row["executions"]:
            accounting[(kind, "evaluation_records")] += 1
            accounting[(kind, "hash_comparisons")] += int(
                bool(execution.get("program_sha256_before"))
                and bool(execution.get("program_sha256_after"))
            )
            accounting[(kind, "unchanged_hashes")] += int(bool(execution.get("program_unchanged")))
            accounting[(kind, "execution_timeouts")] += int(bool(execution.get("timed_out")))
        if kind == "evaluator_failure_placeholder":
            path = ROOT / str(frozen["program_path"])
            text = path.read_text(encoding="utf-8")
            if "Evaluator-owned placeholder" not in text:
                raise ValueError(f"failure artifact is not explicitly evaluator-owned: {path}")
            placeholder_rows.append(
                {
                    "program_id": row["program_id"],
                    "call_order": frozen["call_order"],
                    "case_id": row["case_id"],
                    "agent": row["agent"],
                    "generation_status": frozen["generation_status"],
                    "placeholder_sha256": frozen["program_sha256"],
                    "evaluation_records": len(row["executions"]),
                    "unchanged_hash_comparisons": sum(
                        bool(execution["program_unchanged"]) for execution in row["executions"]
                    ),
                }
            )

    expected_accounting = {
        ("submitted_program", "scheduled_call_records"): 59,
        ("submitted_program", "evaluation_records"): 236,
        ("submitted_program", "hash_comparisons"): 236,
        ("submitted_program", "unchanged_hashes"): 236,
        ("submitted_program", "execution_timeouts"): 0,
        ("evaluator_failure_placeholder", "scheduled_call_records"): 5,
        ("evaluator_failure_placeholder", "evaluation_records"): 20,
        ("evaluator_failure_placeholder", "hash_comparisons"): 20,
        ("evaluator_failure_placeholder", "unchanged_hashes"): 20,
        ("evaluator_failure_placeholder", "execution_timeouts"): 0,
    }
    for key, expected in expected_accounting.items():
        if accounting[key] != expected:
            raise ValueError(f"replay accounting drift for {key}: {accounting[key]} != {expected}")

    overlap = Counter((bool(row["base_output_pass"]), bool(row["all_replay_success"])) for row in results)
    if overlap != Counter({(True, True): 34, (False, False): 30}):
        raise ValueError(f"base/all-replay overlap drift: {overlap}")
    base_passes = sum(bool(row["base_output_pass"]) for row in results)
    all_replays = sum(bool(row["all_replay_success"]) for row in results)
    replay_qualified = sum(bool(row["claim_aligned_replay_success"]) for row in results)
    specification_counts = {
        label: sum(
            bool(next(execution for execution in row["executions"] if execution["label"] == label)["output_pass"])
            for row in results
        )
        for label in ("base", "source_scale", "boundary_data", "resolution")
    }
    if (base_passes, all_replays, replay_qualified) != (34, 34, 34):
        raise ValueError("primary outcome counts differ from 34/34/34")
    if specification_counts != {"base": 34, "source_scale": 34, "boundary_data": 36, "resolution": 50}:
        raise ValueError(f"specification counts drift: {specification_counts}")

    cluster = defaultdict(lambda: {"base": 0, "loss": 0, "replay_only": 0})
    for row in results:
        cluster[row["case_id"]]["base"] += int(bool(row["base_output_pass"]))
        cluster[row["case_id"]]["loss"] += int(bool(row["base_output_pass"]) and not bool(row["all_replay_success"]))
        cluster[row["case_id"]]["replay_only"] += int(bool(row["all_replay_success"]) and not bool(row["base_output_pass"]))
    base_positive_clusters = sum(values["base"] > 0 for values in cluster.values())
    loss_clusters = sum(values["loss"] > 0 for values in cluster.values())
    replay_only_clusters = sum(values["replay_only"] > 0 for values in cluster.values())
    if (len(cluster), base_positive_clusters, loss_clusters, replay_only_clusters) != (32, 18, 0, 0):
        raise ValueError("case-cluster replay overlap drift")

    _, discordance_wilson_upper = wilson_interval(0, 64, alpha=0.05)
    uncertainty = {
        "observed_finite_bank": {
            "base_pass": 34,
            "all_replay_pass": 34,
            "base_only_discordance": 0,
            "replay_only_discordance": 0,
            "signed_difference": 0.0,
            "percentage_points": 0.0,
        },
        "empirical_case_cluster_bootstrap": {
            "replicates": 10000,
            "reported_by_frozen_analysis": frozen_analysis["overall"]["base_replay_gap"]["cluster_bootstrap_ci95"],
            "interpretation": "degenerate empirical distribution; not interpreted as a superpopulation uncertainty interval",
        },
        "paired_discordance_score_envelope": {
            "method": "conservative simultaneous Wilson bounds for the two paired discordance cells",
            "confidence_level": 0.95,
            "interval": [-discordance_wilson_upper, discordance_wilson_upper],
            "percentage_point_interval": [-100 * discordance_wilson_upper, 100 * discordance_wilson_upper],
            "note": "sensitivity envelope for a signed paired difference, not the primary manuscript formulation",
        },
        "exact_upper_bounds": {
            "unconditional_base_only_loss_0_of_64": one_sided_exact_upper_zero(64),
            "conditional_loss_among_base_passing_0_of_34": one_sided_exact_upper_zero(34),
            "cluster_with_any_loss_0_of_32": one_sided_exact_upper_zero(32),
            "loss_among_base_positive_clusters_0_of_18": one_sided_exact_upper_zero(18),
            "confidence_level": 0.95,
            "method": "one-sided Clopper-Pearson exact upper bound",
        },
        "selected_primary_formulation": (
            "Report the observed finite-bank gap of 0.0 percentage points without an inferential interval. "
            "State that the empirical case-cluster bootstrap was degenerate and is not a superpopulation interval. "
            "Use the 0/34 one-sided exact upper bound only as a secondary sensitivity bound on conditional replay loss."
        ),
    }

    replay_source_ids = {str(row["provenance"]["source_case_id"]) for row in replay_cases}
    naturalistic_ids = {str(row["id"]) for row in naturalistic_cases}
    if replay_source_ids != naturalistic_ids:
        raise ValueError("replay base cases do not map one-to-one to the naturalistic case bank")
    with NATURALISTIC_RESULTS.open(encoding="utf-8", newline="") as handle:
        naturalistic_result_rows = list(csv.DictReader(handle))
    if len(naturalistic_result_rows) != 128 or sum(row["endpoint_pass"] == "True" for row in naturalistic_result_rows) != 128:
        raise ValueError("naturalistic result ledger differs from 128/128 endpoint passes")

    comparison = {
        "scientific_case_instances": "same underlying 32 base instances, mapped one-to-one through source_case_id and reparameterized for the callable schema",
        "family_boundary": "same four PDE families",
        "naturalistic_agents": ["Codex GPT-5.5", "Claude Opus 4.7"],
        "replay_configurations": ["Codex GPT-5.5/high", "Codex GPT-5.6 Sol/high"],
        "naturalistic_repeat_structure": "two independent calls per case and agent; 128 submitted programs",
        "replay_repeat_structure": "one single-shot call per case and configuration; 64 scheduled calls, 59 submissions",
        "naturalistic_interface": "standard benchmark scaffold and file-producing solution workflow",
        "replay_interface": "solve(problem_spec) -> numpy.ndarray callable contract with the same program bytes reused",
        "prompt_contract_equal": False,
        "model_configuration_equal": False,
        "repeat_structure_equal": False,
        "interpretation": (
            "The 128/128 naturalistic result and the 34/64 replay-base result are not a controlled before/after contrast. "
            "The causal replay comparison is within the replay experiment: base versus hidden specifications for the identical call record."
        ),
    }

    return {
        "schema_version": 1,
        "status": "PASS",
        "denominators": {
            "scheduled_generation_calls": 64,
            "submitted_programs": 59,
            "generation_failures": 5,
            "base_specifications": 32,
            "hidden_specifications_per_call": 3,
            "planned_evaluation_records": 256,
            "actual_submitted_program_executions": 236,
            "evaluator_placeholder_executions": 20,
            "scheduled_hidden_evaluation_records": 192,
            "actual_submitted_hidden_executions": 177,
            "placeholder_hidden_records": 15,
            "submitted_program_hash_comparisons": 236,
            "placeholder_hash_comparisons": 20,
            "all_hash_comparisons": 256,
            "execution_timeouts": 0,
        },
        "outcomes": {
            "base_output_level_success": base_passes,
            "all_replay_success": all_replays,
            "replay_qualified_success": replay_qualified,
            "source_replay_success": specification_counts["source_scale"],
            "boundary_replay_success": specification_counts["boundary_data"],
            "resolution_replay_success": specification_counts["resolution"],
        },
        "uncertainty": uncertainty,
        "naturalistic_vs_replay": comparison,
        "placeholder_records": placeholder_rows,
    }


def write_outputs(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (OUT / "replay_reconciliation.json").write_text(serialized, encoding="utf-8")
    denominators = payload["denominators"]
    outcomes = payload["outcomes"]
    exact = payload["uncertainty"]["exact_upper_bounds"]
    score = payload["uncertainty"]["paired_discordance_score_envelope"]



    write_csv(
        SOURCE / "execution_accounting.csv",
        [
            {"record_kind": "submitted scientific program", "scheduled_calls": 59, "base_records": 59, "hidden_records": 177, "all_evaluation_records": 236, "hash_comparisons": 236, "unchanged_hashes": 236},
            {"record_kind": "evaluator-owned generation-failure placeholder", "scheduled_calls": 5, "base_records": 5, "hidden_records": 15, "all_evaluation_records": 20, "hash_comparisons": 20, "unchanged_hashes": 20},
            {"record_kind": "all scheduled records", "scheduled_calls": 64, "base_records": 64, "hidden_records": 192, "all_evaluation_records": 256, "hash_comparisons": 256, "unchanged_hashes": 256},
        ],
        ["record_kind", "scheduled_calls", "base_records", "hidden_records", "all_evaluation_records", "hash_comparisons", "unchanged_hashes"],
    )
    write_csv(
        SOURCE / "replay_uncertainty_sensitivity.csv",
        [
            {"analysis": "observed finite-bank signed difference", "numerator": 0, "denominator": 64, "estimate": 0.0, "lower": "", "upper": "", "interpretation": "primary descriptive formulation"},
            {"analysis": "paired discordance Wilson score envelope", "numerator": 0, "denominator": 64, "estimate": 0.0, "lower": score["interval"][0], "upper": score["interval"][1], "interpretation": "conservative signed-difference sensitivity"},
            {"analysis": "conditional replay loss exact upper", "numerator": 0, "denominator": 34, "estimate": 0.0, "lower": 0.0, "upper": exact["conditional_loss_among_base_passing_0_of_34"], "interpretation": "one-sided 95% Clopper-Pearson"},
            {"analysis": "base-positive cluster loss exact upper", "numerator": 0, "denominator": 18, "estimate": 0.0, "lower": 0.0, "upper": exact["loss_among_base_positive_clusters_0_of_18"], "interpretation": "one-sided 95% Clopper-Pearson cluster event"},
        ],
        ["analysis", "numerator", "denominator", "estimate", "lower", "upper", "interpretation"],
    )
    write_csv(
        SOURCE / "naturalistic_vs_replay_design.csv",
        [
            {"feature": "scientific base instances", "naturalistic": "32 original formula-blind cases", "replay": "same 32 underlying cases, reparameterized callable schema"},
            {"feature": "agent/configuration", "naturalistic": "Codex GPT-5.5 and Claude Opus 4.7", "replay": "Codex GPT-5.5/high and GPT-5.6 Sol/high"},
            {"feature": "repeat structure", "naturalistic": "2 calls per case and agent; 128 submissions", "replay": "1 call per case and configuration; 64 scheduled, 59 submissions"},
            {"feature": "interface", "naturalistic": "standard scaffold and file-producing workflow", "replay": "solve(problem_spec) -> numpy.ndarray"},
            {"feature": "valid comparison", "naturalistic": "128/128 within its own design", "replay": "34/64 base and 34/64 all-replay within replay", "note": "not a controlled cross-experiment degradation contrast"},
        ],
        ["feature", "naturalistic", "replay", "note"],
    )

    macros = [
        "% Generated by scripts/reconcile_nmi_phase5_replay.py",
        "\\newcommand{\\NMIReplayScheduledCallCount}{64}",
        "\\newcommand{\\NMIReplayProgramCount}{59}",
        "\\newcommand{\\NMIReplayCaseCount}{32}",
        "\\newcommand{\\NMIReplayAgentConfigurationCount}{2}",
        "\\newcommand{\\NMIReplaySubmittedProgramCount}{59}",
        "\\newcommand{\\NMIReplayGenerationFailureCount}{5}",
        "\\newcommand{\\NMIReplayHiddenSpecificationCount}{3}",
        "\\newcommand{\\NMIReplayScheduledEvaluationRecordCount}{256}",
        "\\newcommand{\\NMIReplayExecutionCount}{236}",
        "\\newcommand{\\NMIReplayPlaceholderExecutionCount}{20}",
        "\\newcommand{\\NMIReplayScheduledHiddenRecordCount}{192}",
        "\\newcommand{\\NMIReplayHiddenExecutionCount}{177}",
        "\\newcommand{\\NMIReplayPlaceholderHiddenRecordCount}{15}",
        "\\newcommand{\\NMIReplayUnchangedHashCount}{236}",
        "\\newcommand{\\NMIReplayPlaceholderHashCount}{20}",
        "\\newcommand{\\NMIReplayAllHashCount}{256}",
        "\\newcommand{\\NMIReplayExecutionTimeoutCount}{0}",
        "\\newcommand{\\NMIReplayBasePassCount}{34}",
        "\\newcommand{\\NMIReplayBasePassRate}{53.1\\%}",
        "\\newcommand{\\NMIReplayAllPassCount}{34}",
        "\\newcommand{\\NMIReplayAllPassRate}{53.1\\%}",
        "\\newcommand{\\NMIReplayPrimaryCount}{34}",
        "\\newcommand{\\NMIReplayPrimaryRate}{53.1\\%}",
        "\\newcommand{\\NMIReplayCILower}{35.9\\%}",
        "\\newcommand{\\NMIReplayCIUpper}{70.3\\%}",
        "\\newcommand{\\NMIReplayEndpointGap}{0.0 percentage points}",
        f"\\newcommand{{\\NMIReplayConditionalLossUpper}}{{{100*exact['conditional_loss_among_base_passing_0_of_34']:.1f}\\%}}",
        "\\newcommand{\\NMIReplayBasePositiveClusterCount}{18}",
        f"\\newcommand{{\\NMIReplayClusterLossUpper}}{{{100*exact['loss_among_base_positive_clusters_0_of_18']:.1f}\\%}}",
        "\\newcommand{\\NMIReplaySourcePassCount}{34}",
        "\\newcommand{\\NMIReplayBoundaryPassCount}{36}",
        "\\newcommand{\\NMIReplayResolutionPassCount}{50}",
        "\\newcommand{\\NMIReplayEstablishedRouteValidCount}{59}",
        "\\newcommand{\\NMIReplayControlProgramCount}{128}",
        "\\newcommand{\\NMIReplayControlExecutionCount}{512}",
        "\\newcommand{\\NMIReplayPositiveControlCount}{32}",
        "\\newcommand{\\NMIReplayNegativeControlCount}{96}",
        "\\newcommand{\\NMIReplayGPTFiveFiveCount}{16}",
        "\\newcommand{\\NMIReplayGPTFiveSixSolCount}{18}",
    ]
    MACROS.write_text("\n".join(macros) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = analyse()
    write_outputs(result)
    print(json.dumps({"status": result["status"], **result["denominators"], **result["outcomes"]}, indent=2, sort_keys=True))
