#!/usr/bin/env python3
"""Build matched-visibility Source Data, manuscript macros and figure inputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_matched_visibility import (  # noqa: E402
    artifact_dir,
    build_records,
    deterministic_route_audit,
    latest_runs,
    read_jsonl,
)

DESIGN = ROOT / "analysis" / "matched_visibility_20260825"
PRIMARY_ANALYSIS = (
    ROOT / "analysis" / "phase2" / "matched_visibility_codex_gpt56sol_scipy_20260825"
)
DIAGNOSTIC_ANALYSIS = (
    ROOT
    / "analysis"
    / "phase2"
    / "matched_visibility_corrected_case_codex_gpt56sol_scipy_20260825"
)
SOURCE_DIR = ROOT / "paper" / "source_data" / "fig3_matched_visibility"
MACROS = ROOT / "Nature" / "generated" / "nmi_matched_visibility_macros.tex"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_sided_zero_event_upper(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - alpha ** (1.0 / n)


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}\\%"


def main() -> int:
    schedule = read_jsonl(DESIGN / "scheduled_observations.jsonl")
    runs = latest_runs(PRIMARY_ANALYSIS / "runs.jsonl")
    records = build_records(schedule, runs, human=None)
    if len(records) != 256:
        raise AssertionError(f"expected 256 primary observations, found {len(records)}")

    observation_rows = []
    all_program_routes = Counter()
    routes_by_arm = Counter()
    for row in records:
        record = runs[row["observation_id"]]
        program_path = artifact_dir(record) / "final_code.py"
        program = program_path.read_text(encoding="utf-8", errors="replace")
        all_status, details = deterministic_route_audit(program)
        route_category = str(details["route"])
        all_program_routes[route_category] += 1
        routes_by_arm[(row["arm"], route_category)] += 1
        observation_rows.append(
            {
                "observation_id": row["observation_id"],
                "source_case_id": row["source_case_id"],
                "pair_id": f"{row['source_case_id']}::repeat{row['repeat']}",
                "arm": row["arm"],
                "repeat": row["repeat"],
                "call_order": row["call_order"],
                "operator_family": row["operator_family"],
                "cell": row["cell"],
                "endpoint_pass": int(row["endpoint_pass"]),
                "failure_code": row["failure_code"],
                "rel_l2": row["rel_l2"],
                "program_route_status_deterministic": all_status,
                "program_route_category_deterministic": route_category,
                "route_audit_status": "complete_deterministic_screen",
            }
        )
    write_csv(
        SOURCE_DIR / "primary_observations.csv",
        list(observation_rows[0]),
        observation_rows,
    )

    pair_map: dict[str, dict[str, bool]] = defaultdict(dict)
    pair_source: dict[str, str] = {}
    for row in records:
        pair_id = f"{row['source_case_id']}::repeat{row['repeat']}"
        pair_map[pair_id][row["arm"]] = bool(row["endpoint_pass"])
        pair_source[pair_id] = row["source_case_id"]
    paired_counts = Counter()
    paired_rows = []
    for pair_id, outcomes in sorted(pair_map.items()):
        visible = outcomes["visible"]
        hidden = outcomes["hidden"]
        category = (
            "both_pass"
            if visible and hidden
            else "visible_only"
            if visible
            else "hidden_only"
            if hidden
            else "neither"
        )
        paired_counts[category] += 1
        paired_rows.append(
            {
                "pair_id": pair_id,
                "source_case_id": pair_source[pair_id],
                "visible_endpoint_pass": int(visible),
                "hidden_endpoint_pass": int(hidden),
                "paired_category": category,
            }
        )
    write_csv(SOURCE_DIR / "paired_endpoint_outcomes.csv", list(paired_rows[0]), paired_rows)

    primary_payload_path = PRIMARY_ANALYSIS / "matched_visibility_deterministic_audit.json"
    primary = json.loads(primary_payload_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(
        (DIAGNOSTIC_ANALYSIS / "corrected_case_sensitivity.json").read_text(encoding="utf-8")
    )
    arm_rows = []
    for arm in ("visible", "hidden"):
        summary = primary["arm_summary"][arm]
        arm_rows.append(
            {
                "arm": arm,
                "scheduled": summary["scheduled"],
                "solver_programs_deterministic": sum(
                    count for (route_arm, _), count in routes_by_arm.items() if route_arm == arm
                ),
                "endpoint_pass": summary["endpoint_pass"],
                "endpoint_fail": summary["scheduled"] - summary["endpoint_pass"],
                "direct_or_mixed_flags_among_endpoint_passes": summary["route_invalid"],
                "route_audit_status": "complete_deterministic_screen",
            }
        )
    write_csv(SOURCE_DIR / "arm_summary.csv", list(arm_rows[0]), arm_rows)

    route_rows = [
        {"arm": arm, "route_category_deterministic": route, "program_count": count}
        for (arm, route), count in sorted(routes_by_arm.items())
    ]
    write_csv(SOURCE_DIR / "program_route_categories.csv", list(route_rows[0]), route_rows)

    sensitivity_rows = []
    for stage, payload in (("frozen_primary", primary), ("corrected_case_diagnostic", diagnostic)):
        for arm in ("visible", "hidden"):
            summary = payload["arm_summary"][arm]
            sensitivity_rows.append(
                {
                    "analysis": stage,
                    "arm": arm,
                    "scheduled": summary["scheduled"],
                    "endpoint_pass": summary["endpoint_pass"],
                    "endpoint_pass_rate": summary["endpoint_pass"] / summary["scheduled"],
                    "interpretation_role": (
                        "primary" if stage == "frozen_primary" else "post_hoc_diagnostic"
                    ),
                }
            )
    write_csv(SOURCE_DIR / "corrected_case_sensitivity.csv", list(sensitivity_rows[0]), sensitivity_rows)

    endpoint_effect = primary["secondary_estimands"]["endpoint_pass"]
    visible_passes = primary["arm_summary"]["visible"]["endpoint_pass"]
    hidden_passes = primary["arm_summary"]["hidden"]["endpoint_pass"]
    total_invalid = sum(
        primary["arm_summary"][arm]["route_invalid"] for arm in ("visible", "hidden")
    )
    summary = {
        "analysis_status": primary["analysis_status"],
        "route_audit_status": "COMPLETE_DETERMINISTIC_SCREEN",
        "primary": {
            "scientific_cases": 64,
            "matched_pairs": 128,
            "calls": 256,
            "visible_endpoint_pass": visible_passes,
            "hidden_endpoint_pass": hidden_passes,
            "visible_minus_hidden_endpoint_pass_rate": endpoint_effect["visible_minus_hidden"],
            "case_cluster_bootstrap_ci95": endpoint_effect["ci95"],
            "paired_endpoint_counts": dict(paired_counts),
            "discordant_source_cases": sorted(
                {row["source_case_id"] for row in paired_rows if row["paired_category"] != "both_pass"}
            ),
            "all_program_route_categories_deterministic": dict(sorted(all_program_routes.items())),
            "direct_or_mixed_flags_among_254_endpoint_passes": total_invalid,
        },
        "corrected_case_diagnostic": diagnostic,
        "statistical_notes": {
            "endpoint_interval": "20,000 case-cluster bootstrap replicates; scientific case is the resampling unit",
            "route_audit": "deterministic complete-program screen; zero flags are reported descriptively without a prevalence interval",
            "diagnostic_inference": "No population interval: one post hoc corrected scientific case, two generations per arm",
        },
    }
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (SOURCE_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (SOURCE_DIR / "README.md").write_text(
        "# Source Data: matched reference visibility\n\n"
        "`primary_observations.csv` retains all 256 calls. "
        "`paired_endpoint_outcomes.csv` contains the 128 matched visible/hidden pairs. "
        "`arm_summary.csv` and `program_route_categories.csv` support the main figure. "
        "`corrected_case_sensitivity.csv` keeps the four post hoc diagnostic calls separate.\n\n"
        "Program-route fields are outputs of a complete deterministic program screen. They are "
        "reported as screened flags rather than independent human-adjudicated labels. The corrected-case "
        "analysis is diagnostic and is never pooled with the primary 256-call estimand.\n",
        encoding="utf-8",
    )

    ci_low, ci_high = endpoint_effect["ci95"]
    macro_text = "\n".join(
        [
            "% Auto-generated by scripts/build_matched_visibility_source_data.py",
            "\\newcommand{\\NMIMatchedScientificCaseCount}{64}",
            "\\newcommand{\\NMIMatchedPrimaryCallCount}{256}",
            "\\newcommand{\\NMIMatchedPairCount}{128}",
            "\\newcommand{\\NMIMatchedReviewItemCount}{254}",
            f"\\newcommand{{\\NMIMatchedVisibleEndpointPassCount}}{{{visible_passes}}}",
            f"\\newcommand{{\\NMIMatchedHiddenEndpointPassCount}}{{{hidden_passes}}}",
            f"\\newcommand{{\\NMIMatchedVisibleEndpointPassRate}}{{{pct(visible_passes / 128)}}}",
            f"\\newcommand{{\\NMIMatchedHiddenEndpointPassRate}}{{{pct(hidden_passes / 128)}}}",
            f"\\newcommand{{\\NMIMatchedEndpointDifferencePP}}{{{100 * endpoint_effect['visible_minus_hidden']:.2f}}}",
            f"\\newcommand{{\\NMIMatchedEndpointDifferenceCILowPP}}{{{100 * ci_low:.2f}}}",
            f"\\newcommand{{\\NMIMatchedEndpointDifferenceCIHighPP}}{{{100 * ci_high:.2f}}}",
            f"\\newcommand{{\\NMIMatchedBothPassPairCount}}{{{paired_counts['both_pass']}}}",
            f"\\newcommand{{\\NMIMatchedVisibleOnlyPairCount}}{{{paired_counts['visible_only']}}}",
            f"\\newcommand{{\\NMIMatchedHiddenOnlyPairCount}}{{{paired_counts['hidden_only']}}}",
            f"\\newcommand{{\\NMIMatchedNeitherPairCount}}{{{paired_counts['neither']}}}",
            f"\\newcommand{{\\NMIMatchedDeterministicInvalidFlagCount}}{{{total_invalid}}}",
            "\\newcommand{\\NMIMatchedCorrectedDiagnosticCount}{4}",
            "\\newcommand{\\NMIMatchedCorrectedDiagnosticPassCount}{4}",
            "\\newcommand{\\NMIMatchedCorrectedVisiblePassCount}{2}",
            "\\newcommand{\\NMIMatchedCorrectedHiddenPassCount}{2}",
            "\\newcommand{\\NMIMatchedDiscordantCaseID}{\\code{O0G1B0D0S0C0\\_s60002\\_203}}",
            "",
        ]
    )
    MACROS.parent.mkdir(parents=True, exist_ok=True)
    MACROS.write_text(macro_text, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
