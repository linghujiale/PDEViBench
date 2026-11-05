#!/usr/bin/env python3
"""Analysis of the analytic-field disclosure experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_ordinary_pass_routes import classify_route  # noqa: E402
from scripts.prepare_matched_visibility import END, START  # noqa: E402
from scripts.run_phase2_wave1 import classify_no_verdict  # noqa: E402

DESIGN = ROOT / "analysis" / "matched_visibility_20260825"
EXPERIMENT_ID = "matched_visibility_codex_gpt56sol_scipy_20260825"
ANALYSIS_DIR = ROOT / "analysis" / "phase2" / EXPERIMENT_ID
RESULTS_ROOT = ROOT / "results" / "batch"
LABELS = {
    "NUMERICAL_SOLVER": "route_valid",
    "DIRECT_REFERENCE_RECOVERY": "direct",
    "MIXED_SOLVER_REFERENCE_RECOVERY": "mixed",
    "INSUFFICIENT_EVIDENCE": "unresolved",
}
ROUTE_VALID_STATIC = {
    "sparse_fd_linear_solve",
    "time_stepping_solver",
    "nonlinear_or_iterative_solver",
    "coordinate_transform_or_polar_solver",
    "masked_domain_solver",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def latest_runs(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["case_id"])] = row
    return latest


def artifact_dir(record: dict[str, Any]) -> Path | None:
    value = (record.get("verdict_summary") or {}).get("artifact_dir")
    if not value:
        value = (record.get("agent_call") or {}).get("artifact_dir")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_verdict(record: dict[str, Any]) -> dict[str, Any] | None:
    path = artifact_dir(record)
    if path is None or not (path / "verdict.json").is_file():
        return None
    return json.loads((path / "verdict.json").read_text(encoding="utf-8"))


def completion_status(schedule: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {row["id"] for row in schedule}
    recorded = expected.intersection(runs)
    verdict_ids = {case_id for case_id in recorded if load_verdict(runs[case_id]) is not None}
    scientific_no_verdict_ids = {
        case_id
        for case_id in recorded - verdict_ids
        if (classify_no_verdict(runs[case_id]) or {}).get("rerun_recommended") is False
    }
    finalized_ids = verdict_ids | scientific_no_verdict_ids
    agent_status = Counter(str((runs[case_id].get("agent_call") or {}).get("status", "missing")) for case_id in recorded)
    return {
        "expected": len(expected),
        "ledger_records": len(recorded),
        "verdicts": len(verdict_ids),
        "scientific_no_verdicts": len(scientific_no_verdict_ids),
        "remaining_retriable_or_unlaunched": len(expected - finalized_ids),
        "agent_status_counts": dict(sorted(agent_status.items())),
        "complete_for_analysis": finalized_ids == expected,
    }


def endpoint_pass(verdict: dict[str, Any]) -> bool:
    stages = verdict.get("stages") or {}
    return all(bool((stages.get(name) or {}).get("pass")) for name in ("execution", "accuracy", "runtime"))


def deterministic_route_audit(code: str) -> tuple[str, dict[str, Any]]:
    details = classify_route(code)
    route = str(details["route"])
    exact = str(details["exact_formula_classifier"]["label"])
    if route in ROUTE_VALID_STATIC:
        status = "route_valid"
    elif exact == "MIXED_SOLVER_WITH_EXACT_HINTS":
        status = "mixed"
    elif route == "exact_or_analytic_reference_suspect" and exact in {"DIRECT_EXACT_FORMULA", "LIKELY_ANALYTIC_FORMULA"}:
        status = "direct"
    else:
        status = "unresolved"
    return status, details


def strip_treatment(statement: str) -> str:
    return re.sub(re.escape(START) + r".*?" + re.escape(END), "", statement, flags=re.DOTALL).rstrip()


def task_payload(case: dict[str, Any]) -> dict[str, Any]:
    spec = case["case_spec"]
    return {
        "pde_statement": strip_treatment(spec["pde"]["statement"]),
        "operator_family": spec["pde"]["operator_family"],
        "parameters": spec["pde"].get("parameters", {}),
        "domain": spec["domain"],
        "boundary_condition": spec["bc"],
        "initial_condition": spec.get("ic"),
        "evaluation_grid": spec["eval_grid"],
        "output_contract": spec["output"],
    }


def build_blinded_packet(records: list[dict[str, Any]], out: Path, seed: int) -> dict[str, Any]:
    passing = [row for row in records if row["endpoint_pass"]]
    rng = random.Random(seed)
    rng.shuffle(passing)
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for row in passing:
        blind_id = "MV-PV-" + hashlib.sha256(f"{seed}|{row['observation_id']}".encode()).hexdigest()[:12].upper()
        public.append({"blind_id": blind_id, "task": task_payload(row["case"]), "submitted_program": row["program"]})
        private.append(
            {
                "blind_id": blind_id,
                "observation_id": row["observation_id"],
                "source_case_id": row["source_case_id"],
                "arm": row["arm"],
                "repeat": row["repeat"],
                "block": row["block"],
                "endpoint_rel_l2": row["rel_l2"],
                "provisional_route": row["route_status"],
                "program_sha256": sha256_text(row["program"]),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    private_dir = out / "unblinding_key"
    private_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "review_items.jsonl", public)
    blank = [{"blind_id": row["blind_id"], "label": "", "confidence": "", "evidence_note": "", "reviewer_id": ""} for row in public]
    write_jsonl(out / "reviewer_A_labels.jsonl", blank)
    write_jsonl(out / "reviewer_B_labels.jsonl", blank)
    write_jsonl(out / "adjudication_labels.jsonl", blank)
    write_jsonl(private_dir / "blinding_key.jsonl", private)
    protocol = f"""# Matched-visibility blinded program review

Review all {len(public)} endpoint-passing programs independently. The packet omits
visibility arm, disclosure text, endpoint error, automated route and artifact path.
Follow the returned-field data flow and assign exactly one label:

- `NUMERICAL_SOLVER`
- `DIRECT_REFERENCE_RECOVERY`
- `MIXED_SOLVER_REFERENCE_RECOVERY`
- `INSUFFICIENT_EVIDENCE`

Analytic expressions used only for forcing, initial/boundary data or coefficients
are compatible with a numerical route. Directly evaluating the final evaluator
field is reference recovery. A solver whose result is overwritten or materially
post-corrected by such a field is mixed. Freeze reviewer A and B files before
opening the private key; adjudicate disagreements without revealing the arm.
"""
    (out / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    manifest = {
        "n_items": len(public),
        "seed": seed,
        "valid_labels": list(LABELS),
        "public_invariants": [
            "no visibility arm",
            "no disclosure text",
            "no endpoint error",
            "no provisional label",
            "no artifact path",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public_text = (out / "review_items.jsonl").read_text(encoding="utf-8").lower()
    for forbidden in ("matched_visibility", "randomized experimental condition", "endpoint_rel_l2", "provisional_route"):
        if forbidden in public_text:
            raise AssertionError(f"public packet leaked {forbidden}")
    return manifest


def load_human_labels(path: Path, private_key: Path) -> dict[str, str]:
    key = {row["blind_id"]: row for row in read_jsonl(private_key)}
    labels: dict[str, str] = {}
    for row in read_jsonl(path):
        label = str(row.get("label", ""))
        if label not in LABELS:
            raise ValueError(f"invalid or missing human label for {row.get('blind_id')}: {label!r}")
        labels[str(key[row["blind_id"]]["observation_id"])] = LABELS[label]
    return labels


def bootstrap_difference(records: list[dict[str, Any]], outcome: str, *, draws: int, seed: int) -> dict[str, Any]:
    cases = sorted({row["source_case_id"] for row in records})
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_case[row["source_case_id"]].append(row)

    def difference(sampled: list[str]) -> float:
        arm_values: dict[str, list[float]] = {"visible": [], "hidden": []}
        for case_id in sampled:
            for row in by_case[case_id]:
                arm_values[row["arm"]].append(float(row[outcome]))
        return float(np.mean(arm_values["visible"]) - np.mean(arm_values["hidden"]))

    estimate = difference(cases)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        sampled = [cases[index] for index in rng.integers(0, len(cases), size=len(cases))]
        boot[i] = difference(sampled)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"visible_minus_hidden": estimate, "ci95": [float(low), float(high)], "draws": draws, "cluster": "source_case_id"}


def build_records(schedule: list[dict[str, Any]], runs: dict[str, dict[str, Any]], human: dict[str, str] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in schedule:
        observation_id = case["id"]
        record = runs[observation_id]
        verdict = load_verdict(record)
        no_verdict = classify_no_verdict(record) if verdict is None else None
        if verdict is None and (no_verdict or {}).get("rerun_recommended") is not False:
            raise AssertionError(f"unfinalized retriable observation: {observation_id}")
        path = artifact_dir(record)
        code_path = None if path is None else path / "final_code.py"
        program = code_path.read_text(encoding="utf-8", errors="replace") if code_path and code_path.is_file() else ""
        is_endpoint = endpoint_pass(verdict) if verdict is not None else False
        if is_endpoint and not program:
            raise AssertionError(f"endpoint pass without program: {observation_id}")
        route_status, details = deterministic_route_audit(program) if is_endpoint else ("not_endpoint_pass", None)
        if human is not None and is_endpoint:
            route_status = human[observation_id]
        stages = (verdict or {}).get("stages") or {}
        rel_l2 = (stages.get("accuracy") or {}).get("rel_l2")
        meta = case["matched_visibility"]
        records.append(
            {
                "observation_id": observation_id,
                "source_case_id": meta["source_case_id"],
                "arm": meta["arm"],
                "repeat": int(meta["repeat"]),
                "block": int(meta["block"]),
                "call_order": int(meta["call_order"]),
                "cell": f"O{case['axes']['O']}G{case['axes']['G']}B{case['axes']['B']}",
                "operator_family": case["case_spec"]["pde"]["operator_family"],
                "endpoint_pass": is_endpoint,
                "route_status": route_status,
                "route_invalid": is_endpoint and route_status in {"direct", "mixed"},
                "route_valid_pass": is_endpoint and route_status == "route_valid",
                "direct": is_endpoint and route_status == "direct",
                "mixed": is_endpoint and route_status == "mixed",
                "unresolved": is_endpoint and route_status == "unresolved",
                "rel_l2": rel_l2,
                "failure_code": (verdict or {}).get("failure_code") or (no_verdict or {}).get("code"),
                "program": program,
                "static_details": details,
                "case": case,
            }
        )
    return records


def analyse(records: list[dict[str, Any]], *, draws: int, seed: int, final_human: bool) -> dict[str, Any]:
    arm_summary: dict[str, Any] = {}
    for arm in ("visible", "hidden"):
        rows = [row for row in records if row["arm"] == arm]
        route_counts = Counter(row["route_status"] for row in rows if row["endpoint_pass"])
        arm_summary[arm] = {
            "scheduled": len(rows),
            "endpoint_pass": sum(row["endpoint_pass"] for row in rows),
            "route_valid_pass": sum(row["route_valid_pass"] for row in rows),
            "route_invalid": sum(row["route_invalid"] for row in rows),
            "route_counts_among_endpoint_passes": dict(sorted(route_counts.items())),
        }
        for key in ("endpoint_pass", "route_valid_pass", "route_invalid"):
            arm_summary[arm][key + "_rate"] = arm_summary[arm][key] / len(rows)
    outcomes = {
        name: bootstrap_difference(records, name, draws=draws, seed=seed + i)
        for i, name in enumerate(("route_invalid", "endpoint_pass", "route_valid_pass", "direct", "mixed", "unresolved"))
    }
    return {
        "analysis_status": "FINAL_HUMAN" if final_human else "FINAL_DETERMINISTIC_COMPLETE_PROGRAM_AUDIT",
        "primary_estimand": outcomes["route_invalid"],
        "secondary_estimands": {key: value for key, value in outcomes.items() if key != "route_invalid"},
        "arm_summary": arm_summary,
        "n_scientific_cases": len({row["source_case_id"] for row in records}),
        "n_observations": len(records),
        "bootstrap_seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DESIGN)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--human-labels", type=Path, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args(argv)

    schedule = read_jsonl(args.design_dir / "scheduled_observations.jsonl")
    runs = latest_runs(args.analysis_dir / "runs.jsonl")
    status = completion_status(schedule, runs)
    status_path = args.analysis_dir / "matched_visibility_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))
    if not status["complete_for_analysis"]:
        print("Outcome analysis withheld until all 256 observations are final and no retriable infrastructure failures remain.")
        return 3

    human = None
    packet_dir = args.analysis_dir / "blinded_route_review"
    if args.human_labels is not None:
        human = load_human_labels(args.human_labels, packet_dir / "unblinding_key" / "blinding_key.jsonl")
    records = build_records(schedule, runs, human)
    if human is None:
        packet_manifest = build_blinded_packet(records, packet_dir, args.seed)
    else:
        manifest_path = packet_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("blinded-review manifest is missing")
        packet_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = analyse(records, draws=args.bootstrap_draws, seed=args.seed, final_human=human is not None)
    payload["blinded_review_packet"] = packet_manifest
    out = args.analysis_dir / ("matched_visibility_final_human.json" if human is not None else "matched_visibility_deterministic_audit.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
