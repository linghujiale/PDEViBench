"""Rerun sampled near-zero primary-panel cases under the prompt that forbids constructing the exact solution.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.exact_formula import EXACT_SHORTCUT_LABELS  # noqa: E402
from scripts import run_phase2_wave1  # noqa: E402


def load_case_order(cases_path: Path) -> dict[str, int]:
    order: dict[str, int] = {}
    for idx, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            case = json.loads(line)
            order[case["id"]] = idx
    return order


def select_replay_cases(
    *,
    audit_json: Path,
    cases_path: Path,
    n_cases: int,
    labels: set[str],
) -> list[dict[str, Any]]:
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    order = load_case_order(cases_path)
    by_case: dict[str, dict[str, Any]] = {}
    for row in audit.get("near_zero_runs", []):
        label = row.get("classification", {}).get("label")
        if label not in labels:
            continue
        case_id = row["case_id"]
        entry = by_case.setdefault(
            case_id,
            {
                "case_id": case_id,
                "count": 0,
                "cell": row.get("cell"),
                "operator_family": row.get("operator_family"),
                "geometry_shape": row.get("geometry_shape"),
                "bc_kind": row.get("bc_kind"),
                "axes": row.get("axes", {}),
                "labels": Counter(),
                "source_order": order.get(case_id, 10**9),
            },
        )
        entry["count"] += 1
        entry["labels"][label] += 1

    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in by_case.values():
        key = (
            str(entry["operator_family"]),
            str(entry["geometry_shape"]),
            str(entry["bc_kind"]),
        )
        buckets[key].append(entry)

    bucket_order = [
        key
        for key, _ in Counter(
            {
                key: sum(item["count"] for item in values)
                for key, values in buckets.items()
            }
        ).most_common()
    ]
    for values in buckets.values():
        values.sort(key=lambda item: (-item["count"], item["source_order"], item["case_id"]))

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < n_cases:
        progressed = False
        for key in bucket_order:
            values = buckets[key]
            while values and values[0]["case_id"] in seen:
                values.pop(0)
            if not values:
                continue
            entry = values.pop(0)
            selected.append(entry)
            seen.add(entry["case_id"])
            progressed = True
            if len(selected) >= n_cases:
                break
        if not progressed:
            break

    for entry in selected:
        entry["labels"] = dict(entry["labels"])
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-experiment-id",
        default="phase2_wave1_codex_scipy_20260522T030958Z",
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmark" / "cases" / "pilot_phase1_8.jsonl")
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--analysis-root", type=Path, default=ROOT / "analysis" / "phase2")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "batch")
    parser.add_argument("--n-cases", type=int, default=24)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--runner", choices=("claude", "codex"), default="codex")
    parser.add_argument("--library", default="scipy")
    parser.add_argument("--eval-mode", choices=("host", "docker"), default="host")
    parser.add_argument("--wall-timeout", type=float, default=600.0)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--label", action="append", default=None,
                        help="Near-zero classification label to sample. May repeat.")
    args = parser.parse_args(argv)

    if args.n_cases <= 0 or args.n_runs <= 0:
        print("ERROR: --n-cases and --n-runs must be positive.", file=sys.stderr)
        return 2

    exp_id = args.experiment_id or (
        "phase2_no_exact_replay_codex_scipy_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    audit_json = args.audit_json or (
        args.analysis_root / args.source_experiment_id / "near_zero_audit.json"
    )
    labels = set(args.label or sorted(EXACT_SHORTCUT_LABELS))
    selected = select_replay_cases(
        audit_json=audit_json,
        cases_path=args.cases,
        n_cases=args.n_cases,
        labels=labels,
    )
    if not selected:
        print("ERROR: no replay cases selected.", file=sys.stderr)
        return 2

    out_root = args.analysis_root / exp_id
    out_root.mkdir(parents=True, exist_ok=True)
    selection_path = out_root / "selected_cases.json"
    selection_payload = {
        "experiment_id": exp_id,
        "source_experiment_id": args.source_experiment_id,
        "audit_json": str(audit_json),
        "selected_labels": sorted(labels),
        "n_cases_requested": args.n_cases,
        "n_cases_selected": len(selected),
        "n_runs": args.n_runs,
        "cases": selected,
    }
    selection_path.write_text(json.dumps(selection_payload, indent=2, ensure_ascii=False))

    wave_args = [
        "--experiment-id", exp_id,
        "--cases", str(args.cases),
        "--analysis-root", str(args.analysis_root),
        "--results-root", str(args.results_root),
        "--n-runs", str(args.n_runs),
        "--runner", args.runner,
        "--library", args.library,
        "--eval-mode", args.eval_mode,
        "--wall-timeout", str(args.wall_timeout),
        "--max-turns", str(args.max_turns),
    ]
    if args.resume:
        wave_args.append("--resume")
    if args.skip_eval:
        wave_args.append("--skip-eval")
    for entry in selected:
        wave_args.extend(["--case-id", entry["case_id"]])

    command = ["python3", "scripts/run_phase2_wave1.py", *wave_args]
    print(f"selected_cases={selection_path}")
    print(f"planned_calls={len(selected) * args.n_runs}")
    print("command:")
    print(" ".join(command))
    if args.dry_run:
        print("dry_run=true; not launching agent calls")
        return 0

    return run_phase2_wave1.main(wave_args)


if __name__ == "__main__":
    raise SystemExit(main())
