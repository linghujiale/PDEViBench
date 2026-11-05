"""Route attribution of ordinary passing programs."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.exact_formula import classify_code, extract_evidence_lines  # noqa: E402


SOLVER_EVIDENCE_RE = re.compile(
    r"spsolve|splu|cg\(|gmres\(|bicgstab\(|scipy\.sparse|"
    r"diags\(|kron\(|lil_matrix|csr_matrix|assemble_matrix|assemble_vector|"
    r"np\.linalg\.solve|solve_ivp|root\(|fsolve|newton|picard|fixed.point",
    re.IGNORECASE,
)
TIME_CONTROL_RE = re.compile(
    r"(?m)^\s*(?:dt|delta_t|t_?final|final_time|n_?steps|num_steps|nt)\s*=",
    re.IGNORECASE,
)
TIME_LOOP_RE = re.compile(
    r"for\s+\w+\s+in\s+range\(\s*"
    r"(?:1\s*,\s*)?(?:n_?steps|num_steps|nt)(?:\s*\+\s*1)?\s*\)|"
    r"while\s+(?:t|time)\s*<\s*(?:t_?final|final_time)",
    re.IGNORECASE,
)
TIME_INTEGRATOR_RE = re.compile(r"solve_ivp|odeint|expm_multiply", re.IGNORECASE)
DIRECT_OUTPUT_FORMULA_RE = re.compile(
    r"(?m)^\s*(u|U|solution|solution_2d|sol)\s*=\s*"
    r"(?:np\.)?(sin|cos|exp|zeros_like|ones_like|where|\()|"
    r"np\.savez\([^\\n]*(?:np\.)?(sin|cos|exp)",
)


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    return {
        c["id"]: c
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for c in [json.loads(line)]
    }


def load_latest_runs(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[(r["case_id"], int(r["run_idx"]))] = r
    return list(latest.values())


def artifact_path(record: dict[str, Any]) -> Path | None:
    artifact = (record.get("verdict_summary") or {}).get("artifact_dir")
    if not artifact:
        return None
    p = Path(artifact)
    if not p.is_absolute():
        p = ROOT / p
    code = p / "final_code.py"
    return code if code.exists() else None


def cell_key(case: dict[str, Any]) -> str:
    ax = case.get("axes", {})
    return f"O{ax.get('O')}G{ax.get('G')}B{ax.get('B')}"


def classify_route(code: str) -> dict[str, Any]:
    exact = classify_code(code)
    lower = code.lower()
    route_tags: list[str] = []
    has_sparse_or_solver = bool(SOLVER_EVIDENCE_RE.search(code))
    has_time_stepping = bool(
        TIME_INTEGRATOR_RE.search(code)
        or (TIME_CONTROL_RE.search(code) and TIME_LOOP_RE.search(code))
    )
    has_direct_output_formula = bool(DIRECT_OUTPUT_FORMULA_RE.search(code))
    has_solver_evidence = has_sparse_or_solver or has_time_stepping or exact["has_solver_call"]
    if exact["loads_external_data"]:
        route = "file_load_or_external_data_suspect"
    elif (
        exact["label"] in {"DIRECT_EXACT_FORMULA", "LIKELY_ANALYTIC_FORMULA"}
        or has_direct_output_formula
    ) and not has_solver_evidence:
        route = "exact_or_analytic_reference_suspect"
    else:
        if re.search(r"spsolve|splu|cg\(|gmres\(|bicgstab\(|scipy\.sparse", code):
            route_tags.append("sparse_fd_linear_solve")
        if has_time_stepping:
            route_tags.append("time_stepping_solver")
        if re.search(r"newton|root\(|fixed.point|picard|nonlinear|fsolve", lower):
            route_tags.append("nonlinear_or_iterative_solver")
        if re.search(r"theta|atan2|polar|radius|r_in|r_out|meshgrid\(r|n_radial", lower):
            route_tags.append("coordinate_transform_or_polar_solver")
        if re.search(r"nan|mask|in_omega|hole|l_shape|square_with_hole", lower):
            route_tags.append("masked_domain_solver")
        if route_tags:
            priority = [
                "nonlinear_or_iterative_solver",
                "time_stepping_solver",
                "masked_domain_solver",
                "coordinate_transform_or_polar_solver",
                "sparse_fd_linear_solve",
            ]
            route = next(tag for tag in priority if tag in route_tags)
        elif exact["has_solver_call"]:
            route = "unclassified_solver_like"
        else:
            route = "unclassified_non_solver"
    return {
        "route": route,
        "route_tags": route_tags,
        "route_flags": {
            "has_solver_evidence": has_solver_evidence,
            "has_sparse_or_solver": has_sparse_or_solver,
            "has_time_stepping": has_time_stepping,
            "has_direct_output_formula": has_direct_output_formula,
            "has_exact_words": exact["has_exact_words"],
            "has_formula_tokens": exact["has_formula_tokens"],
        },
        "exact_formula_classifier": exact,
        "evidence_lines": extract_evidence_lines(code, max_lines=10),
    }


def select_sample(records: list[dict[str, Any]], cases: dict[str, dict[str, Any]], *, per_cell: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if (r.get("verdict_summary") or {}).get("case_pass") is not True:
            continue
        case = cases.get(r.get("case_id"))
        if case is None:
            continue
        groups[cell_key(case)].append(r)
    selected: list[dict[str, Any]] = []
    for _, group in sorted(groups.items()):
        ordered = list(group)
        rng.shuffle(ordered)
        selected.extend(ordered[:per_cell])
    return selected


def audit_experiment(
    *,
    experiment_id: str,
    cases_path: Path,
    analysis_root: Path,
    per_cell: int,
    seed: int,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    records = load_latest_runs(analysis_root / experiment_id / "runs.jsonl")
    selected = select_sample(records, cases, per_cell=per_cell, seed=seed)
    rows: list[dict[str, Any]] = []
    missing_code: list[dict[str, Any]] = []
    route_counts = Counter()
    route_by_cell: dict[str, Counter[str]] = defaultdict(Counter)

    for r in selected:
        case = cases[r["case_id"]]
        code_path = artifact_path(r)
        if code_path is None:
            missing_code.append({"case_id": r["case_id"], "run_idx": int(r["run_idx"])})
            continue
        code = code_path.read_text(encoding="utf-8", errors="replace")
        cls = classify_route(code)
        route_counts[cls["route"]] += 1
        route_by_cell[cell_key(case)][cls["route"]] += 1
        rows.append(
            {
                "case_id": r["case_id"],
                "run_idx": int(r["run_idx"]),
                "cell": cell_key(case),
                "operator_family": case["case_spec"]["pde"]["operator_family"],
                "geometry": case["case_spec"]["domain"]["spec"].get("shape"),
                "bc": case["case_spec"]["bc"]["kind"],
                "rel_l2": (r.get("verdict_summary") or {}).get("rel_l2"),
                "artifact_dir": (r.get("verdict_summary") or {}).get("artifact_dir"),
                **cls,
            }
        )
    return {
        "experiment_id": experiment_id,
        "cases_path": str(cases_path),
        "sampling": {"per_cell": per_cell, "seed": seed},
        "n_sampled": len(rows),
        "n_selected": len(selected),
        "missing_code": missing_code,
        "route_counts": dict(route_counts),
        "route_counts_by_cell": {cell: dict(cnt) for cell, cnt in sorted(route_by_cell.items())},
        "sample": rows,
    }


def write_markdown(payload: dict[str, Any], out: Path) -> None:
    lines = [
        "# Ordinary-Pass Computational Route Audit",
        "",
        f"Experiment: `{payload['experiment_id']}`",
        f"Sampled passing runs: {payload['n_sampled']}",
        "",
        "Static labels are deterministic review aids, not final human adjudication.",
        "",
        "## Route Counts",
        "",
        "| Route | Count |",
        "|---|---:|",
    ]
    for route, n in sorted(payload["route_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{route}` | {n} |")
    lines.extend(["", "## Route Counts by O/G/B Cell", "", "| Cell | Counts |", "|---|---|"])
    for cell, counts in payload["route_counts_by_cell"].items():
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"| `{cell}` | {summary} |")
    lines.extend(["", "## Review Queue", "", "| Case | Run | Cell | Route | rel_l2 | Evidence |", "|---|---:|---|---|---:|---|"])
    for row in payload["sample"]:
        evidence = "<br>".join(str(x).replace("|", "\\|") for x in row["evidence_lines"][:3])
        rel = row["rel_l2"]
        rel_s = f"{rel:.3e}" if isinstance(rel, (int, float)) else "NA"
        lines.append(
            f"| `{row['case_id']}` | {row['run_idx']} | `{row['cell']}` | "
            f"`{row['route']}` | {rel_s} | {evidence} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--cases", type=Path, default=ROOT / "benchmark" / "cases" / "pilot_phase1_8.jsonl")
    p.add_argument("--analysis-root", type=Path, default=ROOT / "analysis" / "phase2")
    p.add_argument("--per-cell", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260604)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args(argv)

    payload = audit_experiment(
        experiment_id=args.experiment_id,
        cases_path=args.cases,
        analysis_root=args.analysis_root,
        per_cell=args.per_cell,
        seed=args.seed,
    )
    out_json = args.out_json or (args.analysis_root / args.experiment_id / "ordinary_pass_route_audit.json")
    out_md = args.out_md or (args.analysis_root / args.experiment_id / "ordinary_pass_route_audit.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, out_md)
    print(json.dumps({
        "experiment_id": payload["experiment_id"],
        "n_selected": payload["n_selected"],
        "n_sampled": payload["n_sampled"],
        "missing_code": len(payload["missing_code"]),
        "route_counts": payload["route_counts"],
    }, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
