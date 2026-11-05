"""Failure-inclusive accuracy summary of one experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize as opt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_phase2_wave1 import classify_no_verdict, is_infra_failure  # noqa: E402

AXES = ("O", "G", "B", "D", "S", "C")


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    return {
        c["id"]: c
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for c in [json.loads(line)]
    }


def load_latest_runs(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    trailing_write_in_progress = bool(text) and not text.endswith(("\n", "\r"))
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if trailing_write_in_progress and index == len(lines) - 1:
                continue
            raise
        latest[(r["case_id"], int(r["run_idx"]))] = r
    return list(latest.values())


def record_class(record: dict[str, Any]) -> str:
    verdict = record.get("verdict_summary") or {}
    if verdict.get("case_pass") is True:
        return "pass"
    if verdict.get("case_pass") is False:
        failure = verdict.get("failure_code") or ""
        if str(failure).startswith("F-Acc"):
            return "F-Acc"
        if str(failure).startswith("F-Exec"):
            return "F-Exec"
        if str(failure).startswith("F-Time"):
            return "F-Time"
        if str(failure).startswith("F-Method"):
            return "F-Method"
        return "fail_other"
    classification = classify_no_verdict(record)
    if classification is not None:
        return str(classification["code"])
    status = (record.get("agent_call") or {}).get("status")
    if status:
        return f"AGENT_{str(status).upper()}"
    return "NO_VERDICT_UNKNOWN"


def usable_for_pass_model(record: dict[str, Any]) -> bool:
    """Keep scientific non-pass outcomes; drop retriable infrastructure."""
    if record.get("verdict_summary"):
        return True
    classification = classify_no_verdict(record)
    if classification is None:
        return False
    return not bool(classification.get("rerun_recommended", True))


def design_row(case: dict[str, Any], axes: tuple[str, ...]) -> list[float]:
    ax = case.get("axes", {})
    return [1.0] + [float(ax.get(a, 0)) for a in axes]


def drop_constant_axes(rows: list[list[float]], axes: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(rows, dtype=float)
    keep = [0]
    used: list[str] = []
    for i, axis in enumerate(axes, start=1):
        if float(np.std(arr[:, i])) > 0:
            keep.append(i)
            used.append(axis)
    return arr[:, keep], used


def fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def nll(beta: np.ndarray) -> float:
        z = np.clip(X @ beta, -40, 40)
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    def grad(beta: np.ndarray) -> np.ndarray:
        z = np.clip(X @ beta, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        return X.T @ (p - y)

    beta0 = np.zeros(X.shape[1], dtype=float)
    res = opt.minimize(nll, beta0, jac=grad, method="BFGS")
    beta = np.asarray(res.x, dtype=float)
    z = np.clip(X @ beta, -40, 40)
    p = 1.0 / (1.0 + np.exp(-z))
    W = p * (1.0 - p)
    hess = X.T @ (X * W[:, None])
    try:
        cov = np.linalg.pinv(hess)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        se = np.full_like(beta, np.nan)
    return beta, se


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se


def coef_table(beta: np.ndarray, se: np.ndarray, axes_used: list[str], *, logistic: bool) -> dict[str, Any]:
    names = ["intercept"] + axes_used
    out: dict[str, Any] = {}
    for name, b, s in zip(names, beta, se):
        row: dict[str, Any] = {"coef": float(b), "se": float(s)}
        if math.isfinite(float(s)):
            row["ci95"] = [float(b - 1.96 * s), float(b + 1.96 * s)]
        if logistic and name != "intercept":
            row["odds_ratio"] = _safe_exp(float(b))
            if math.isfinite(float(s)):
                row["odds_ratio_ci95"] = [
                    _safe_exp(float(b - 1.96 * s)),
                    _safe_exp(float(b + 1.96 * s)),
                ]
        out[name] = row
    return out


def _safe_exp(x: float) -> float:
    if x > 700:
        return float("inf")
    if x < -745:
        return 0.0
    return float(math.exp(x))


def analyze(*, experiment_id: str, cases_path: Path, analysis_root: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    runs_path = analysis_root / experiment_id / "runs.jsonl"
    records = load_latest_runs(runs_path)

    model_rows: list[list[float]] = []
    model_y: list[float] = []
    class_counts = Counter()
    cell_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_case_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        case = cases.get(record.get("case_id"))
        if case is None:
            continue
        cls = record_class(record)
        class_counts[cls] += 1
        ax = case["axes"]
        cell = f"O{ax.get('O')}G{ax.get('G')}B{ax.get('B')}"
        cell_counts[cell][cls] += 1
        per_case_records[record["case_id"]].append(record)
        if usable_for_pass_model(record):
            model_rows.append(design_row(case, AXES))
            model_y.append(1.0 if cls == "pass" else 0.0)

    X_logit, axes_used_logit = drop_constant_axes(model_rows, AXES)
    y_logit = np.asarray(model_y, dtype=float)
    beta_logit, se_logit = fit_logistic(X_logit, y_logit)

    case_rows: list[list[float]] = []
    case_y: list[float] = []
    case_n: list[int] = []
    for case_id, group in sorted(per_case_records.items()):
        usable = [r for r in group if usable_for_pass_model(r)]
        if not usable:
            continue
        case = cases[case_id]
        n = len(usable)
        passes = sum(1 for r in usable if record_class(r) == "pass")
        case_rows.append(design_row(case, AXES))
        case_y.append(passes / n)
        case_n.append(n)
    X_case, axes_used_case = drop_constant_axes(case_rows, AXES)
    y_case = np.asarray(case_y, dtype=float)
    beta_case, se_case = fit_ols(X_case, y_case)

    return {
        "experiment_id": experiment_id,
        "cases_path": str(cases_path),
        "n_latest_records": len(records),
        "record_class_counts": dict(class_counts),
        "n_records_in_pass_model": int(len(y_logit)),
        "pass_rate_model": {
            "outcome": "record-level pass vs scientific/deterministic non-pass",
            "axes_used": axes_used_logit,
            "coefficients": coef_table(beta_logit, se_logit, axes_used_logit, logistic=True),
        },
        "per_case_pass_fraction_model": {
            "n_cases": int(len(y_case)),
            "mean_records_per_case": float(np.mean(case_n)) if case_n else None,
            "axes_used": axes_used_case,
            "coefficients": coef_table(beta_case, se_case, axes_used_case, logistic=False),
        },
        "verdict_taxonomy_by_OGB_cell": {
            cell: dict(counter) for cell, counter in sorted(cell_counts.items())
        },
    }


def write_markdown(payload: dict[str, Any], out: Path) -> None:
    logit = payload["pass_rate_model"]["coefficients"]
    case = payload["per_case_pass_fraction_model"]["coefficients"]
    lines = [
        "# Failure-Inclusive Frontier Sensitivity",
        "",
        f"Experiment: `{payload['experiment_id']}`",
        f"Latest records: {payload['n_latest_records']}",
        "",
        "This analysis complements the manuscript's conditional passing-case rel_l2 slope. It keeps deterministic scientific non-pass outcomes in view rather than treating them as missing data.",
        "",
        "## Record Taxonomy",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for cls, n in sorted(payload["record_class_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{cls}` | {n} |")
    lines.extend(["", "## Record-Level Logistic Pass Model", "", "| Term | Coef | OR | 95% OR CI |", "|---|---:|---:|---|"])
    for term, row in logit.items():
        if term == "intercept":
            lines.append(f"| `{term}` | {row['coef']:+.3f} | NA | NA |")
        else:
            ci = row.get("odds_ratio_ci95", [None, None])
            lines.append(
                f"| `{term}` | {row['coef']:+.3f} | {row['odds_ratio']:.3f} | "
                f"{ci[0]:.3f} to {ci[1]:.3f} |"
            )
    lines.extend(["", "## Per-Case Pass-Fraction OLS", "", "| Term | Coef | 95% CI |", "|---|---:|---|"])
    for term, row in case.items():
        ci = row.get("ci95", [None, None])
        lines.append(f"| `{term}` | {row['coef']:+.3f} | {ci[0]:+.3f} to {ci[1]:+.3f} |")
    lines.extend(["", "## Verdict Taxonomy by O/G/B Cell", "", "| Cell | Counts |", "|---|---|"])
    for cell, counts in payload["verdict_taxonomy_by_OGB_cell"].items():
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"| `{cell}` | {summary} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--cases", type=Path, default=ROOT / "benchmark" / "cases" / "pilot_phase1_8.jsonl")
    p.add_argument("--analysis-root", type=Path, default=ROOT / "analysis" / "phase2")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args(argv)

    payload = analyze(
        experiment_id=args.experiment_id,
        cases_path=args.cases,
        analysis_root=args.analysis_root,
    )
    out_json = args.out_json or (args.analysis_root / args.experiment_id / "failure_inclusive_frontier.json")
    out_md = args.out_md or (args.analysis_root / args.experiment_id / "failure_inclusive_frontier.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, out_md)
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
