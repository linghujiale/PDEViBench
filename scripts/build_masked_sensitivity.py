#!/usr/bin/env python3
"""Sensitivity of the primary-panel statistics to the 20 masked-domain cases."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.regenerate_nmi_phase5_figure2 import PREDICTORS, fit_case_models  # noqa: E402

LEDGER = ROOT / "analysis/nmi_phase5_20260813/final_postreview_route_ledger.csv"
W1 = "phase2_wave1_codex_scipy_20260522T030958Z"
W1B = "phase2_wave1b_codex_scipy_no_exact_20260523T224200Z"
STATS = ROOT / "analysis/nmi_next_round_20260826/statistical_reanalysis.json"
FIG2_EFFECTS = ROOT / "paper/source_data/fig_program_aware_score_and_robustness/fig2_factor_effects.csv"
PROMPT_SUMMARY = ROOT / "paper/source_data/fig3_exact_formula_audit/prompt_comparison_summary.json"
OUT_DIR = ROOT / "paper/source_data/masked_domain_sensitivity"
MACROS = ROOT / "Nature/generated/nmi_masked_sensitivity_macros.tex"
INVALID = {"direct_recovery", "mixed_route"}


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    ledger = pd.read_csv(LEDGER, dtype={"endpoint_pass": str})
    ledger["endpoint_pass"] = ledger["endpoint_pass"].eq("True")
    w1 = ledger[ledger["experiment_id"].eq(W1)].copy()
    w1b = ledger[ledger["experiment_id"].eq(W1B)].copy()
    assert len(w1) == len(w1b) == 1450
    masked_calls = w1.groupby("case_id")["tau_acc"].apply(lambda s: int(s.isna().sum()))
    assert set(masked_calls.unique()) == {0, 5}
    masked = sorted(masked_calls[masked_calls.eq(5)].index)
    assert len(masked) == 20
    assert not w1[w1["case_id"].isin(masked)]["endpoint_pass"].any()
    return w1, w1b, masked


def gap_bootstrap(w1: pd.DataFrame) -> dict[str, float]:
    """Same estimator, seed and draw count as run_nmi_next_round_reanalysis.primary_statistics."""
    by_case: dict[str, list[int]] = defaultdict(list)
    for row in w1.itertuples(index=False):
        by_case[row.case_id].append(int(bool(row.endpoint_pass) and row.route_status in INVALID))
    values = np.asarray([sum(by_case[c]) / 5.0 for c in sorted(by_case)], dtype=float)
    rng = np.random.default_rng(20260826)
    boot = np.empty(20000, dtype=float)
    for i in range(len(boot)):
        boot[i] = rng.choice(values, size=len(values), replace=True).mean()
    return {"estimate": float(values.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])]}


def case_rows(w1: pd.DataFrame) -> pd.DataFrame:
    grouped = w1.groupby("case_id", sort=True)
    rows = grouped.agg(
        scheduled=("case_id", "size"),
        endpoint_pass=("endpoint_pass", "sum"),
        **{axis: (axis, "first") for axis in PREDICTORS},
    ).reset_index()
    valid = (
        w1.assign(rv=w1["route_status"].eq("route_valid").astype(int)).groupby("case_id", sort=True)["rv"].sum()
    )
    rows["route_valid_pass"] = rows["case_id"].map(valid).astype(int)
    rows["endpoint_pass_fraction"] = rows["endpoint_pass"] / rows["scheduled"]
    rows["route_valid_pass_fraction"] = rows["route_valid_pass"] / rows["scheduled"]
    return rows


def prompt_differences(w1: pd.DataFrame, w1b: pd.DataFrame) -> dict[str, object]:
    """Same estimator as scripts/build_prompt_comparison_source.py."""
    per_case = {W1: defaultdict(lambda: [0, 0, 0]), W1B: defaultdict(lambda: [0, 0, 0])}
    for frame, experiment in ((w1, W1), (w1b, W1B)):
        for row in frame.itertuples(index=False):
            c = per_case[experiment][row.case_id]
            c[2] += 1
            c[0] += bool(row.endpoint_pass)
            c[1] += row.route_status == "route_valid"
    cases = sorted(per_case[W1])
    assert cases == sorted(per_case[W1B])
    a = np.array([per_case[W1][c] for c in cases], float)
    b = np.array([per_case[W1B][c] for c in cases], float)

    def diffs(a, b):
        n = a[:, 2].sum()
        return [100 * (b[:, 0].sum() - a[:, 0].sum()) / n, 100 * (b[:, 1].sum() - a[:, 1].sum()) / n]

    rng = np.random.default_rng(20261029)
    n_cases = len(cases)
    boot = np.array([diffs(a[i], b[i]) for i in (rng.integers(0, n_cases, n_cases) for _ in range(10000))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    point = diffs(a, b)
    return {
        "endpoint_diff_pp": point[0], "endpoint_diff_ci95": [float(lo[0]), float(hi[0])],
        "route_valid_diff_pp": point[1], "route_valid_diff_ci95": [float(lo[1]), float(hi[1])],
        "prohibition_endpoint": int(b[:, 0].sum()), "prohibition_route_valid": int(b[:, 1].sum()),
        "calls": int(a[:, 2].sum()),
    }


def analyse(w1: pd.DataFrame, w1b: pd.DataFrame) -> dict[str, object]:
    endpoint = int(w1["endpoint_pass"].sum())
    valid = int((w1["endpoint_pass"] & w1["route_status"].eq("route_valid")).sum())
    calls = len(w1)
    rows = case_rows(w1)
    return {
        "cases": int(w1["case_id"].nunique()),
        "calls": calls,
        "endpoint_passes": endpoint,
        "route_valid_passes": valid,
        "endpoint_rate": endpoint / calls,
        "route_valid_rate": valid / calls,
        "gap": gap_bootstrap(w1),
        "factor_models": fit_case_models(rows, 10000, 20260811),
        "prompt": prompt_differences(w1, w1b),
    }


def check_full_panel(full: dict[str, object]) -> None:
    stats = json.loads(STATS.read_text(encoding="utf-8"))["primary_score_inflation"]
    assert abs(stats["confirmed_score_inflation"] - full["gap"]["estimate"]) < 1e-12
    assert np.allclose(stats["case_cluster_bootstrap_95ci"], full["gap"]["ci95"], atol=1e-12)
    with FIG2_EFFECTS.open(newline="", encoding="utf-8") as handle:
        effects = list(csv.DictReader(handle))
    for row in effects:
        stats_row = full["factor_models"][row["estimand"]][row["term"]]
        assert abs(float(row["estimate"]) - stats_row["estimate"]) < 1e-12, row
        assert abs(float(row["ci95_low"]) - stats_row["ci95"][0]) < 1e-12, row
    prompt = json.loads(PROMPT_SUMMARY.read_text(encoding="utf-8"))
    assert round(full["prompt"]["route_valid_diff_pp"], 2) == prompt["route_valid_diff_pp"]
    assert [round(x, 2) for x in full["prompt"]["route_valid_diff_ci95"]] == prompt["route_valid_diff_ci95"]
    assert round(full["prompt"]["endpoint_diff_pp"], 2) == prompt["endpoint_diff_pp"]


def f3(x: float) -> str:
    return f"{0.0 if abs(x) < 0.0005 else x:.3f}"


def table_rows(full: dict, sub: dict) -> list[str]:
    def rate(r, kind):
        n = r[f"{kind}_passes"]
        return f"{n:,}/{r['calls']:,} ({100 * n / r['calls']:.2f}\\%)"

    def gap(r):
        g = r["gap"]
        return f"{100 * g['estimate']:.2f} [{100 * g['ci95'][0]:.2f}, {100 * g['ci95'][1]:.2f}]"

    def coef(r, axis):
        m = r["factor_models"]
        return f"{f3(m['endpoint'][axis]['estimate'])} / {f3(m['route_valid'][axis]['estimate'])}"

    def diff(r, axis):
        d = r["factor_models"]["route_valid_minus_endpoint"][axis]
        return f"{f3(d['estimate'])} [{f3(d['ci95'][0])}, {f3(d['ci95'][1])}]"

    def prompt(r, kind):
        p = r["prompt"]
        return f"{p[f'{kind}_diff_pp']:.2f} [{p[f'{kind}_diff_ci95'][0]:.2f}, {p[f'{kind}_diff_ci95'][1]:.2f}]"

    lines = [
        f"Endpoint pass rate & {rate(full, 'endpoint')} & {rate(sub, 'endpoint')} \\\\",
        f"Route-valid pass rate & {rate(full, 'route_valid')} & {rate(sub, 'route_valid')} \\\\",
        f"Endpoint minus route-valid gap, pp [95\\% CI] & {gap(full)} & {gap(sub)} \\\\",
    ]
    for axis in PREDICTORS:
        lines.append(f"{axis} coefficient, endpoint / route-valid & {coef(full, axis)} & {coef(sub, axis)} \\\\")
    for axis in PREDICTORS:
        lines.append(f"{axis} coefficient, route-valid minus endpoint [95\\% CI] & {diff(full, axis)} & {diff(sub, axis)} \\\\")
    lines.append(f"Prohibition minus original, endpoint, pp [95\\% CI] & {prompt(full, 'endpoint')} & {prompt(sub, 'endpoint')} \\\\")
    lines.append(f"Prohibition minus original, route-valid, pp [95\\% CI] & {prompt(full, 'route_valid')} & {prompt(sub, 'route_valid')} \\\\")
    return lines


def main() -> int:
    w1, w1b, masked = load()
    full = analyse(w1, w1b)
    check_full_panel(full)
    keep1 = w1[~w1["case_id"].isin(masked)]
    keep1b = w1b[~w1b["case_id"].isin(masked)]
    sub = analyse(keep1, keep1b)
    assert sub["cases"] == 270 and sub["calls"] == 1350
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "masked_domain_cases": masked,
        "masked_domain_rule": "primary-panel cases whose ledger tau_acc is empty; all 100 calls ended at the mask check",
        "all_290_cases": full,
        "without_masked_270_cases": sub,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = sub
    lines = [
        "% Generated by scripts/build_masked_sensitivity.py; do not edit.",
        f"\\newcommand{{\\NMIMaskedCaseCount}}{{{len(masked)}}}",
        f"\\newcommand{{\\NMIMaskedCallCount}}{{{len(masked) * 5}}}",
        f"\\newcommand{{\\NMIUnmaskedCaseCount}}{{{s['cases']}}}",
        f"\\newcommand{{\\NMIUnmaskedCallCount}}{{{s['calls']:,}}}",
        f"\\newcommand{{\\NMIUnmaskedEndpointRate}}{{{100 * s['endpoint_rate']:.2f}\\%}}",
        f"\\newcommand{{\\NMIUnmaskedRouteValidRate}}{{{100 * s['route_valid_rate']:.2f}\\%}}",
        f"\\newcommand{{\\NMIUnmaskedGapPP}}{{{100 * s['gap']['estimate']:.2f}}}",
        f"\\newcommand{{\\NMIUnmaskedGapCILow}}{{{100 * s['gap']['ci95'][0]:.2f}}}",
        f"\\newcommand{{\\NMIUnmaskedGapCIHigh}}{{{100 * s['gap']['ci95'][1]:.2f}}}",
        f"\\newcommand{{\\NMIUnmaskedOperatorDifference}}{{{f3(s['factor_models']['route_valid_minus_endpoint']['O']['estimate'])}}}",
        f"\\newcommand{{\\NMIUnmaskedOperatorDifferenceCILow}}{{{f3(s['factor_models']['route_valid_minus_endpoint']['O']['ci95'][0])}}}",
        f"\\newcommand{{\\NMIUnmaskedOperatorDifferenceCIHigh}}{{{f3(s['factor_models']['route_valid_minus_endpoint']['O']['ci95'][1])}}}",
        f"\\newcommand{{\\NMIUnmaskedPromptRouteValidDiffPP}}{{{s['prompt']['route_valid_diff_pp']:.2f}}}",
        f"\\newcommand{{\\NMIUnmaskedPromptEndpointDiffPP}}{{{s['prompt']['endpoint_diff_pp']:.2f}}}",
        "\\newcommand{\\NMIMaskedSensitivityRows}{%",
        *table_rows(full, sub),
        "}",
    ]
    MACROS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
