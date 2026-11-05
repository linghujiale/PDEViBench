#!/usr/bin/env python3
"""Original versus prohibition prompt: factor coefficients and paired call transitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.regenerate_nmi_phase5_figure2 import PREDICTORS, percentile_interval  # noqa: E402

LEDGER = ROOT / "analysis/nmi_phase5_20260813/final_postreview_route_ledger.csv"
W1 = "phase2_wave1_codex_scipy_20260522T030958Z"
W1B = "phase2_wave1b_codex_scipy_no_exact_20260523T224200Z"
FIG2_EFFECTS = ROOT / "paper/source_data/fig_program_aware_score_and_robustness/fig2_factor_effects.csv"
OUT = ROOT / "paper/source_data/fig3_exact_formula_audit/prompt_transitions.json"
MACROS = ROOT / "Nature/generated/nmi_prompt_transition_macros.tex"
N_BOOTSTRAP = 10000
SEED = 20260811
INVALID = {"direct_recovery", "mixed_route"}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(LEDGER, dtype={"endpoint_pass": str})
    ledger["endpoint_pass"] = ledger["endpoint_pass"].eq("True")
    w1 = ledger[ledger["experiment_id"].eq(W1)].copy()
    w1b = ledger[ledger["experiment_id"].eq(W1B)].copy()
    for frame in (w1, w1b):
        assert len(frame) == 1450 and not frame.duplicated(["case_id", "run_idx"]).any()
    assert set(w1b.loc[w1b["endpoint_pass"], "route_status"]) == {"route_valid"}
    return w1, w1b


def per_case(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("case_id", sort=True)
    rows = grouped.agg(
        scheduled=("case_id", "size"),
        endpoint_pass=("endpoint_pass", "sum"),
        **{axis: (axis, "first") for axis in PREDICTORS},
    )
    rows["route_valid_pass"] = (
        frame.assign(rv=(frame["endpoint_pass"] & frame["route_status"].eq("route_valid")).astype(int))
        .groupby("case_id", sort=True)["rv"].sum()
    )
    assert rows["scheduled"].eq(5).all()
    return rows


def main() -> int:
    w1, w1b = load()
    a, b = per_case(w1), per_case(w1b)
    assert list(a.index) == list(b.index) and len(a) == 290
    assert (a[PREDICTORS].to_numpy() == b[PREDICTORS].to_numpy()).all()
    x = np.column_stack([np.ones(len(a))] + [a[name].to_numpy(dtype=float) for name in PREDICTORS])
    y = {
        "original_endpoint": (a["endpoint_pass"] / a["scheduled"]).to_numpy(dtype=float),
        "original_route_valid": (a["route_valid_pass"] / a["scheduled"]).to_numpy(dtype=float),
        "rerun": (b["endpoint_pass"] / b["scheduled"]).to_numpy(dtype=float),
    }
    coef = {k: np.linalg.lstsq(x, v, rcond=None)[0] for k, v in y.items()}
    rng = np.random.default_rng(SEED)
    boot = {k: np.empty((N_BOOTSTRAP, x.shape[1])) for k in y}
    for index in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(a), size=len(a))
        for k, v in y.items():
            boot[k][index] = np.linalg.lstsq(x[sample], v[sample], rcond=None)[0]
    names = ["intercept", *PREDICTORS]
    models = {}
    for pos, name in enumerate(names):
        models[name] = {
            **{k: float(coef[k][pos]) for k in y},
            "rerun_minus_original_route_valid": {
                "estimate": float(coef["rerun"][pos] - coef["original_route_valid"][pos]),
                "ci95": percentile_interval(boot["rerun"][:, pos] - boot["original_route_valid"][:, pos]),
            },
            "rerun_minus_original_endpoint": {
                "estimate": float(coef["rerun"][pos] - coef["original_endpoint"][pos]),
                "ci95": percentile_interval(boot["rerun"][:, pos] - boot["original_endpoint"][:, pos]),
            },
        }
    effects = pd.read_csv(FIG2_EFFECTS)
    for estimand, key in (("endpoint", "original_endpoint"), ("route_valid", "original_route_valid")):
        for name in names:
            stored = float(effects.loc[effects["estimand"].eq(estimand) & effects["term"].eq(name), "estimate"].iloc[0])
            assert abs(stored - models[name][key]) < 1e-12, (estimand, name)

    m = w1.set_index(["case_id", "run_idx"])
    n = w1b.set_index(["case_id", "run_idx"])
    assert set(m.index) == set(n.index)
    joined = m[["endpoint_pass", "route_status"]].join(n[["endpoint_pass"]], rsuffix="_rerun")
    original_class = np.where(
        ~joined["endpoint_pass"], "fail",
        np.where(joined["route_status"].eq("route_valid"), "numerical", np.where(joined["route_status"].isin(INVALID), "invalid", "other")),
    )
    assert "other" not in set(original_class)
    joined["original_class"] = original_class
    table = joined.groupby(["original_class", "endpoint_pass_rerun"]).size().unstack(fill_value=0)
    counts = {
        cls: {"rerun_pass": int(table.loc[cls].get(True, 0)), "rerun_fail": int(table.loc[cls].get(False, 0))}
        for cls in ("numerical", "invalid", "fail")
    }
    assert sum(v["rerun_pass"] + v["rerun_fail"] for v in counts.values()) == 1450
    inv_count = counts["invalid"]["rerun_pass"] + counts["invalid"]["rerun_fail"]
    payload = {
        "original_experiment": W1,
        "rerun_experiment": W1B,
        "model": "per-case pass fraction; OLS on intercept + O, G, B, D, S; 10,000 paired case bootstraps, seed 20260811",
        "coefficients": models,
        "transitions": counts,
        "original_invalid_count": inv_count,
        "original_invalid_rerun_pass_rate": counts["invalid"]["rerun_pass"] / inv_count,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def f3(v: float) -> str:
        return f"{0.0 if abs(v) < 0.0005 else v:.3f}"

    rows = []
    for name in PREDICTORS:
        r = models[name]
        rv, ep = r["rerun_minus_original_route_valid"], r["rerun_minus_original_endpoint"]
        rows.append(
            f"{name} & {f3(r['original_endpoint'])} & {f3(r['original_route_valid'])} & {f3(r['rerun'])} & "
            f"{f3(rv['estimate'])} [{f3(rv['ci95'][0])}, {f3(rv['ci95'][1])}] & "
            f"{f3(ep['estimate'])} [{f3(ep['ci95'][0])}, {f3(ep['ci95'][1])}] \\\\"
        )
    lines = [
        "% Generated by scripts/build_prompt_transitions.py; do not edit.",
        "\\newcommand{\\NMIPromptCoefficientRows}{%",
        *rows,
        "}",
        f"\\newcommand{{\\NMITransitionNumPass}}{{{counts['numerical']['rerun_pass']:,}}}",
        f"\\newcommand{{\\NMITransitionNumFail}}{{{counts['numerical']['rerun_fail']:,}}}",
        f"\\newcommand{{\\NMITransitionInvPass}}{{{counts['invalid']['rerun_pass']:,}}}",
        f"\\newcommand{{\\NMITransitionInvFail}}{{{counts['invalid']['rerun_fail']:,}}}",
        f"\\newcommand{{\\NMITransitionFailPass}}{{{counts['fail']['rerun_pass']:,}}}",
        f"\\newcommand{{\\NMITransitionFailFail}}{{{counts['fail']['rerun_fail']:,}}}",
        f"\\newcommand{{\\NMITransitionInvCount}}{{{inv_count:,}}}",
        f"\\newcommand{{\\NMIPromptRerunCoefO}}{{{f3(models['O']['rerun'])}}}",
        f"\\newcommand{{\\NMIPromptRerunCoefD}}{{{f3(models['D']['rerun'])}}}",
        f"\\newcommand{{\\NMIPromptOriginalEndpointCoefD}}{{{f3(models['D']['original_endpoint'])}}}",
        f"\\newcommand{{\\NMIPromptOriginalRouteValidCoefD}}{{{f3(models['D']['original_route_valid'])}}}",
        f"\\newcommand{{\\NMITransitionInvPassRate}}{{{100 * payload['original_invalid_rerun_pass_rate']:.2f}\\%}}",
    ]
    MACROS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
