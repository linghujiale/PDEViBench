#!/usr/bin/env python3
"""Score the naturalistic panel from its complete run archives."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FAMILY_LABELS = {
    "heterogeneous_diffusion": "Heterogeneous\ndiffusion",
    "advection_diffusion_reaction": "ADR",
    "lshape_poisson": "L-shaped\ndomain",
    "coupled_elliptic": "Coupled\ntwo-field",
}
AGENT_COLORS = {"Codex GPT-5.5": "#2864dc", "Claude Opus 4.7": "#df6b21"}


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def _load_agent(
    *,
    experiment_root: Path,
    agent_name: str,
    provider_model: str,
    agent_label: str,
    cases: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        family = case["case_spec"]["pde"]["operator_family"]
        base = experiment_root / case_id / agent_name / provider_model
        for run_idx in (0, 1):
            verdict_path = base / f"run{run_idx}" / "verdict.json"
            if not verdict_path.is_file():
                continue
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            stages = verdict.get("stages", {})
            accuracy = stages.get("accuracy", {})
            runtime = stages.get("runtime", {})
            method = stages.get("method", {})
            execution_pass = bool(stages.get("execution", {}).get("pass"))
            accuracy_pass = bool(accuracy.get("pass"))
            runtime_pass = bool(runtime.get("pass"))
            method_pass = bool(method.get("pass"))
            endpoint_pass = execution_pass and accuracy_pass and runtime_pass
            rel_l2 = accuracy.get("rel_l2")
            rows.append({
                "agent": agent_label,
                "case_id": case_id,
                "run_idx": run_idx,
                "family": family,
                "case_pass": bool(verdict.get("case_pass")),
                "endpoint_pass": endpoint_pass,
                "execution_pass": execution_pass,
                "accuracy_pass": accuracy_pass,
                "runtime_pass": runtime_pass,
                "method_pass": method_pass,
                "failure_code": str(verdict.get("failure_code")),
                "rel_l2": float(rel_l2) if isinstance(rel_l2, (int, float)) else None,
                "tau_acc": float(case["thresholds"]["tau_acc"]),
                "normalized_rel_l2": (
                    float(rel_l2) / float(case["thresholds"]["tau_acc"])
                    if isinstance(rel_l2, (int, float)) and math.isfinite(float(rel_l2))
                    else None
                ),
                "wall_seconds_mean": runtime.get("wall_seconds_mean"),
                "artifact_dir": str(verdict_path.parent),
            })
    return rows


def _case_bootstrap_rate(
    rows: list[dict[str, Any]],
    *,
    field: str,
    reps: int = 10000,
    seed: int = 20260811,
) -> tuple[float, float, float]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(float(bool(row[field])))
    ids = sorted(by_case)
    values = np.asarray([np.mean(by_case[c]) for c in ids], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        boot[i] = float(rng.choice(values, size=len(values), replace=True).mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def _paired_case_bootstrap_delta(
    rows: list[dict[str, Any]], *, field: str, reps: int = 10000
) -> tuple[float, float, float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["agent"])].append(float(bool(row[field])))
    agents = sorted({row["agent"] for row in rows})
    if agents != ["Claude Opus 4.7", "Codex GPT-5.5"]:
        raise ValueError(f"unexpected agents: {agents}")
    cases = sorted({row["case_id"] for row in rows})
    delta = np.asarray([
        np.mean(grouped[(case_id, "Codex GPT-5.5")])
        - np.mean(grouped[(case_id, "Claude Opus 4.7")])
        for case_id in cases
    ])
    rng = np.random.default_rng(20260812)
    boot = np.asarray([
        rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(reps)
    ])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(delta.mean()), float(lo), float(hi)


def _summary(
    rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "frozen_external_naturalistic_validation",
        "claim_status": "confirmatory_external_validation",
        "expected_records": 128,
        "observed_records": len(rows),
        "complete": len(rows) == 128,
        "agents": {},
        "paired_codex_minus_claude": {},
    }
    if cases:
        gaps = {
            case_id: float(case["reference"]["convergence_check"]["gap_relative_l2"])
            for case_id, case in cases.items()
        }
        payload["reference_convergence"] = {
            "n_cases": len(gaps),
            "median_gap_relative_l2": float(np.median(list(gaps.values()))),
            "maximum_gap_relative_l2": float(max(gaps.values())),
            "acceptance_limit_relative_l2": 0.05,
            "all_below_acceptance_limit": all(value < 0.05 for value in gaps.values()),
        }
    for agent in ("Codex GPT-5.5", "Claude Opus 4.7"):
        arm = [row for row in rows if row["agent"] == agent]
        agent_out: dict[str, Any] = {
            "records": len(arm),
            "case_pass": sum(row["case_pass"] for row in arm),
            "endpoint_pass": sum(row["endpoint_pass"] for row in arm),
            "method_pass": sum(row["method_pass"] for row in arm),
            "endpoint_pass_reclassified_by_method": sum(
                row["endpoint_pass"] and not row["case_pass"] for row in arm
            ),
            "failure_codes": dict(Counter(row["failure_code"] for row in arm)),
            "families": {},
        }
        for field in ("case_pass", "endpoint_pass"):
            est, lo, hi = _case_bootstrap_rate(arm, field=field)
            agent_out[f"{field}_rate"] = est
            agent_out[f"{field}_case_bootstrap_95ci"] = [lo, hi]
        for family in FAMILY_LABELS:
            subset = [row for row in arm if row["family"] == family]
            est, lo, hi = _case_bootstrap_rate(subset, field="case_pass")
            agent_out["families"][family] = {
                "records": len(subset),
                "case_pass": sum(row["case_pass"] for row in subset),
                "case_pass_rate": est,
                "case_bootstrap_95ci": [lo, hi],
            }
        agreement = 0
        by_case: dict[str, list[bool]] = defaultdict(list)
        for row in arm:
            by_case[row["case_id"]].append(row["case_pass"])
        for values in by_case.values():
            if len(values) == 2 and values[0] == values[1]:
                agreement += 1
        agent_out["repeat_agreement_cases"] = agreement
        agent_out["repeat_agreement_rate"] = agreement / 32.0
        payload["agents"][agent] = agent_out
    for field in ("case_pass", "endpoint_pass"):
        est, lo, hi = _paired_case_bootstrap_delta(rows, field=field)
        payload["paired_codex_minus_claude"][field] = {
            "difference": est,
            "case_bootstrap_95ci": [lo, hi],
        }
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    stem: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.04, 1.0])
    agents = ["Codex GPT-5.5", "Claude Opus 4.7"]

    ax = fig.add_subplot(gs[0, 0])
    families = list(FAMILY_LABELS)
    x = np.arange(len(families))
    width = 0.36
    for j, agent in enumerate(agents):
        values = [summary["agents"][agent]["families"][f]["case_pass_rate"] for f in families]
        cis = [summary["agents"][agent]["families"][f]["case_bootstrap_95ci"] for f in families]
        yerr = np.asarray([[v-ci[0] for v,ci in zip(values,cis)], [ci[1]-v for v,ci in zip(values,cis)]])
        ax.bar(x + (j - 0.5) * width, np.asarray(values) * 100, width,
               color=AGENT_COLORS[agent], label=agent, edgecolor="white", linewidth=0.5,
               yerr=yerr*100, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x, [FAMILY_LABELS[f] for f in families])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Composite endpoint pass (%)")
    ax.set_title("a  Formula-blind viability by PDE family", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    metric_names = ["Endpoint", "Route-gated"]
    for j, agent in enumerate(agents):
        vals = [
            summary["agents"][agent]["endpoint_pass_rate"],
            summary["agents"][agent]["case_pass_rate"],
        ]
        ax.bar(np.arange(2) + (j - 0.5) * width, np.asarray(vals) * 100, width,
               color=AGENT_COLORS[agent], edgecolor="white", linewidth=0.5)
        for k, value in enumerate(vals):
            ax.text(k + (j - 0.5) * width, value * 100 + 1.5, f"{value*100:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(2), metric_names)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Pass (%)")
    ax.set_title("b  Endpoint and route-gated verdicts", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(18)
    positions = []
    tick_labels = []
    pos = 0
    for family in families:
        for agent in agents:
            vals = [r["normalized_rel_l2"] for r in rows
                    if r["family"] == family and r["agent"] == agent
                    and r["normalized_rel_l2"] is not None and r["normalized_rel_l2"] > 0]
            logvals = np.log10(np.asarray(vals)) if vals else np.asarray([])
            jitter = rng.uniform(-0.12, 0.12, size=len(logvals))
            ax.scatter(np.full(len(logvals), pos) + jitter, logvals, s=8,
                       color=AGENT_COLORS[agent], alpha=0.55, linewidth=0)
            if len(logvals):
                ax.plot([pos - 0.18, pos + 0.18], [np.median(logvals)] * 2,
                        color="black", lw=1.2)
            positions.append(pos)
            tick_labels.append("C" if agent.startswith("Codex") else "A")
            pos += 1
        pos += 0.35
    ax.axhline(0, color="#222222", lw=0.9, ls="--")
    ax.set_xticks(positions, tick_labels)
    ax.set_ylabel(r"$\log_{10}$(relative L2 / threshold)")
    ax.set_title("c  Accuracy margin (C, Codex; A, Claude)", loc="left", fontweight="bold")
    ax.text(0.99, 0.96, "Pass accuracy gate", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(19)
    for pos, family in enumerate(families):
        gaps = np.asarray([
            100 * float(case["reference"]["convergence_check"]["gap_relative_l2"])
            for case in cases.values()
            if case["case_spec"]["pde"]["operator_family"] == family
        ])
        jitter = rng.uniform(-0.12, 0.12, size=len(gaps))
        ax.scatter(np.full(len(gaps), pos) + jitter, gaps, s=13,
                   color="#377f6a", alpha=0.72, linewidth=0)
        ax.plot([pos - 0.18, pos + 0.18], [np.median(gaps)] * 2,
                color="black", lw=1.2)
    ax.axhline(5.0, color="#c74b50", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 7.5)
    ax.set_xticks(np.arange(len(families)), ["Het.", "ADR", "L-shape", "Coupled"])
    ax.set_ylabel("61-to-121 reference gap (%)")
    ax.set_title("d  Independent reference convergence", loc="left", fontweight="bold")
    ax.text(0.99, 0.90, "Frozen 5% limit", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="#a33b40")
    ax.spines[["top", "right"]].set_visible(False)

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def _write_tex_macros(path: Path, summary: dict[str, Any]) -> None:
    codex = summary["agents"]["Codex GPT-5.5"]
    claude = summary["agents"]["Claude Opus 4.7"]
    delta = summary["paired_codex_minus_claude"]["case_pass"]
    lines = [
        "% Generated by scripts/analyze_external_naturalistic_validation.py",
        rf"\newcommand{{\ExternalCodexPassCount}}{{{codex['case_pass']}}}",
        rf"\newcommand{{\ExternalClaudePassCount}}{{{claude['case_pass']}}}",
        rf"\newcommand{{\ExternalRecordsPerAgent}}{{{codex['records']}}}",
        rf"\newcommand{{\ExternalCodexPassRate}}{{{100*codex['case_pass_rate']:.1f}\%}}",
        rf"\newcommand{{\ExternalClaudePassRate}}{{{100*claude['case_pass_rate']:.1f}\%}}",
        rf"\newcommand{{\ExternalCodexRepeatAgreement}}{{{codex['repeat_agreement_cases']}/32}}",
        rf"\newcommand{{\ExternalClaudeRepeatAgreement}}{{{claude['repeat_agreement_cases']}/32}}",
        rf"\newcommand{{\ExternalEndpointReclassifications}}{{{codex['endpoint_pass_reclassified_by_method'] + claude['endpoint_pass_reclassified_by_method']}}}",
        rf"\newcommand{{\ExternalPassDelta}}{{{100*delta['difference']:+.1f}}}",
        rf"\newcommand{{\ExternalPassDeltaLow}}{{{100*delta['case_bootstrap_95ci'][0]:+.1f}}}",
        rf"\newcommand{{\ExternalPassDeltaHigh}}{{{100*delta['case_bootstrap_95ci'][1]:+.1f}}}",
    ]
    for agent, prefix in (("Codex GPT-5.5", "Codex"), ("Claude Opus 4.7", "Claude")):
        for family, macro in (
            ("heterogeneous_diffusion", "Heterogeneous"),
            ("advection_diffusion_reaction", "Advection"),
            ("lshape_poisson", "LShape"),
            ("coupled_elliptic", "Coupled"),
        ):
            item = summary["agents"][agent]["families"][family]
            lines.append(
                rf"\newcommand{{\External{prefix}{macro}PassCount}}{{{item['case_pass']}}}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, default=ROOT / "benchmark/cases/external_naturalistic_32.jsonl")
    p.add_argument("--results-root", type=Path, default=Path("raw_runs/results/batch"))
    p.add_argument("--codex-experiment", default="external_naturalistic32_v2_codex_gpt55_scipy_n2_20260811T000000Z")
    p.add_argument("--claude-experiment", default="external_naturalistic32_v2_claude_opus47_scipy_n2_20260811T000000Z")
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis/external_validation_20260811/results")
    p.add_argument("--figure-stem", type=Path, default=ROOT / "Nature/figs/fig_external_validation")
    p.add_argument("--tex-macros", type=Path, default=ROOT / "Nature/generated/external_validation_macros.tex")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv)
    cases = _load_cases(args.cases)
    rows = _load_agent(
        experiment_root=args.results_root / args.codex_experiment,
        agent_name="codex_cli_batch", provider_model="openai-gpt-5.5",
        agent_label="Codex GPT-5.5", cases=cases,
    )
    rows += _load_agent(
        experiment_root=args.results_root / args.claude_experiment,
        agent_name="claude_code_cli_batch", provider_model="anthropic-claude-opus-4-7",
        agent_label="Claude Opus 4.7", cases=cases,
    )
    if len(rows) != 128 and not args.allow_partial:
        raise RuntimeError(f"external panel incomplete: {len(rows)}/128 verdicts")
    summary = _summary(rows, cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.out_dir / "run_level_source_data.csv", rows)
    md = [
        "# Frozen naturalistic external validation",
        "",
        f"Complete: **{summary['complete']}** ({summary['observed_records']}/128 records).",
        "",
    ]
    for agent, item in summary["agents"].items():
        ci = item["case_pass_case_bootstrap_95ci"]
        md.append(
            f"- {agent}: {item['case_pass']}/{item['records']} composite passes "
            f"({100*item['case_pass_rate']:.1f}%, case-bootstrap 95% CI "
            f"{100*ci[0]:.1f}–{100*ci[1]:.1f}%); endpoint-pass method "
            f"reclassifications={item['endpoint_pass_reclassified_by_method']}; "
            f"repeat agreement={item['repeat_agreement_cases']}/32 cases."
        )
    delta = summary["paired_codex_minus_claude"]["case_pass"]
    md += [
        "",
        f"Paired Codex-minus-Claude composite-pass difference: {100*delta['difference']:+.1f} pp "
        f"(case-bootstrap 95% CI {100*delta['case_bootstrap_95ci'][0]:+.1f} to "
        f"{100*delta['case_bootstrap_95ci'][1]:+.1f} pp).",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if summary["complete"]:
        _plot(rows, summary, cases, args.figure_stem)
        _write_tex_macros(args.tex_macros, summary)
    print(json.dumps({
        "observed_records": len(rows),
        "complete": summary["complete"],
        "out_dir": str(args.out_dir),
        "figure_stem": str(args.figure_stem) if summary["complete"] else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
