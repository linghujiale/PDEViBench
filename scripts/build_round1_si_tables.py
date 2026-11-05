#!/usr/bin/env python3
"""SI tables: census of every panel in the route ledger and label sensitivity."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "analysis/nmi_phase5_20260813/final_postreview_route_ledger.csv"
KEY = ROOT / "analysis/nmi_route_mixture_20260811/blinded_review_extension/unblinding_key/blinding_key.jsonl"
LABELS = ROOT / "analysis/reviewer_labels_expanded_218.csv"
REAUDIT = ROOT / "analysis/reaudit_20261030/reaudit_relabels_20261030.csv"
OUT = ROOT / "Nature/generated/nmi_round1_si_tables.tex"

PANELS = [
    ("phase2_wave1_codex", "Primary, 290 cases", "Codex (GPT-5.2)", "Original"),
    ("phase2_wave1b_codex", "Primary, 290 cases", "Codex (GPT-5.2)", "Prohibition"),
    ("phase2_wave1b_claude", "Primary, 290 cases", "Claude Code (Claude Opus 4.7)", "Prohibition"),
    ("phase2_wave1b_openrouter_deepseek", "Primary, 290 cases", "DeepSeek V3.2", "Prohibition"),
    ("axis1024_codex", "Breadth, 1,024 cases", "Codex (GPT-5.5)", "Prohibition"),
    ("axis1024_claude", "Breadth, 1,024 cases", "Claude Code (Claude Opus 4.7)", "Prohibition"),
    ("scaffold64_fullguide", "Guide ablation, 64 cases, full guide", "Codex (GPT-5.5)", "Prohibition"),
    ("scaffold64_minimalguide", "Guide ablation, 64 cases, minimal guide", "Codex (GPT-5.5)", "Prohibition"),
    ("pcsep128_fullguide", "Three-arm, 128 cases, full", "Codex (GPT-5.5)", "Prohibition"),
    ("pcsep128_prescriptionguide", "Three-arm, 128 cases, prescription-only", "Codex (GPT-5.5)", "Prohibition"),
    ("pcsep128_nocodeguide", "Three-arm, 128 cases, choice-free", "Codex (GPT-5.5)", "Prohibition"),
    ("phase2_confirm72_codex_dolfinx", "Backend, 72 cases, DOLFINx", "Codex (GPT-5.2)", "Prohibition"),
    ("phase2_confirm72_gemini", "Provider, 72 cases", "Gemini 2.5 Pro", "Prohibition"),
    ("external_naturalistic32_v2_codex", "Naturalistic, 32 cases", "Codex (GPT-5.5)", "Evaluator-held"),
    ("external_naturalistic32_v2_claude", "Naturalistic, 32 cases", "Claude Code (Claude Opus 4.7)", "Evaluator-held"),
]

rows = list(csv.DictReader(LEDGER.open(newline="", encoding="utf-8")))
by_exp = defaultdict(list)
for r in rows:
    by_exp[r["experiment_id"]].append(r)
exp_of = {}
for prefix, *_ in PANELS:
    hits = [e for e in by_exp if e.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    exp_of[prefix] = hits[0]
assert set(exp_of.values()) == set(by_exp), set(by_exp) - set(exp_of.values())

def fmt(n): return f"{n:,}"
census, tot = [], Counter()
for prefix, panel, agent, prompt in PANELS:
    rs = by_exp[exp_of[prefix]]
    c = Counter(r["route_status"] for r in rs)
    cases = len({r["case_id"] for r in rs}); calls = len(rs)
    ep = sum(r["endpoint_pass"] == "True" for r in rs)
    assert ep == c["route_valid"] + c["direct_recovery"] + c["mixed_route"]
    tot.update(calls=calls, ep=ep, rv=c["route_valid"], d=c["direct_recovery"], m=c["mixed_route"])
    census.append(f"{panel} & {agent} & {prompt} & {fmt(cases)}$\\times${calls // cases} & {fmt(calls)} & {fmt(ep)} & {fmt(c['route_valid'])} & {c['direct_recovery']} & {c['mixed_route']} \\\\")
census.append("\\midrule")
census.append(f"\\textbf{{All panels}} & & & & \\textbf{{{fmt(tot['calls'])}}} & \\textbf{{{fmt(tot['ep'])}}} & \\textbf{{{fmt(tot['rv'])}}} & \\textbf{{{tot['d']}}} & \\textbf{{{tot['m']}}} \\\\")

key = {k["blind_id"]: k for k in map(json.loads, KEY.open(encoding="utf-8"))}
ledger_by_key = {(r["experiment_id"], r["case_id"], int(r["run_idx"])): r for r in rows}

contested_keys = set()
for r in csv.DictReader(LABELS.open(newline="", encoding="utf-8")):
    if r["binary_agreement"] == "False":
        assert r["adjudicator_label"] == "NUMERICAL_SOLVER" and r["reviewer_A_binary"] == "ROUTE_INVALID"
        k = key[r["blind_id"]]
        contested_keys.add((k["experiment_id"], k["case_id"], int(k["run_idx"])))
assert len(contested_keys) == 47

reaudit_keys = {
    (r["experiment_id"], r["case_id"], int(r["run_idx"]))
    for r in csv.DictReader(REAUDIT.open(newline="", encoding="utf-8"))
}
assert len(reaudit_keys) == 42 and not (reaudit_keys & contested_keys)
for k in reaudit_keys | contested_keys:
    assert ledger_by_key[k]["endpoint_pass"] == "True" and ledger_by_key[k]["route_status"] == "route_valid", k
reaudit = Counter(k[0] for k in reaudit_keys)
contested = Counter(k[0] for k in contested_keys)


def label_sets(e):
    rs = by_exp[e]; n = len(rs)
    ep = sum(r["endpoint_pass"] == "True" for r in rs)
    rv = sum(r["route_status"] == "route_valid" for r in rs)
    return n, ep, rv, rv - reaudit[e], rv - reaudit[e] - contested[e]


sens = []
for prefix, panel, agent, prompt in PANELS:
    e = exp_of[prefix]
    if not (reaudit[e] or contested[e]):
        continue
    n, ep, cur, b, c = label_sets(e)
    sens.append(f"{panel} & {agent} & {prompt} & {fmt(cur)} ({100*cur/n:.2f}\\%) & {fmt(b)} ({100*b/n:.2f}\\%) & {fmt(c)} ({100*c/n:.2f}\\%) \\\\")

n, ep, cur, b, c = label_sets(exp_of["phase2_wave1_codex"])
assert n == 1450
pp = lambda x: f"{100 * x / n:.2f}"
sens_macros = [
    f"\\newcommand{{\\NMILabelSensitivityReauditCount}}{{{len(reaudit_keys)}}}",
    f"\\newcommand{{\\NMILabelSensitivityContestedCount}}{{{len(contested_keys)}}}",
    f"\\newcommand{{\\NMILabelSensitivityPrimaryGapCurrentPP}}{{{pp(ep - cur)}}}",
    f"\\newcommand{{\\NMILabelSensitivityPrimaryGapConsensusPP}}{{{pp(ep - b)}}}",
    f"\\newcommand{{\\NMILabelSensitivityPrimaryGapReviewerAPP}}{{{pp(ep - c)}}}",
]
arms = {name: label_sets(exp_of[prefix]) for name, prefix in (
    ("full", "pcsep128_fullguide"), ("prescription", "pcsep128_prescriptionguide"), ("choice_free", "pcsep128_nocodeguide"))}
for tag, idx in (("Current", 2), ("Consensus", 3), ("ReviewerA", 4)):
    pc = 100 * (arms["prescription"][idx] - arms["choice_free"][idx]) / 128
    pf = 100 * (arms["prescription"][idx] - arms["full"][idx]) / 128
    sens_macros.append(f"\\newcommand{{\\NMILabelSensitivityPrescChoiceFree{tag}PP}}{{{pc:.2f}}}")
    sens_macros.append(f"\\newcommand{{\\NMILabelSensitivityPrescFull{tag}PP}}{{{pf:.2f}}}")

lines = ["% Generated by scripts/build_round1_si_tables.py; do not edit.",
         "\\newcommand{\\NMIPanelCensusRows}{%", *census, "}",
         "\\newcommand{\\NMILabelSensitivityRows}{%", *sens, "}", *sens_macros,
         f"\\newcommand{{\\NMICensusTotalCalls}}{{{fmt(tot['calls'])}}}",
         f"\\newcommand{{\\NMICensusTotalEndpoint}}{{{fmt(tot['ep'])}}}"]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
