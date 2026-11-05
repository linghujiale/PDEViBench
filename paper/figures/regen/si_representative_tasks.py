"""Supplementary Fig. 2 and Supplementary Table 14: reference fields and selected programs of twelve representative tasks.
Reuses the data loaders of si_representative_case_atlas.py; prints the LaTeX table rows.
"""
import csv, json, sys, importlib.util
from collections import defaultdict
from pathlib import Path
import numpy as np, matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("atlasgen", HERE / "si_representative_case_atlas.py")
A = importlib.util.module_from_spec(spec); sys.modules["atlasgen"] = A; spec.loader.exec_module(A)
matplotlib.rcParams.update({'axes.labelsize': 8.0, 'axes.linewidth': 0.7, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.titlelocation': 'left', 'axes.titlesize': 8.0, 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'], 'font.size': 7.6, 'legend.fontsize': 7.0, 'legend.frameon': False, 'lines.linewidth': 1.2, 'patch.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.dpi': 300.0, 'savefig.facecolor': 'white', 'svg.fonttype': 'none', 'svg.hashsalt': 'pdevibench-supp-atlas', 'xtick.labelsize': 6.0, 'xtick.major.size': 3.0, 'xtick.major.width': 0.6, 'ytick.labelsize': 6.0, 'ytick.major.size': 3.0, 'ytick.major.width': 0.6})
PDE = {"Low-wavenumber Helmholtz": "Helmholtz"}
GEO = {"unit square": "square"}
BC = {"homogeneous Dirichlet": "Dirichlet"}
SOLVER = {"sparse_fd_linear_solve": "sparse FD solve", "coordinate_transform_or_polar_solver": "polar-grid solve", "time_stepping_solver": "time stepping"}

def load():
    cases = A.load_cases(); summary = list(csv.DictReader(A.SUMMARY_CSV.open(encoding="utf-8")))
    route = {}
    with A.ROUTE_CENSUS.open(newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h): route[(r["experiment_id"], r["case_id"], r["run_idx"])] = r
    runs, reps = {}, {}
    for agent, d in A.AGENTS.items():
        idx = defaultdict(dict)
        for rec in A.read_jsonl(d / "runs.jsonl"): idx[rec["case_id"]][int(rec["run_idx"])] = rec
        runs[agent] = {c: list(v.values()) for c, v in idx.items()}
        reps[agent] = {e["case_id"]: e for e in json.loads((d / "wave1_report.json").read_text())["per_case"]}
    items = []
    for row in summary:
        cid = row["case_id"]; case = cases[cid]; sel = {}
        for agent in A.AGENTS:
            recs = sorted(runs[agent][cid], key=lambda r: int(r["run_idx"]))
            s = A.select_record(recs, reps[agent][cid])
            if s["selected_run_idx"] != "":
                s["static_route"] = route[(A.AGENT_EXPERIMENT[agent], cid, str(s["selected_run_idx"]))]["static_route"]
            sel[agent] = s
        items.append((int(row["index"]), case, A.load_reference(case), sel))
    return items

def label(case):
    p = A.PDE_LABEL[case["case_spec"]["pde"]["operator_family"]]; g = A.GEOMETRY_LABEL[A.shape_of(case)]; b = A.BC_LABEL[A.bc_of(case)]
    return PDE.get(p, p), GEO.get(g, g), BC.get(b, b)

def sci(x):
    x = float(x); e = int(np.floor(np.log10(x))); return f"${x/10**e:.1f}\\times10^{{{e}}}$"

def entry(s):
    v = s["selected_verdict"]
    if v == "pass": return f"{sci(s['selected_rel_l2'])}, {SOLVER[s['static_route']]}"
    if v.startswith("HOST_EVAL_DOMAIN_MASK"): return "mask check"
    if v == "F-Acc": return f"accuracy failure, {float(s['selected_rel_l2']):.2f}"
    return v

def main():
    items = load()
    fig, axes = plt.subplots(3, 4, figsize=(7.09, 5.6))
    for ax, (i, case, field, sel) in zip(axes.ravel(), items):
        im = A.plot_reference(ax, field, case); p, g, b = label(case)
        ax.set_title(f"{p}, {g}\n{b}", fontsize=7, pad=3, loc="left")
        ax.annotate("abcdefghijkl"[i - 1], xy=(0, 1), xycoords="axes fraction", xytext=(-9, 21), textcoords="offset points", fontsize=8, fontweight="bold", ha="left", va="top")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6, length=2, width=0.5); cb.outline.set_linewidth(0.5)
        cb.locator = matplotlib.ticker.MaxNLocator(3); cb.update_ticks()
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03, wspace=0.35, hspace=0.5)
    fig.savefig(A.ROOT / "Nature" / "figs" / "supp_representative_tasks.pdf", bbox_inches="tight")
    for i, case, field, sel in items:
        p, g, b = label(case); cell = "".join(f"{a}{case['axes'][a]}" for a in "OGBDSC")
        print(f"{'abcdefghijkl'[i-1]} & {p}, {g}, {b} & {cell} & {entry(sel['Codex'])} & {entry(sel['Claude'])} \\\\")

if __name__ == "__main__":
    main()
