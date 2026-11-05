#!/usr/bin/env python3
"""Render Fig. 5: unchanged-program replay with original and structural variants."""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from _nmi_editorial_style import PALETTE, apply_editorial_style, figure_size, save_figure

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "analysis/round1_review_20261029/structural_replay/structural_replay_program_results.csv"
OUT_SD = ROOT / "paper/source_data/fig5_structural_replay"
COLS = [("base", "Base"), ("source_scale", "Source\namplitude"), ("boundary_data", "Dirichlet\nshift"),
        ("resolution", "Grid\nrefinement"), ("structural", "Structural\nvariants")]
FAMS = [("heterogeneous_diffusion", "Heterogeneous diffusion"), ("advection_diffusion_reaction", "Advection-diffusion-reaction"),
        ("lshape_poisson", "L-shaped Poisson"), ("coupled_elliptic", "Coupled elliptic*")]
CTLS = [("positive_parameterized", "Parameterised solver"), ("negative_affine_reconstruction", "Affine reconstruction"),
        ("negative_mixed", "Mixed solver/target"), ("negative_direct_target", "Direct target"),
        ("negative_hardcoded_instance", "One-instance hard-coded")]

def t(v): return v == "True"

def tables():
    rows = list(csv.DictReader(SRC.open(newline="", encoding="utf-8")))
    agent, ctl = [], []
    for key, label in FAMS:
        A = [r for r in rows if r["kind"] == "agent" and r["family"] == key]
        assert len(A) == 16
        rec = {"row": label, "n": 16}
        for c, _ in COLS[:-1]:
            rec[c] = sum(t(r["pass__" + c]) for r in A)
        rec["structural"] = sum(t(r["structural_primary_all"]) for r in A)
        rec["structural_n"] = 16
        agent.append(rec)
    for key, label in CTLS:
        C = [r for r in rows if r["kind"] != "agent" and r["control_type"] == key]
        assert len(C) == 32
        rec = {"row": label, "n": 32}
        for c, _ in COLS[:-1]:
            rec[c] = sum(t(r["pass__" + c]) for r in C)
        sp = sum(int(r["structural_primary_n_pass"] or 0) for r in C)
        sn = sum(int(r["structural_primary_n"] or 0) for r in C)
        if key == "positive_parameterized":
            cut = [r for r in rows if r["control_type"] == "positive_parameterized_reads_cut"]
            assert len(cut) == 8
            lsh = [r for r in C if r["family"] == "lshape_poisson"]
            sp += sum(int(r["structural_primary_n_pass"] or 0) for r in cut) - sum(int(r["structural_primary_n_pass"] or 0) for r in lsh)
        rec["structural"], rec["structural_n"] = sp, sn
        ctl.append(rec)
    return agent, ctl

def tile(ax, recs, title):
    cmap = LinearSegmentedColormap.from_list("pass", ["#F2F3F5", "#AFC2D5"])
    for i, rec in enumerate(recs):
        for j, (c, _) in enumerate(COLS):
            n = rec["structural_n"] if c == "structural" else rec["n"]
            k = rec[c]; f = k / n
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(f), edgecolor="white", lw=1.2))
            ax.text(j + 0.5, i + 0.5, f"{k}/{n}", ha="center", va="center", fontsize=7.0,
                    color=PALETTE["text"])
    ax.set_xlim(0, len(COLS)); ax.set_ylim(len(recs), 0)
    ax.set_xticks(np.arange(len(COLS)) + 0.5, [c[1] for c in COLS], fontsize=6.8)
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=2)
    ax.set_yticks(np.arange(len(recs)) + 0.5, [r["row"] for r in recs], fontsize=7.0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc="left", pad=24)

def main():
    agent, ctl = tables()
    OUT_SD.mkdir(parents=True, exist_ok=True)
    with (OUT_SD / "replay_tiles.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["panel", "row", "n", *[c for c, _ in COLS], "structural_n"])
        w.writeheader()
        for p, recs in (("a", agent), ("b", ctl)):
            for r in recs: w.writerow({"panel": p, **r})
    apply_editorial_style()
    def build():
        fig = plt.figure(figsize=figure_size(62))
        ax_a = fig.add_axes([0.19, 0.08, 0.315, 0.66])
        ax_b = fig.add_axes([0.675, 0.08, 0.315, 0.66])
        tile(ax_a, agent, "Agent programs")
        tile(ax_b, ctl, "Constructed reference programs")
        fig.text(0.01, 0.97, "a", fontsize=8, fontweight="bold", va="top", color=PALETTE["text"])
        fig.text(0.535, 0.97, "b", fontsize=8, fontweight="bold", va="top", color=PALETTE["text"])
        return fig
    for stem in (ROOT / "paper/figures/fig5_structural_replay", ROOT / "Nature/figs/fig4_unchanged_program_replay"):
        save_figure(build(), stem, tiff=True)
    print(agent); print(ctl)

if __name__ == "__main__":
    main()
