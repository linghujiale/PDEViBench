"""Regenerate every figure and generated LaTeX table of the PDEViBench manuscript."""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    'scripts/build_faithful_rates.py',
    'scripts/build_masked_sensitivity.py',
    'scripts/build_matched_visibility_source_data.py',
    'scripts/build_prompt_comparison_source.py',
    'scripts/build_prompt_transitions.py',
    'scripts/build_reaudit_si_table.py',
    'scripts/build_round1_si_tables.py',
    'scripts/build_structural_replay_table.py',
    'scripts/generate_nmi_extended_data_tables.py',
    'scripts/generate_nmi_main_tables.py',
    'scripts/generate_nmi_next_round_macros.py',
    'scripts/generate_nmi_structural_sources.py',
    'scripts/reconcile_nmi_phase5_replay.py',
    'scripts/regenerate_nmi_phase5_figure2.py',
    'paper/figures/regen/render_nmi_extended_data_editorial.py --figures 1 2',
    'paper/figures/regen/fig_endpoint_nonidentifiability.py',
    'paper/figures/regen/fig2_route_mixture_editorial.py',
    'paper/figures/regen/fig3_matched_visibility.py',
    'paper/figures/regen/fig5_structural_replay.py',
    'paper/figures/regen/si_representative_tasks.py',
]

def main():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "paper" / "figures" / "regen")]), MPLBACKEND="Agg")
    for step in STEPS:
        print(f"[reproduce] {step}", flush=True)
        subprocess.run([sys.executable, *step.split()], cwd=ROOT, env=env, check=True)
    print("[reproduce] done")

if __name__ == "__main__":
    main()
