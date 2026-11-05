# PDEViBench — program source distribution

PDEViBench benchmarks coding agents on partial differential equations and checks
the computation behind each returned field. This folder contains the program
sources and the inputs needed for the benchmark. Manuscript material and existing
experimental results are excluded.

## Contents

| Path | Purpose |
|---|---|
| `benchmark/generators/` | PDE task generation |
| `benchmark/reference/` | Reference construction code and stored reference arrays |
| `benchmark/cases/` | Task inputs, case lists and experiment manifests |
| `evaluator/` | Execution, accuracy, runtime and method checks |
| `agents/` | Agent runners, prompt templates and library guides |
| `schemas/` | Task, trace and result JSON schemas |
| `scripts/` | Experiment runners, statistical analysis and table-generation code |
| `analysis/axis_models.py` | Statistical modelling code |
| `analysis/.../*.jsonl` | Two retained experiment inputs: schedule and replay case bank |
| `analysis/.../frozen_programs/` | Five original failed-generation placeholder programs |
| `paper/figures/regen/` | Plotting source code only |
| `docker/` | Evaluator / DOLFINx images and a smoke-test program |
| `reproduce.py` | Original paper figure/table reproduction driver; requires external result data |

The five files in `frozen_programs/` raise generation-failure exceptions; they
are preserved source artifacts, not successful numerical solvers. The original
freeze manifest lists 64 programs, but this supplied source directory contains
only these five Python files. No missing programs have been reconstructed.

## Setup and benchmark input checks

The original project requires Python >=3.11,<3.13. From this folder:

```bash
python -m pip install -e .
python -m benchmark.validate --cases benchmark/cases/pilot_phase1_8.jsonl --check-references
python scripts/run_phase2_wave1.py --help
```

For agent SDK dependencies, install `python -m pip install -e ".[agents]"`.
Statistical analysis extras are available with `python -m pip install -e ".[analysis]"`.
Agent execution also needs the appropriate CLI/backend setup and credentials in
the environment. Some experiment drivers use Linux facilities (e.g. `fcntl`),
so use Linux/WSL or the documented containers for those drivers.
See `docker/README.md` for image build commands.

`scripts/run_case_batch.py` has a legacy default pointing to the absent
`pilot_phase1_6.jsonl`; explicitly pass `--cases benchmark/cases/pilot_phase1_8.jsonl`.

## Analysis and plotting inputs

All original Python sources are retained unchanged, including the plotting code
under `paper/figures/regen/`. The name `paper` here preserves paths used by the
programs; it contains no manuscript text or existing figures.

Historical run records, paper source-data tables, and generated LaTeX macros are
excluded. Consequently, `python reproduce.py` and historical-result analyses
cannot reproduce the paper from this source distribution alone. They require the
corresponding original data in its original paths. Certain historical experiment
drivers also reference schedules/manifests already absent in the supplied source
(e.g. `generation_schedule.jsonl` and `execution_freeze_manifest.json` for frozen
program generation). This folder is not a complete paper-reproduction archive.

## License

The original project's MIT license and author metadata are retained; see `LICENSE`.
