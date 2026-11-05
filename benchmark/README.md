# Benchmark tasks and references

A task is specified by the levels of six design factors: operator (O), geometry (G), boundary condition (B), discretisation specificity (D), structure hint (S) and coupling (C).

| Path | Contents |
|---|---|
| `generators/` | One module per factor and `compose.py`, which assembles a task from a factor configuration |
| `reference/` | Manufactured-solution references (`manufactured.py`), numerical references (`high_fidelity.py`) and the stored reference fields of the numerical-reference and naturalistic panels |
| `cases/` | Task files of every panel (`*.jsonl`) and case lists of the guide experiments |
| `baselines.json` | Finite-difference baseline errors used to normalise accuracy across design cells |

Every generator takes a seed and is deterministic given the seed and the pinned dependencies.
