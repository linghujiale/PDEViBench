# Frozen-program scientific replay task

Write exactly one Python file, `solver.py`, that exposes this public API:

```python
def solve(problem_spec: dict) -> numpy.ndarray:
    ...
```

The returned array must satisfy the dtype and shape declared inside the
`problem_spec` argument. The evaluator will import the same submitted file and
call `solve` repeatedly. It will first use the base specification below and then
use three hidden, within-family specifications. Your program will not be
regenerated or edited between calls.

## Scientific requirements

- Numerically discretize and solve the PDE described by the argument received
  on each call.
- Read coefficients, forcing scale, boundary data, domain parameters and
  evaluation-grid resolution from `problem_spec`; do not substitute frozen
  base-case constants for argument values.
- It is acceptable to define reusable helper functions in `solver.py`.
- Use NumPy and SciPy only.
- Do not read files, environment variables, command-line arguments or network
  resources. The only scientific input is the `problem_spec` argument.
- Do not embed or reconstruct a reference array.
- Do not write output files. Return one `numpy.ndarray` directly.
- Seed any randomness with `{seed}`, although deterministic solvers are
  preferred.

## Public schema

The argument has the following top-level fields:

- `schema_version`, `case_id`, `spec_id`, `family`;
- `parameters`, including the family coefficients, `forcing_rep` and
  `forcing_scale`;
- `domain`;
- `boundary`, including a scalar Dirichlet value shared by the active boundary
  (and by both components in the coupled family);
- `eval_grid`, including `nx`, `ny`, indexing and mask information;
- `output`, including `dtype` and `shape`.

Hidden replays remain in the same PDE family and change one preregistered
scientific factor at a time. No hidden values are supplied here.

## Base specification

```json
{case_spec_json}
```

## Deliverable

Write `solver.py` in the current working directory and stop. You may test
`solve` on the base specification, but the final file must not contain generated
outputs or any hidden-specification guess.
