{# Agent-facing prompt template used by every runner.                           #}
{# Placeholders are filled with str.format_map; these comment lines are         #}
{# removed before formatting.                                                   #}
{#                                                                              #}
{# Tokens (curly-brace, str.format-style):                                       #}
{#   {case_id}                — opaque identifier                                #}
{#   {library}                — library named in the task                       #}
{#   {pde_statement}          — natural-language statement                       #}
{#   {operator_family}        — canonical family label                           #}
{#   {parameters_block}       — Markdown bullet list of numerical parameters     #}
{#   {domain_block}           — Markdown block describing geometry               #}
{#   {bc_block}               — Markdown block describing boundary conditions    #}
{#   {ic_block}               — Markdown block describing initial conditions    #}
{#                              (empty if steady)                                #}
{#   {eval_grid_block}        — Markdown block describing the evaluation grid    #}
{#   {output_spec}            — dtype + shape requirement                        #}
{#   {library_guide}          — content of agents/prompts/library_guides/<lib>.md#}
{#   {seed}                   — integer seed for reproducibility                 #}

# PDE Solver Task

You are an expert numerical-analyst engineer working inside an isolated
container. Your sole task is to write a single Python file `solver.py` that
solves the PDE described below, using the **{library}** library, and writes
the solution to `solution.npz` (in your current working directory).

## Hard requirements

* Save the solution as a NumPy array named `solution` with **dtype = `{output_spec.dtype}`** and **shape = `{output_spec.shape}`**, written with `numpy.savez` to `solution.npz` (in your current working directory).
* Use the **{library}** library; do not import any solver from another PDE library.
* Solve the PDE numerically by discretizing the differential operator, applying
  the boundary/initial conditions, and solving the resulting algebraic or time
  stepping problem. Do **not** directly construct the manufactured, analytic, or
  exact solution on the evaluation grid, even if you can infer such a formula
  from the forcing or initial-condition text.
* Do not name or encode variables such as `u_exact`, `exact_solution`, or
  `manufactured_solution` as the submitted solution. Near-zero outputs produced
  by direct formula evaluation fail the method check.
* Do not access the network.
* You may write intermediate files only in your current working directory.
* Wall-clock budget: 300 seconds inside the container. Exceeding it counts as a failure.
* Use the integer seed `{seed}` to seed any random-number generator you invoke.
* Do not print large arrays to stdout; keep stdout under 64 KiB.

## Case

Identifier: `{case_id}`

### PDE

Operator family: `{operator_family}`

{pde_statement}

### Parameters

{parameters_block}

### Domain

{domain_block}

### Boundary conditions

{bc_block}

### Initial conditions

{ic_block}

### Evaluation grid

The submitted solution will be compared against a reference solution sampled
on the following grid:

{eval_grid_block}

You **must** evaluate your computed solution on **exactly** this grid before
writing it to `solution.npz` (in your current working directory).

## Output format

* File: `solution.npz` (in your current working directory)
* Key: `solution`
* dtype: `{output_spec.dtype}`
* shape: `{output_spec.shape}`
* No additional keys; no pickled objects.

## Library guide

{library_guide}

## What to deliver

When you are confident your solver is correct, write the final file to
`solver.py` (in your current working directory) and exit. The grading harness will execute it and
evaluate the output.

Do not include explanations in `solver.py` — just runnable code. Keep
imports at the top.
