## scipy / NumPy host-mode numerical prescriptions

The execution environment provides `numpy`, `scipy`, and `sympy`. It does not
provide DOLFINx, Firedrake, or another finite-element package, and network
access is disabled.

Implement a hand-rolled finite-difference numerical solve of the stated PDE
with NumPy/SciPy. Do not bypass the discretization by evaluating an analytic or
manufactured-solution formula directly. Use SciPy sparse matrices and sparse
linear algebra for linear systems.

For a unit-square Dirichlet problem, use the standard five-point
finite-difference Laplacian on interior unknowns. Number unknowns in row-major
`(j, i)` order with `x` varying fastest. Assemble the diagonal coefficient and
the four nearest-neighbour coefficients into a sparse matrix, solve the
interior system, and restore boundary values before flattening.

For time-dependent problems, use the following numerical prescriptions:

- heat: backward Euler or Crank--Nicolson with 200 uniform time steps;
- wave: explicit leapfrog with at least 2,000 uniform time steps, subject to
  the CFL condition;
- Burgers: centred finite differences for convection and an IMEX
  Crank--Nicolson treatment with implicit diffusion and explicit convection.

When the evaluation grid is coarse, solve on an internal mesh refined fourfold
in each coordinate direction and subsample back to the evaluation grid. For a
16 by 16 unit-square evaluation grid, this gives 61 nodes per coordinate.

For a homogeneous Neumann boundary, retain the boundary node as an unknown and
use the ghost-point relation in which the exterior value equals the reflected
interior value. This changes the boundary second-derivative row to twice the
interior-neighbour contribution minus twice the boundary-node contribution,
divided by the squared grid spacing.

The case specification defines the evaluation grid and domain. For a unit
square, report values in row-major `(j, i)` order with `x` varying fastest. For
polar or masked non-trivial domains, follow the coordinate order and mask in
the case's `domain.spec` and `eval_grid.spec` and flatten in the specified
row-major order.

Write `solution.npz` in the current working directory with a single finite
`float64` array named `solution`, flattened to the exact required length.

This guide intentionally contains no executable code block or copyable program
fragment. It fixes numerical choices but leaves their implementation to you.
