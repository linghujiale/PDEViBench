## scipy / NumPy host-mode benchmark interface

The execution environment provides `numpy`, `scipy`, and `sympy`. It does not
provide DOLFINx, Firedrake, or another finite-element package, and network
access is disabled.

Implement a hand-rolled finite-difference numerical solve of the stated PDE
with NumPy/SciPy. Apply the supplied boundary and initial conditions and
evaluate the computed solution on the case's specified grid. Do not bypass the
discretization by evaluating an analytic or manufactured-solution formula
directly.

The case specification defines the evaluation grid and domain. For a unit
square, report values in row-major `(j, i)` order with `x` varying fastest. For
polar or masked non-trivial domains, follow the coordinate order and mask in
the case's `domain.spec` and `eval_grid.spec`; flatten the resulting grid in
the specified row-major order. These are benchmark-interface semantics, not a
solver prescription.

Choose the stencil, internal resolution, linear or nonlinear solver, time
integrator, time-step size, and boundary discretization yourself. No code
example, mesh-refinement factor, time-step count, or boundary formula is
prescribed by this guide.

Write `solution.npz` in the current working directory with a single
`solution` array. The array must be finite `float64`, flattened to the exact
length and ordering required by the case.
