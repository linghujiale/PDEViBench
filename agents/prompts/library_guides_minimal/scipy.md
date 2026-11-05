## scipy / NumPy host-mode environment

The execution environment provides `numpy`, `scipy`, and `sympy`. It does not
provide DOLFINx, Firedrake, or another finite-element package, and network
access is disabled.

Implement a numerical discretization of the stated PDE with NumPy/SciPy,
apply the supplied boundary and initial conditions, solve the resulting
algebraic or time-stepping problem, and evaluate the computed solution on the
specified grid. The task-level requirements above govern the output file,
shape, dtype, runtime, and prohibition on direct analytic/manufactured-formula
evaluation.

No discretization stencil, mesh-refinement rule, coordinate-mapping recipe,
time integrator, or boundary-condition implementation is prescribed here.
