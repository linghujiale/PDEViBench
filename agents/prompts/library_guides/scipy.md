## scipy / NumPy host-mode quick reference

You are running on a **host fallback** environment with `numpy`, `scipy`,
and `sympy` available — **no DOLFINx, no Firedrake, no FEM library**.
Network access is disabled.

You must implement the solver as a hand-rolled finite-difference (FD)
discretization on the case's evaluation grid, using `scipy.sparse` for the
linear algebra. The reference solutions for this benchmark are themselves
FD solutions on a refined mesh, then subsampled to the case's eval grid.

Do not bypass the discretization by evaluating an analytic/manufactured
solution formula directly. Even if the forcing or initial-condition text lets
you infer a closed form, the expected submission is a numerical FD solve
(`scipy.sparse` assembly, linear/nonlinear solve, or time stepping) followed by
sampling on the requested grid.

### Building the linear system

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Example: 5-point FD Laplacian on a unit-square Dirichlet-0 problem.
# eval grid: (nx, ny) per case.eval_grid; here nx=ny=16 → 14×14 interior.
nx, ny = 16, 16
h = 1.0 / (nx - 1)
N = (nx - 2) * (ny - 2)

# Assemble -Δ_h on interior unknowns.
rows, cols, vals = [], [], []
for j in range(ny - 2):
    for i in range(nx - 2):
        k = j * (nx - 2) + i
        rows.append(k); cols.append(k); vals.append(4.0 / h**2)
        if i > 0:
            rows.append(k); cols.append(k - 1); vals.append(-1.0 / h**2)
        if i < nx - 3:
            rows.append(k); cols.append(k + 1); vals.append(-1.0 / h**2)
        if j > 0:
            rows.append(k); cols.append(k - (nx - 2)); vals.append(-1.0 / h**2)
        if j < ny - 3:
            rows.append(k); cols.append(k + (nx - 2)); vals.append(-1.0 / h**2)

A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
b = np.zeros(N)  # fill with f(x_i, y_j) at interior nodes
u_int = spla.spsolve(A.tocsc(), b)
```

### Eval-grid mapping

The case specifies the eval grid (`case_spec.eval_grid.spec` — e.g.
`nx=16, ny=16` for unit_square). **Solve and report on EXACTLY this grid.**

For unit_square problems the standard ordering is row-major over (j, i)
with `x = i * h, y = j * h`. Reshape to a flat length-256 vector with
`u.reshape(-1)`.

For non-trivial geometries (annulus, sector, L-shape, square-with-hole)
the eval grid lives in polar or masked-Cartesian coordinates; consult the
case's `domain.spec` block for the exact mapping. Polar-coordinate cases
use a regular (r, θ) mesh; flatten in (i_radial × n_angular + j_angular)
row-major order.

### Time-dependent problems

* Heat: Backward-Euler or Crank-Nicolson + a 2D FD Laplacian. CN is
  recommended for accuracy at `Δt = T_final / 200`.
* Wave: Explicit leapfrog with `Δt = T_final / 2000` or smaller (CFL).
* Burgers: IMEX-CN (implicit diffusion, explicit convection); centered
  FD first-derivatives for the convection term.

### Refined-mesh strategy

If the eval grid is coarse (16×16 on unit_square gives h=1/15), solve on
a refined internal mesh — e.g. 4× per direction, so (16-1)*4+1 = 61 nodes
per axis — and subsample with `U_refined[::4, ::4]`. This brings the
discretization error well under `τ_acc = 1e-2` even for stiff problems.

### Mixed Dirichlet/Neumann (B1)

For homogeneous Neumann on one boundary, use the ghost-point method:
keep the boundary node as an unknown, and on its row of the FD stencil
substitute `u_{N+1} = u_{N-1}` (so the second-derivative stencil becomes
`(2 u_{N-1} - 2 u_N) / h²`).

### Output

```python
import numpy as np
# Write solution.npz to the current working directory — the grading harness
# will pick it up from there.
np.savez("solution.npz", solution=u.astype("float64").reshape(-1))
```
