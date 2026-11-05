## DOLFINx 0.9.0 — quick reference for this task

You are running against **DOLFINx 0.9.0** with PETSc 3.21 in MPI mode. The
container has `dolfinx`, `ufl`, `basix`, `mpi4py`, `numpy`, `scipy`, `h5py`,
and `meshio` installed. Network access is disabled.

### Mesh

```python
from mpi4py import MPI
from dolfinx import mesh
m = mesh.create_unit_square(MPI.COMM_WORLD, nx=64, ny=64, cell_type=mesh.CellType.triangle)
```

For non-trivial geometries, use `dolfinx.mesh.create_mesh` with a
`basix.ufl.element`, or import a Gmsh `.msh` file via `dolfinx.io.gmshio`.

### Function spaces

```python
from dolfinx import fem
V = fem.functionspace(m, ("Lagrange", 1))                # P1
V_vec = fem.functionspace(m, ("Lagrange", 2, (m.geometry.dim,)))  # vector P2
W = fem.functionspace(m, ("RT", 1))                       # Raviart–Thomas
```

For Stokes / Darcy / Maxwell, use the appropriate inf-sup-stable pair
(Taylor–Hood = ("Lagrange", 2) for velocity, ("Lagrange", 1) for pressure).

### Variational form

```python
import ufl
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
```

### Boundary conditions

```python
boundary_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
bc = fem.dirichletbc(value, boundary_dofs, V)
```

For Neumann data, use `g * v * ufl.ds(tag)` with a tagged `MeshTags` object.

### Solve

```python
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc],
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
```

For large problems, prefer iterative solvers: `{"ksp_type": "cg", "pc_type": "hypre"}` for SPD systems.

### Evaluate on the case's evaluation grid

The evaluation grid is given as a list of points `pts` (shape `(N, 3)`). Use
`bb_tree` + `compute_collisions_points` to evaluate the discrete solution at
arbitrary points:

```python
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
tree = bb_tree(m, m.topology.dim)
collisions = compute_collisions_points(tree, pts)
colliding = compute_colliding_cells(m, collisions, pts)

# DOLFINx 0.9 returns an AdjacencyList, not a Python list of lists.
cells = np.full(pts.shape[0], -1, dtype=np.int32)
if hasattr(colliding, "offsets") and hasattr(colliding, "array"):
    offsets, array = colliding.offsets, colliding.array
    for i in range(pts.shape[0]):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if end > start:
            cells[i] = int(array[start])
else:
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = int(links[0])

solution = np.full(pts.shape[0], np.nan, dtype=np.float64)
inside = cells >= 0
solution[inside] = np.asarray(uh.eval(pts[inside], cells[inside])).reshape(-1)
```

### Saving the output

```python
import numpy as np
np.savez("/scratch/solution.npz", solution=solution.astype("float64").reshape(<expected_shape>))
```

### Common pitfalls

* P1 on biharmonic problems: not valid. Use a C^0-IP method or a mixed pair.
* Equal-order Stokes: unstable. Use Taylor–Hood (P2/P1) or MINI.
* Forgetting `from petsc4py import PETSc` before `LinearProblem`.
* MPI parallelism: the container runs single-rank by default. Do not assume
  `MPI.COMM_WORLD.size > 1`.

### Common 0.9 / UFL 2024 API changes from older versions

The DOLFINx 0.9 and UFL 2024 releases renamed or removed several symbols
that pre-2024 codebases still use. If you instinctively reach for any of the
left-hand forms, use the right-hand form instead:

| Older (pre-2024) | Use in DOLFINx 0.9 / UFL 2024 |
|---|---|
| `ufl.atan_2(y, x)` | `ufl.atan2(y, x)` |
| `ufl.Cell("triangle", geometric_dimension=2)` | `ufl.Cell("triangle")` (drop `geometric_dimension=`) |
| `ufl.VectorElement("Lagrange", cell, 1)` | `basix.ufl.element("Lagrange", cell, 1, shape=(d,))` |
| `dolfinx.fem.petsc.assemble_matrix(...)` without import | first `import dolfinx.fem.petsc` (it is not auto-imported) |
| `u.vector.array[:] = ...` | `u.x.petsc_vec.array[:] = ...` |
| `assemble_matrix(a, bcs=[bc], diag=1.0)` | `assemble_matrix(a, bcs=[bc])` (drop `diag=`) |
| `apply_lifting(b, [a], bcs=[[bc]], scale=1.0)` | `apply_lifting(b, [a], bcs=[[bc]])` (drop `scale=`) |
| `NonlinearProblem(F, u, bcs, petsc_options_prefix="snes_")` | drop the kwarg; set the prefix on the SNES solver via `solver.setOptionsPrefix(...)` |

For time-dependent problems, **do not** write `ufl.diff(u, t)` to express
∂u/∂t. UFL `diff(...)` differentiates a UFL expression with respect to a UFL
`Variable` or `SpatialCoordinate`; the time variable `t` in a discretised
problem is a Python float, not a UFL object. Instead, introduce a
previous-step `Function` `u_n` and assemble the time-discretised residual,
for example with backward Euler:

```python
u_n = fem.Function(V)        # solution at previous step
dt = fem.Constant(m, 0.01)   # time step

# Backward-Euler heat residual: (u - u_n)/dt - laplacian(u) = f
F = ((u - u_n) / dt) * v * ufl.dx \
  + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
  - f * v * ufl.dx
```

### Anti-cheat reminder

The evaluator compares your `solution.npz` against a held-out reference. You
cannot reach the reference by reading any file outside `/scratch/`. Do not
attempt to bypass this; it will be detected.
