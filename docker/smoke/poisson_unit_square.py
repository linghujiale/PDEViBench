"""Smoke test: solve Poisson on unit square with DOLFINx."""

from __future__ import annotations

import sys

import numpy as np
from mpi4py import MPI

import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem


def main() -> int:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 16, 16, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    boundary_dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
    )
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = 2 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    err = uh.x.array - u_exact.x.array
    rel_l2 = np.linalg.norm(err) / max(np.linalg.norm(u_exact.x.array), 1e-30)

    print(f"dolfinx-smoke: rel_l2={rel_l2:.3e}")
    return 0 if rel_l2 < 5e-2 else 2


if __name__ == "__main__":
    sys.exit(main())
