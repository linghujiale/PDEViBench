"""High-fidelity numerical references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_reference(case: dict[str, Any], *, out_root: Path) -> Path:
    """Build a high-fidelity reference for ``case``. Dispatches on geometry."""
    geo = case["case_spec"]["domain"]
    shape = geo.get("spec", {}).get("shape", "")
    op = case["case_spec"]["pde"]["operator_family"]
    bc_kind = case["case_spec"]["bc"]["kind"]

    if shape == "unit_square" and op in {
        "Poisson",
        "Helmholtz_low_k",
        "heat",
        "wave",
    } and bc_kind in {
        "dirichlet_homogeneous",
        "dirichlet_neumann_mixed",
    }:
        arr = _build_linear_unit_square(case)
    elif shape == "annulus" and op in {
        "Poisson",
        "Helmholtz_low_k",
        "heat",
        "wave",
    } and bc_kind in {
        "dirichlet_homogeneous",
        "dirichlet_neumann_mixed",
    }:
        arr = _build_linear_annulus(case)
    elif shape == "L_shape_2d" and op == "Poisson" and bc_kind == "dirichlet_homogeneous":
        arr = _build_poisson_lshape(case)
    elif shape == "square_with_hole" and op == "Poisson" and bc_kind == "dirichlet_homogeneous":
        arr = _build_poisson_square_with_hole(case)
    else:
        raise NotImplementedError(
            f"high_fidelity.build_reference does not yet support "
            f"(operator={op!r}, shape={shape!r}, bc={bc_kind!r})"
        )

    arr = _maybe_vector_lift(case, arr)
    expected = tuple(int(s) for s in case["case_spec"]["output"]["shape"])
    if arr.shape != expected:
        raise ValueError(
            f"high-fidelity reference shape {arr.shape} != expected {expected}"
        )
    out_path = out_root / case["reference"]["path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, solution=arr.astype(case["case_spec"]["output"]["dtype"]))
    return out_path


def _maybe_vector_lift(case: dict[str, Any], arr: np.ndarray) -> np.ndarray:
    expected = tuple(int(s) for s in case["case_spec"]["output"]["shape"])
    if len(expected) == 2 and expected[-1] == 2 and arr.ndim == 1:
        return np.stack([arr, 0.5 * arr], axis=-1)
    return arr


def _forcing_value(case: dict[str, Any], default: float = 1.0) -> float:
    params = case.get("case_spec", {}).get("pde", {}).get("parameters", {})
    return float(params.get("forcing_value", default))


def _forcing_xy_grid(case: dict[str, Any], n_grid: int) -> np.ndarray:
    params = case.get("case_spec", {}).get("pde", {}).get("parameters", {})
    profile = params.get("forcing_profile", "constant")
    value = _forcing_value(case)
    x = np.linspace(0.0, 1.0, n_grid)
    y = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if profile == "constant":
        return np.full((n_grid, n_grid), value, dtype=np.float64)
    if profile == "trig_poly_2d":
        a = float(params.get("forcing_a", 0.0))
        b = float(params.get("forcing_b", 0.0))
        freq_x = int(params.get("forcing_freq_x", 1))
        freq_y = int(params.get("forcing_freq_y", 1))
        return (
            value
            + a * np.sin(freq_x * np.pi * X) * np.cos(freq_y * np.pi * Y)
            + b * X * (1.0 - Y)
        ).astype(np.float64)
    raise ValueError(f"unsupported forcing_profile={profile!r}")


def _forcing_radial_grid(
    case: dict[str, Any],
    n_radial: int,
    *,
    r_in: float,
    r_out: float,
) -> np.ndarray:
    params = case.get("case_spec", {}).get("pde", {}).get("parameters", {})
    profile = params.get("forcing_profile", "constant")
    value = _forcing_value(case)
    r = np.linspace(r_in, r_out, n_radial)
    if profile == "constant":
        return np.full(n_radial, value, dtype=np.float64)
    if profile == "radial_poly":
        a = float(params.get("forcing_a", 0.0))
        b = float(params.get("forcing_b", 0.0))
        rho = (r - r_in) / max(r_out - r_in, 1e-30)
        return (value + a * (rho - 0.5) + b * rho * (1.0 - rho)).astype(np.float64)
    raise ValueError(f"unsupported radial forcing_profile={profile!r}")


def _unit_square_operator_system(
    *,
    n_grid: int,
    bc_kind: str,
    forcing_value: float = 1.0,
    forcing: np.ndarray | None = None,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    """Build A, b, mask for A u = f where A discretizes -Delta."""
    h = 1.0 / (n_grid - 1)
    inv_h2 = 1.0 / (h * h)
    rhs = (
        np.asarray(forcing, dtype=np.float64)
        if forcing is not None
        else np.full((n_grid, n_grid), forcing_value, dtype=np.float64)
    )

    def is_dirichlet(i: int, j: int) -> bool:
        if bc_kind == "dirichlet_homogeneous":
            return i == 0 or i == n_grid - 1 or j == 0 or j == n_grid - 1
        if bc_kind == "dirichlet_neumann_mixed":
            return i == 0 or j == 0 or j == n_grid - 1
        raise ValueError(f"unsupported unit_square bc_kind={bc_kind!r}")

    unk_mask = np.zeros((n_grid, n_grid), dtype=bool)
    for i in range(n_grid):
        for j in range(n_grid):
            unk_mask[i, j] = not is_dirichlet(i, j)

    unk_idx = np.full((n_grid, n_grid), -1, dtype=np.int64)
    unk_idx[unk_mask] = np.arange(int(unk_mask.sum()))
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(int(unk_mask.sum()), dtype=np.float64)

    for i in range(n_grid):
        for j in range(n_grid):
            if not unk_mask[i, j]:
                continue
            k = int(unk_idx[i, j])
            b[k] = rhs[i, j]
            rows.append(k); cols.append(k); vals.append(4.0 * inv_h2)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n_grid and 0 <= nj < n_grid:
                    if unk_mask[ni, nj]:
                        rows.append(k); cols.append(int(unk_idx[ni, nj])); vals.append(-inv_h2)
                    continue
                if bc_kind == "dirichlet_neumann_mixed" and ni == n_grid and 0 < j < n_grid - 1:
                    rows.append(k); cols.append(int(unk_idx[i - 1, j])); vals.append(-inv_h2)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(int(unk_mask.sum()), int(unk_mask.sum())))
    return A, b, unk_mask


def _unit_square_solve(
    *,
    n_grid: int,
    bc_kind: str,
    forcing_value: float = 1.0,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve -Delta u = f on the unit square with B0 or B1 conditions."""
    A, b, unk_mask = _unit_square_operator_system(
        n_grid=n_grid,
        bc_kind=bc_kind,
        forcing_value=forcing_value,
        forcing=forcing,
    )
    u_unk = spla.spsolve(A, b)
    u_full = np.zeros((n_grid, n_grid), dtype=np.float64)
    u_full[unk_mask] = u_unk
    return u_full


def _solve_linear_from_operator(
    *,
    A: sp.csc_matrix,
    b: np.ndarray,
    op: str,
    params: dict[str, Any],
    n_grid: int,
) -> np.ndarray:
    """Solve formula-blind linear scalar operators from the same -Delta matrix."""
    n = A.shape[0]
    I = sp.eye(n, format="csc")
    if op == "Poisson":
        return spla.spsolve(A, b)
    if op == "Helmholtz_low_k":
        k = float(params.get("k", 2.0))
        return spla.spsolve(A - (k * k) * I, b)
    if op == "heat":
        alpha = float(params.get("alpha", 1.0))
        T = float(params.get("T_final", 0.5))
        n_steps = max(80, 2 * (n_grid - 1))
        dt = T / n_steps
        lhs = (I + 0.5 * dt * alpha * A).tocsc()
        rhs_mat = (I - 0.5 * dt * alpha * A).tocsc()
        solve = spla.factorized(lhs)
        u = np.zeros(n, dtype=np.float64)
        for _ in range(n_steps):
            u = solve(rhs_mat @ u + dt * b)
        return u
    if op == "wave":
        c = float(params.get("c", 1.0))
        T = float(params.get("T_final", 1.0))
        beta = 0.25
        gamma = 0.5
        n_steps = max(160, 4 * (n_grid - 1))
        dt = T / n_steps
        lhs = (I + beta * dt * dt * c * c * A).tocsc()
        solve = spla.factorized(lhs)
        u = np.zeros(n, dtype=np.float64)
        v = np.zeros(n, dtype=np.float64)
        a = b - c * c * (A @ u)
        for _ in range(n_steps):
            pred = u + dt * v + dt * dt * (0.5 - beta) * a
            u_next = solve(pred + beta * dt * dt * b)
            a_next = b - c * c * (A @ u_next)
            v = v + dt * ((1.0 - gamma) * a + gamma * a_next)
            u, a = u_next, a_next
        return u
    raise ValueError(f"unsupported linear high-fidelity operator={op!r}")


def _build_linear_unit_square(case: dict[str, Any]) -> np.ndarray:
    spec = case["case_spec"]["eval_grid"]["spec"]
    nx_eval = int(spec["nx"])
    ny_eval = int(spec["ny"])
    assert nx_eval == ny_eval == 16, "this builder assumes 16x16 eval grid"
    bc_kind = case["case_spec"]["bc"]["kind"]
    forcing_value = _forcing_value(case)
    op = case["case_spec"]["pde"]["operator_family"]
    params = case["case_spec"]["pde"].get("parameters", {})
    n_levels = [16, 61, 121]
    u_by_n: dict[int, np.ndarray] = {}
    for n in n_levels:
        A, b, unk_mask = _unit_square_operator_system(
            n_grid=n,
            bc_kind=bc_kind,
            forcing_value=forcing_value,
            forcing=_forcing_xy_grid(case, n),
        )
        u_unk = _solve_linear_from_operator(A=A, b=b, op=op, params=params, n_grid=n)
        u_full = np.zeros((n, n), dtype=np.float64)
        u_full[unk_mask] = u_unk
        u_by_n[n] = u_full
    u_fine = u_by_n[n_levels[-1]]
    u_mid = u_by_n[n_levels[-2]]
    sub_mid_in_fine = (n_levels[-1] - 1) // (n_levels[-2] - 1)
    u_fine_on_mid = u_fine[::sub_mid_in_fine, ::sub_mid_in_fine]
    rel_l2_gap = float(
        np.linalg.norm(u_fine_on_mid - u_mid)
        / max(np.linalg.norm(u_mid), 1e-30)
    )
    if rel_l2_gap >= 0.05:
        raise ValueError(
            f"unit_square {op} convergence check FAILED: rel_l2 gap = {rel_l2_gap:.3e} >= 5%."
        )

    case.setdefault("_high_fidelity_info", {})["convergence_check"] = {
        "levels": list(n_levels),
        "gap_relative_l2": rel_l2_gap,
        "polynomial_order": 1,
    }
    sub_fine = (n_levels[-1] - 1) // (nx_eval - 1)
    u_eval = u_fine[::sub_fine, ::sub_fine]
    assert u_eval.shape == (nx_eval, ny_eval)
    return u_eval.reshape(-1).astype("float64")


def _annulus_radial_operator_system(
    *,
    n_radial: int,
    r_in: float,
    r_out: float,
    bc_kind: str,
    forcing_value: float = 1.0,
    forcing: np.ndarray | None = None,
) -> tuple[sp.csc_matrix, np.ndarray, list[int]]:
    """Build radial A, b, unknown indices where A discretizes -Delta."""
    r = np.linspace(r_in, r_out, n_radial)
    h = float(r[1] - r[0])
    rhs = (
        np.asarray(forcing, dtype=np.float64)
        if forcing is not None
        else np.full(n_radial, forcing_value, dtype=np.float64)
    )

    def is_dirichlet(i: int) -> bool:
        if bc_kind == "dirichlet_homogeneous":
            return i == 0 or i == n_radial - 1
        if bc_kind == "dirichlet_neumann_mixed":
            return i == 0
        raise ValueError(f"unsupported annulus bc_kind={bc_kind!r}")

    unk = [i for i in range(n_radial) if not is_dirichlet(i)]
    idx = {i: k for k, i in enumerate(unk)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(len(unk), dtype=np.float64)

    for i in unk:
        k = idx[i]
        b[k] = rhs[i]
        ri = float(r[i])
        rm = ri - 0.5 * h
        rp = ri + 0.5 * h
        if i == n_radial - 1 and bc_kind == "dirichlet_neumann_mixed":
            coeff_m = rm / (ri * h * h)
            rows.append(k); cols.append(k); vals.append(coeff_m)
            rows.append(k); cols.append(idx[i - 1]); vals.append(-coeff_m)
            continue

        coeff_m = rm / (ri * h * h)
        coeff_p = rp / (ri * h * h)
        rows.append(k); cols.append(k); vals.append(coeff_m + coeff_p)
        if i - 1 in idx:
            rows.append(k); cols.append(idx[i - 1]); vals.append(-coeff_m)
        if i + 1 in idx:
            rows.append(k); cols.append(idx[i + 1]); vals.append(-coeff_p)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(len(unk), len(unk)))
    return A, b, unk


def _annulus_radial_solve(
    *,
    n_radial: int,
    r_in: float,
    r_out: float,
    bc_kind: str,
    forcing_value: float = 1.0,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve radial -1/r d/dr(r du/dr) = f on an annulus."""
    A, b, unk = _annulus_radial_operator_system(
        n_radial=n_radial,
        r_in=r_in,
        r_out=r_out,
        bc_kind=bc_kind,
        forcing_value=forcing_value,
        forcing=forcing,
    )
    idx = {i: k for k, i in enumerate(unk)}
    u_unk = spla.spsolve(A, b)
    u = np.zeros(n_radial, dtype=np.float64)
    for i, k in idx.items():
        u[i] = u_unk[k]
    return u


def _build_linear_annulus(case: dict[str, Any]) -> np.ndarray:
    spec = case["case_spec"]["eval_grid"]["spec"]
    n_radial_eval = int(spec["n_radial"])
    n_angular = int(spec["n_angular"])
    r_in = float(spec.get("r_in", 0.5))
    r_out = float(spec.get("r_out", 1.0))
    bc_kind = case["case_spec"]["bc"]["kind"]
    forcing_value = _forcing_value(case)
    op = case["case_spec"]["pde"]["operator_family"]
    params = case["case_spec"]["pde"].get("parameters", {})

    n_levels = [8, 57, 113]
    u_by_n: dict[int, np.ndarray] = {}
    for n in n_levels:
        A, b, unk = _annulus_radial_operator_system(
            n_radial=n,
            r_in=r_in,
            r_out=r_out,
            bc_kind=bc_kind,
            forcing_value=forcing_value,
            forcing=_forcing_radial_grid(case, n, r_in=r_in, r_out=r_out),
        )
        u_unk = _solve_linear_from_operator(A=A, b=b, op=op, params=params, n_grid=n)
        u = np.zeros(n, dtype=np.float64)
        for i, k in {i: k for k, i in enumerate(unk)}.items():
            u[i] = u_unk[k]
        u_by_n[n] = u
    u_fine = u_by_n[n_levels[-1]]
    u_mid = u_by_n[n_levels[-2]]
    sub_mid_in_fine = (n_levels[-1] - 1) // (n_levels[-2] - 1)
    u_fine_on_mid = u_fine[::sub_mid_in_fine]
    rel_l2_gap = float(
        np.linalg.norm(u_fine_on_mid - u_mid)
        / max(np.linalg.norm(u_mid), 1e-30)
    )
    if rel_l2_gap >= 0.05:
        raise ValueError(
            f"annulus {op} convergence check FAILED: rel_l2 gap = {rel_l2_gap:.3e} >= 5%."
        )

    case.setdefault("_high_fidelity_info", {})["convergence_check"] = {
        "levels": list(n_levels),
        "gap_relative_l2": rel_l2_gap,
        "polynomial_order": 1,
    }
    sub_fine = (n_levels[-1] - 1) // (n_radial_eval - 1)
    u_eval_radial = u_fine[::sub_fine]
    assert u_eval_radial.shape == (n_radial_eval,)
    arr = np.repeat(u_eval_radial[:, None], n_angular, axis=1)
    return arr.reshape(-1).astype("float64")


def _lshape_solve(
    *,
    n_grid: int,
    forcing_value: float = 1.0,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve −Δu = 1 on the grid-aligned L-shape with u = 0 on ∂L.

    Grid-aligned L-shape convention:
      Ω = (0,1)² \\ [x_cut, 1]²  with  x_cut = 8/15 ≈ 0.5333.
    The cut boundary at x_cut = 8/15 aligns *exactly* with the 16-point
    eval grid (eval-grid point i=8 sits on the cut boundary). The canonical
    continuous L-shape uses x_cut = 0.5; we align to the grid so the same
    discrete domain is representable consistently at every refinement
    level, allowing the convergence check to compare like with like.
    x_cut = 0.5 geometry.

    Requirement: ``(n_grid - 1)`` must be divisible by 15 so that
    ``cut_int = 8 * (n_grid - 1) // 15`` is an integer aligned with the
    16-eval-grid every-h_eval subsampling.

    Returns the full ``n_grid × n_grid`` solution array, with NaN at points
    strictly inside the cut quadrant (i ≥ cut_int AND j ≥ cut_int).
    Boundary nodes (those in_omega adjacent to a not-in-omega neighbour)
    are set to Dirichlet 0; interior unknowns are solved.
    """
    h = 1.0 / (n_grid - 1)
    rhs = (
        np.asarray(forcing, dtype=np.float64)
        if forcing is not None
        else np.full((n_grid, n_grid), forcing_value, dtype=np.float64)
    )
    assert (n_grid - 1) % 15 == 0, "n_grid - 1 must be divisible by 15 for grid alignment"
    cut_int = 8 * (n_grid - 1) // 15
    in_omega = np.ones((n_grid, n_grid), dtype=bool)
    cut_mask = np.zeros_like(in_omega)
    cut_mask[cut_int:, cut_int:] = True
    in_omega &= ~cut_mask

    bc_dirichlet = np.zeros_like(in_omega)
    for i in range(n_grid):
        for j in range(n_grid):
            if not in_omega[i, j]:
                continue
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if not (0 <= ni < n_grid and 0 <= nj < n_grid) or not in_omega[ni, nj]:
                    bc_dirichlet[i, j] = True
                    break

    unk_mask = in_omega & (~bc_dirichlet)
    unk_idx = np.full((n_grid, n_grid), -1, dtype=np.int64)
    unk_idx[unk_mask] = np.arange(int(unk_mask.sum()))
    N_unk = int(unk_mask.sum())

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(N_unk)
    inv_h2 = 1.0 / (h ** 2)

    for i in range(n_grid):
        for j in range(n_grid):
            if not unk_mask[i, j]:
                continue
            k = unk_idx[i, j]
            rows.append(k); cols.append(k); vals.append(4.0 * inv_h2)
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n_grid and 0 <= nj < n_grid and in_omega[ni, nj]:
                    if unk_mask[ni, nj]:
                        rows.append(k); cols.append(unk_idx[ni, nj]); vals.append(-inv_h2)
            b[k] = rhs[i, j]

    A = sp.csc_matrix((vals, (rows, cols)), shape=(N_unk, N_unk))
    u_unk = spla.spsolve(A, b)
    u_full = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
    u_full[bc_dirichlet] = 0.0
    u_full[unk_mask] = u_unk
    return u_full


def _build_poisson_lshape(case: dict[str, Any]) -> np.ndarray:
    """Build the L-shape Poisson reference at the case's eval grid resolution.

    Strategy:
    1. Solve at refined internal mesh ``n_fine``.
    2. Solve at coarser internal mesh ``n_coarse``.
    3. Compute the L² gap between the two (restricted to common points).
    4. If gap < 5%, accept the fine solution.
    5. Subsample the fine solution to the case's 16×16 eval grid.

    NaN at masked (cut-region) points; the evaluator's accuracy stage
    must skip NaN comparisons.
    """
    spec = case["case_spec"]["eval_grid"]["spec"]
    nx_eval = int(spec["nx"])
    ny_eval = int(spec["ny"])
    assert nx_eval == ny_eval == 16, "this builder assumes 16×16 eval grid"

    n_levels = [16, 61, 121]
    n_fine = n_levels[-1]
    sub_fine = (n_fine - 1) // (nx_eval - 1)
    assert sub_fine * (nx_eval - 1) == n_fine - 1, "eval-grid not exactly subsampled"

    u_by_n: dict[int, np.ndarray] = {
        n: _lshape_solve(
            n_grid=n,
            forcing_value=_forcing_value(case),
            forcing=_forcing_xy_grid(case, n),
        )
        for n in n_levels
    }

    u_fine = u_by_n[n_levels[-1]]
    u_mid = u_by_n[n_levels[-2]]
    sub_mid_in_fine = (n_levels[-1] - 1) // (n_levels[-2] - 1)
    u_fine_on_mid = u_fine[::sub_mid_in_fine, ::sub_mid_in_fine]
    mask = ~np.isnan(u_fine_on_mid) & ~np.isnan(u_mid)
    rel_l2_gap = float(
        np.linalg.norm((u_fine_on_mid - u_mid)[mask])
        / max(np.linalg.norm(u_mid[mask]), 1e-30)
    )
    if rel_l2_gap >= 0.05:
        raise ValueError(
            f"L-shape high-fidelity convergence check FAILED: rel_l2 gap = {rel_l2_gap:.3e} ≥ 5%. "
            f"Refine more."
        )

    case.setdefault("_high_fidelity_info", {})["convergence_check"] = {
        "levels": list(n_levels),
        "gap_relative_l2": rel_l2_gap,
        "polynomial_order": 1,
    }

    u_eval = u_fine[::sub_fine, ::sub_fine]
    assert u_eval.shape == (nx_eval, ny_eval)
    return u_eval.reshape(-1).astype("float64")


def _square_with_hole_in_omega(n_grid: int, *, center: tuple[float, float], radius: float) -> np.ndarray:
    """Discrete in_omega mask: True iff (x_i, y_j) is strictly outside the
    closed disk of radius ``radius`` around ``center``.

    Both the reference (multi-level) and the agent's coarse solver use this
    same mask function, so the discrete hole agrees at every resolution
    (modulo stair-stepping). NaN at points inside the disk.
    """
    coords = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    dist2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    in_disk = dist2 <= radius ** 2
    return ~in_disk


def _square_with_hole_solve(
    *,
    n_grid: int,
    center: tuple[float, float],
    radius: float,
    forcing_value: float = 1.0,
    forcing: np.ndarray | None = None,
) -> np.ndarray:
    """Solve −Δu = f on Ω = (0,1)² \\ B(center, radius) with u=0 on ∂Ω."""
    h = 1.0 / (n_grid - 1)
    rhs = (
        np.asarray(forcing, dtype=np.float64)
        if forcing is not None
        else np.full((n_grid, n_grid), forcing_value, dtype=np.float64)
    )
    in_omega = _square_with_hole_in_omega(n_grid, center=center, radius=radius)

    bc_dirichlet = np.zeros_like(in_omega)
    for i in range(n_grid):
        for j in range(n_grid):
            if not in_omega[i, j]:
                continue
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if not (0 <= ni < n_grid and 0 <= nj < n_grid) or not in_omega[ni, nj]:
                    bc_dirichlet[i, j] = True
                    break

    unk_mask = in_omega & (~bc_dirichlet)
    unk_idx = np.full((n_grid, n_grid), -1, dtype=np.int64)
    unk_idx[unk_mask] = np.arange(int(unk_mask.sum()))
    N_unk = int(unk_mask.sum())

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(N_unk)
    inv_h2 = 1.0 / (h ** 2)

    for i in range(n_grid):
        for j in range(n_grid):
            if not unk_mask[i, j]:
                continue
            k = unk_idx[i, j]
            rows.append(k); cols.append(k); vals.append(4.0 * inv_h2)
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n_grid and 0 <= nj < n_grid and in_omega[ni, nj]:
                    if unk_mask[ni, nj]:
                        rows.append(k); cols.append(unk_idx[ni, nj]); vals.append(-inv_h2)
            b[k] = rhs[i, j]

    A = sp.csc_matrix((vals, (rows, cols)), shape=(N_unk, N_unk))
    u_unk = spla.spsolve(A, b)
    u_full = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
    u_full[bc_dirichlet] = 0.0
    u_full[unk_mask] = u_unk
    return u_full


def _build_poisson_square_with_hole(case: dict[str, Any]) -> np.ndarray:
    """High-fidelity reference for Poisson on unit_square with a circular hole.

    Domain Ω = (0,1)² \\ B(center, radius). Center and radius read from the
    case's domain spec. Both reference and solver use the *same* discrete
    "all grid points with (x-cx)² + (y-cy)² > r²" rule, so the NaN-mask
    agrees at every resolution.

    The hole boundary is *smooth* (no re-entrant corner), so FD convergence
    is O(h²) globally — better than L-shape. Convergence check at n=16,
    61, 121 (same as L-shape).
    """
    domain_spec = case["case_spec"]["domain"]["spec"]
    center = tuple(domain_spec.get("hole_center", [0.5, 0.5]))
    radius = float(domain_spec.get("hole_radius", 0.2))
    assert len(center) == 2

    n_levels = [16, 61, 121]
    n_fine = n_levels[-1]
    sub_fine = (n_fine - 1) // 15
    assert sub_fine * 15 == n_fine - 1, "eval-grid not exactly subsampled"

    u_by_n: dict[int, np.ndarray] = {
        n: _square_with_hole_solve(
            n_grid=n,
            center=center,
            radius=radius,
            forcing_value=_forcing_value(case),
            forcing=_forcing_xy_grid(case, n),
        )
        for n in n_levels
    }

    u_fine = u_by_n[n_levels[-1]]
    u_mid = u_by_n[n_levels[-2]]
    sub_mid_in_fine = (n_levels[-1] - 1) // (n_levels[-2] - 1)
    u_fine_on_mid = u_fine[::sub_mid_in_fine, ::sub_mid_in_fine]
    mask = ~np.isnan(u_fine_on_mid) & ~np.isnan(u_mid)
    rel_l2_gap = float(
        np.linalg.norm((u_fine_on_mid - u_mid)[mask])
        / max(np.linalg.norm(u_mid[mask]), 1e-30)
    )
    if rel_l2_gap >= 0.05:
        raise ValueError(
            f"square_with_hole convergence check FAILED: rel_l2 gap = {rel_l2_gap:.3e} ≥ 5%."
        )

    case.setdefault("_high_fidelity_info", {})["convergence_check"] = {
        "levels": list(n_levels),
        "gap_relative_l2": rel_l2_gap,
        "polynomial_order": 1,
    }

    u_eval = u_fine[::sub_fine, ::sub_fine]
    assert u_eval.shape == (16, 16)
    return u_eval.reshape(-1).astype("float64")


__all__ = ["build_reference"]
