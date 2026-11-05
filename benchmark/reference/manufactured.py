"""Manufactured-solution references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp


def build_reference(case: dict[str, Any], *, out_root: Path) -> Path:
    """Build and write the manufactured reference for ``case``.

    Dispatches on ``(operator_family, geometry_shape, bc_kind)``. Supports:
    * G0 / unit_square × B0 / dirichlet_homogeneous × all five O0 families.
    * G0 / unit_square × B1 / dirichlet_neumann_mixed × {Poisson}.
    * G1 / annulus    × B0 / dirichlet_homogeneous × {Poisson}.

    Any other combination raises ``NotImplementedError``.
    """
    op = case["case_spec"]["pde"]["operator_family"]
    geo = case["case_spec"]["domain"]
    geo_shape = geo.get("spec", {}).get("shape", "")
    bc_kind = case["case_spec"]["bc"]["kind"]

    if geo_shape == "sector":
        if bc_kind != "dirichlet_homogeneous":
            raise NotImplementedError(
                f"manufactured.py sector only supports B0; got {bc_kind}"
            )
        builder = _SECTOR_BUILDERS.get(op)
        if builder is None:
            raise NotImplementedError(
                f"manufactured.py has no sector builder for operator={op!r}"
            )
        arr = builder(case)
    elif geo_shape == "annulus":
        if bc_kind == "dirichlet_homogeneous":
            builder = _ANNULUS_B0_BURGERS_BUILDERS.get(op) or _ANNULUS_BUILDERS.get(op)
        elif bc_kind == "dirichlet_neumann_mixed":
            builder = _ANNULUS_B1_BUILDERS.get(op)
        else:
            raise NotImplementedError(
                f"manufactured.py annulus supports only B0 or B1; got {bc_kind}"
            )
        if builder is None:
            raise NotImplementedError(
                f"manufactured.py has no (annulus, {bc_kind}) builder for operator={op!r}"
            )
        arr = builder(case)
    elif geo_shape == "unit_square":
        if bc_kind == "dirichlet_homogeneous":
            builder = _BUILDERS.get(op)
        elif bc_kind == "dirichlet_neumann_mixed":
            builder = _B1_BUILDERS.get(op)
        else:
            raise NotImplementedError(
                f"manufactured.py unit_square supports only B0 or B1; got {bc_kind}"
            )
        if builder is None:
            raise NotImplementedError(
                f"manufactured.py has no (unit_square, {bc_kind}) builder for operator={op!r}"
            )
        arr = builder(case)
    else:
        raise NotImplementedError(
            f"manufactured.py geometry={geo_shape!r} not yet implemented"
        )

    arr = _maybe_vector_lift(case, arr)
    expected = tuple(int(s) for s in case["case_spec"]["output"]["shape"])
    if arr.shape != expected:
        raise ValueError(
            f"manufactured builder for ({op}, {geo_shape}) returned shape "
            f"{arr.shape}, expected {expected}"
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


def _grid_xy(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spec = case["case_spec"]["eval_grid"]["spec"]
    nx = int(spec["nx"])
    ny = int(spec["ny"])
    x = np.linspace(float(spec.get("x0", 0.0)), float(spec.get("x1", 1.0)), nx)
    y = np.linspace(float(spec.get("y0", 0.0)), float(spec.get("y1", 1.0)), ny)
    return np.meshgrid(x, y, indexing="xy")


def _build_poisson(case: dict[str, Any]) -> np.ndarray:
    """MMS Poisson on unit_square B0 — NON-EIGENMODE u via exp-shifted sin·sin.

    Vanishes on ∂Ω (via sin factors); NOT a Laplacian eigenfunction (the exp
    factor shifts the spectrum). Forcing f is a linear combination of
    sin(πx)sin(πy) and sin(π(x+y)), so the agent cannot read u_exact off of f's
    algebraic structure.
    """
    X, Y = _grid_xy(case)
    u = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(X + Y - 1.0)
    return u.reshape(-1).astype("float64")


def _build_helmholtz_low_k(case: dict[str, Any]) -> np.ndarray:
    """MMS Helmholtz_low_k on unit_square B0 — same non-eigenmode u as Poisson."""
    X, Y = _grid_xy(case)
    u = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(X + Y - 1.0)
    return u.reshape(-1).astype("float64")


def _build_heat(case: dict[str, Any]) -> np.ndarray:
    """MMS heat on unit_square B0 — non-eigenmode u × non-decay time factor.

    u(x,y,t) = (1 + t²) · sin(πx)·sin(πy)·exp(x+y−1).
    IC u(·,·,0) = sin(πx)·sin(πy)·exp(x+y-1); time factor grows
    (not decays) so the agent cannot template-match "heat ≈ exp(-λt)".
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params["T_final"])
    X, Y = _grid_xy(case)
    u = (1.0 + T ** 2) * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(X + Y - 1.0)
    return u.reshape(-1).astype("float64")


def _build_wave(case: dict[str, Any]) -> np.ndarray:
    """MMS wave on unit_square B0 — non-eigenmode u × non-oscillating time factor.

    u(x,y,t) = (1 + t²/2) · sin(πx)·sin(πy)·exp(x+y−1).
    ∂_t u(·,·,0) = 0 (v₀ = 0). No cos(ωt) structure to template-match.
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params["T_final"])
    X, Y = _grid_xy(case)
    u = (1.0 + 0.5 * T ** 2) * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(X + Y - 1.0)
    return u.reshape(-1).astype("float64")


def _build_burgers(case: dict[str, Any]) -> np.ndarray:
    """MMS scalar viscous Burgers (O1) on unit_square + Dirichlet 0.

    Manufactured u(x, y, t) = exp(−α t) sin(π x) sin(π y), with α = 1.

    Burgers PDE  ∂_t u + u (∂_x u + ∂_y u) − ν Δu = f  is forced; the
    forcing absorbs the algebraic residual so that the manufactured u is
    *exactly* the solution. Reference is u at t = T_final.

    SymPy-derived forcing (general):
        f = e^{-2αt} sin(πx) sin(πy) [(−α + 2π² ν) e^{αt} + π sin(π(x+y))]
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params["T_final"])
    alpha = 1.0
    X, Y = _grid_xy(case)
    u = np.exp(-alpha * T) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    return u.reshape(-1).astype("float64")


def _build_linear_elasticity(case: dict[str, Any]) -> np.ndarray:
    """MMS linear elasticity on unit_square B0 — non-eigenmode u₁, u₂=0.

    u₁(x,y) = sin(πx)·sin(πy)·exp(x+y−1),  u₂ = 0.
    SymPy-derived body force has terms in sin·sin, sin(π(x+y)), and
    cos(πx)cos(πy); see forcing_descriptor for the closed form.
    """
    X, Y = _grid_xy(case)
    u1 = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(X + Y - 1.0)
    u2 = np.zeros_like(u1)
    arr = np.stack([u1.reshape(-1), u2.reshape(-1)], axis=-1)
    return arr.astype("float64")


_BUILDERS = {
    "Poisson": _build_poisson,
    "Helmholtz_low_k": _build_helmholtz_low_k,
    "heat": _build_heat,
    "wave": _build_wave,
    "linear_elasticity": _build_linear_elasticity,
    "Burgers": _build_burgers,
}


def _build_poisson_b1_mixed(case: dict[str, Any]) -> np.ndarray:
    """Setup:  Γ_D = {x=0, y=0, y=1}  →  u = 0,
            Γ_N = {x=1}             →  ∂u/∂x = 0.

        u_exact(x, y) = sin(π x / 2) · sin(π y) · x · (2 − x).

    Boundary check:
        u(0, y) = 0  (x = 0 factor)
        u(x, 0) = u(x, 1) = 0  (sin(πy) factor)
        ∂u/∂x|_{x=1}: the x(2-x) factor has derivative 2 - 2x = 0 at x=1,
        and sin(πx/2) has derivative (π/2)cos(π/2) = 0 at x=1, so
        the product rule gives ∂u/∂x|_{x=1} = 0 (Neumann).

    The x(2-x) factor breaks the Laplacian eigenmode structure: f is no longer
    proportional to u.
    """
    X, Y = _grid_xy(case)
    u = np.sin(np.pi * X / 2.0) * np.sin(np.pi * Y) * X * (2.0 - X)
    return u.reshape(-1).astype("float64")


def _build_helmholtz_b1_mixed(case: dict[str, Any]) -> np.ndarray:
    """Helmholtz_low_k on unit_square + B1 — same non-eigenmode u as Poisson_B1."""
    X, Y = _grid_xy(case)
    u = np.sin(np.pi * X / 2.0) * np.sin(np.pi * Y) * X * (2.0 - X)
    return u.reshape(-1).astype("float64")


def _build_heat_b1_mixed(case: dict[str, Any]) -> np.ndarray:
    """Heat on unit_square + B1 — non-eigenmode u × non-decay time factor.

    u(x, y, t) = (1 + t²) · sin(πx/2) · sin(πy) · x(2 − x).
    IC u(·,·,0) = u_spatial; reference is u at T_final.
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 0.5))
    X, Y = _grid_xy(case)
    u = (1.0 + T ** 2) * np.sin(np.pi * X / 2.0) * np.sin(np.pi * Y) * X * (2.0 - X)
    return u.reshape(-1).astype("float64")


def _build_wave_b1_mixed(case: dict[str, Any]) -> np.ndarray:
    """Wave on unit_square + B1 — non-eigenmode u × non-oscillating time factor.

    u(x, y, t) = (1 + t²/2) · sin(πx/2) · sin(πy) · x(2 − x).
    ∂_t u(·,·,0) = 0 (v₀ = 0).
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 1.0))
    X, Y = _grid_xy(case)
    u = (1.0 + 0.5 * T ** 2) * np.sin(np.pi * X / 2.0) * np.sin(np.pi * Y) * X * (2.0 - X)
    return u.reshape(-1).astype("float64")


def _build_burgers_b1_mixed(case: dict[str, Any]) -> np.ndarray:
    """u(x, y, t) = e^(−αt) · sin(πx/2) · sin(πy) · x(2 − x),  α=1, ν=0.1, T=0.5.
    BC: Dirichlet 0 on {x=0, y=0, y=1}; ∂u/∂x = 0 on {x=1}.
    SymPy-derived forcing in the reference solver matches Cartesian Burgers.
    Reference at T=0.5: e^(−0.5) · sin(πx/2) · sin(πy) · x(2−x).
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params["T_final"])
    alpha = 1.0
    X, Y = _grid_xy(case)
    u = np.exp(-alpha * T) * np.sin(np.pi * X / 2.0) * np.sin(np.pi * Y) * X * (2.0 - X)
    return u.reshape(-1).astype("float64")


_B1_BUILDERS = {
    "Poisson": _build_poisson_b1_mixed,
    "Helmholtz_low_k": _build_helmholtz_b1_mixed,
    "heat": _build_heat_b1_mixed,
    "wave": _build_wave_b1_mixed,
    "Burgers": _build_burgers_b1_mixed,
}


def _grid_polar_annulus(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate-grid sampling for an annulus eval grid.

    Returns (R, Theta, meta) where ``R`` and ``Theta`` are the polar
    coordinates on the eval grid, in row-major (n_radial, n_angular) order.
    """
    spec = case["case_spec"]["eval_grid"]["spec"]
    n_r = int(spec["n_radial"])
    n_a = int(spec["n_angular"])
    r_in = float(spec["r_in"])
    r_out = float(spec["r_out"])
    r = np.linspace(r_in, r_out, n_r)
    theta = np.linspace(0.0, 2.0 * np.pi, n_a, endpoint=False)
    R, T = np.meshgrid(r, theta, indexing="ij")
    return R, T, {"n_r": n_r, "n_a": n_a, "r_in": r_in, "r_out": r_out}


def _build_poisson_annulus(case: dict[str, Any]) -> np.ndarray:
    """MMS Poisson on annulus with homogeneous Dirichlet.

    Choose:    u(r, θ) = sin(π (r − r_in) / (r_out − r_in)) · cos(2 θ).
    Vanishes at r = r_in and r = r_out (Dirichlet-0 on both circular
    boundaries) and is 2π-periodic in θ.

    Forcing:   f = −Δu in polar coords,
               Δu = (1/r) ∂_r(r ∂_r u) + (1/r²) ∂²u/∂θ².
    Sympy-derived expression is embedded in compose.py via
    forcing_descriptor(operator_family='Poisson_annulus', ...).
    """
    R, T, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    radial = np.sin(np.pi * (R - r_in) / L)
    angular = np.cos(2.0 * T)
    u = radial * angular
    return u.reshape(-1).astype("float64")


def _build_helmholtz_annulus(case: dict[str, Any]) -> np.ndarray:
    """Helmholtz_low_k on annulus + B0.

    Same eigenmode-like u as Poisson_annulus B0:
        u_exact(r, θ) = sin(π(r − r_in)/L) cos(2θ).
    SymPy-derived forcing (k=2, specialised at r_in=0.5, r_out=1.0):
        f = (cos(2θ)/r²) [(4 − π²) r² sin(2πr) + 2π r cos(2πr)·sign − 4 sin(2πr)]·(-1)
    (Note: with r_in=0.5, L=0.5, sin(π(r−r_in)/L) = sin(2π(r−0.5)) = −sin(2πr),
    which the symbolic form below absorbs.)
    """
    R, T, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    radial = np.sin(np.pi * (R - r_in) / L)
    angular = np.cos(2.0 * T)
    u = radial * angular
    return u.reshape(-1).astype("float64")


def _build_heat_annulus(case: dict[str, Any]) -> np.ndarray:
    """Heat equation on annulus + B0.

    u(r, θ, t) = e^(−t) sin(π(r − r_in)/L) cos(2θ); α=1; forced so manufactured u is exact.
    Reference at T_final (default 0.5):  e^(−T_final) sin(π(r-r_in)/L) cos(2θ).
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 0.5))
    R, Th, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    u = np.exp(-T) * np.sin(np.pi * (R - r_in) / L) * np.cos(2.0 * Th)
    return u.reshape(-1).astype("float64")


def _build_wave_annulus(case: dict[str, Any]) -> np.ndarray:
    """Wave equation on annulus + B0.

    u(r, θ, t) = cos(t) sin(π(r − r_in)/L) cos(2θ); c=1.
    IC: u(·,·,0) = sin(...)cos(2θ); ∂_t u(·,·,0) = 0.
    Reference at T_final (default 1.0): cos(T_final) sin(π(r-r_in)/L) cos(2θ).
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 1.0))
    R, Th, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    u = np.cos(T) * np.sin(np.pi * (R - r_in) / L) * np.cos(2.0 * Th)
    return u.reshape(-1).astype("float64")


_ANNULUS_BUILDERS = {
    "Poisson": _build_poisson_annulus,
    "Helmholtz_low_k": _build_helmholtz_annulus,
    "heat": _build_heat_annulus,
    "wave": _build_wave_annulus,
}


def _build_poisson_annulus_b1(case: dict[str, Any]) -> np.ndarray:
    """MMS Poisson on annulus with mixed Dirichlet (inner) + Neumann (outer).

    Setup:    Γ_D = {r = r_in}  →  u = 0,
              Γ_N = {r = r_out} →  ∂u/∂r = 0.

    Manufactured u_exact(r, θ) = sin(π (r − r_in) / (2 (r_out − r_in))) · cos(2 θ).
        At r = r_in:  sin(0) = 0  ⇒ Dirichlet.
        At r = r_out: ∂u/∂r = (π/(2L)) cos(π/2) cos(2θ) = 0  ⇒ Neumann.
        2π-periodic in θ.

    Forcing (specialized at r_in=0.5, r_out=1.0):
        f(r, θ) = −(π² r² cos(π r) + π r sin(π r) + 4 cos(π r)) · cos(2 θ) / r²
    (Derived in SymPy; see forcing_descriptor for the symbolic form.)
    """
    R, T, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    radial = np.sin(np.pi * (R - r_in) / (2.0 * L))
    angular = np.cos(2.0 * T)
    u = radial * angular
    return u.reshape(-1).astype("float64")


def _build_helmholtz_annulus_b1(case: dict[str, Any]) -> np.ndarray:
    """Helmholtz_low_k on annulus + B1.

    Same eigenmode-like manufactured u as Poisson_annulus_b1:
        u_exact(r, θ) = sin(π (r − r_in)/(2L)) cos(2θ),  L = r_out − r_in.
    At r_in=0.5, r_out=1.0 this reduces to −cos(π r) cos(2θ).
    Inner Dirichlet, outer Neumann (∂u/∂r = 0).
    SymPy-derived forcing (k=2 specialised):
        f = cos(2θ) [(4 − π²) r² cos(π r) − π r sin(π r) − 4 cos(π r)] / r².
    """
    R, T, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    radial = np.sin(np.pi * (R - r_in) / (2.0 * L))
    angular = np.cos(2.0 * T)
    u = radial * angular
    return u.reshape(-1).astype("float64")


def _build_heat_annulus_b1(case: dict[str, Any]) -> np.ndarray:
    """Heat on annulus + B1.

    u(r, θ, t) = e^(−t) sin(π(r − r_in)/(2L)) cos(2θ);
    forced so that the manufactured u is exactly the PDE solution.
    Reference at T = T_final (default 0.5):
        u = e^(−T) sin(π(r − r_in)/(2L)) cos(2θ).
    At r_in=0.5, r_out=1.0:  u = −e^(−T) cos(π r) cos(2θ).
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 0.5))
    R, Th, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    u = np.exp(-T) * np.sin(np.pi * (R - r_in) / (2.0 * L)) * np.cos(2.0 * Th)
    return u.reshape(-1).astype("float64")


def _build_wave_annulus_b1(case: dict[str, Any]) -> np.ndarray:
    """Wave on annulus + B1.

    u(r, θ, t) = cos(t) sin(π(r − r_in)/(2L)) cos(2θ); c=1.
    IC: u(·,·,0) = sin(...)cos(2θ); ∂_t u(·,·,0) = 0.
    Reference at T = T_final (default 1.0):
        u = cos(T) sin(π(r − r_in)/(2L)) cos(2θ).
    At r_in=0.5, r_out=1.0:  u = −cos(T) cos(π r) cos(2θ).
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 1.0))
    R, Th, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    u = np.cos(T) * np.sin(np.pi * (R - r_in) / (2.0 * L)) * np.cos(2.0 * Th)
    return u.reshape(-1).astype("float64")


def _build_burgers_annulus_b1(case: dict[str, Any]) -> np.ndarray:
    """Manufactured u(r, θ, t) = e^(−t) sin(π(r − r_in)/(2L)) cos(2θ), α=1.
    At r_in=0.5, r_out=1.0: u = −e^(−t) cos(πr) cos(2θ).
    Inner Dirichlet (r=r_in), outer Neumann ∂u/∂r=0 (r=r_out).
    Forcing (SymPy-derived) absorbs the algebraic residual.
    Reference at T = T_final (default 0.5).
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 0.5))
    R, Th, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    u = np.exp(-T) * np.sin(np.pi * (R - r_in) / (2.0 * L)) * np.cos(2.0 * Th)
    return u.reshape(-1).astype("float64")


_ANNULUS_B1_BUILDERS = {
    "Poisson": _build_poisson_annulus_b1,
    "Helmholtz_low_k": _build_helmholtz_annulus_b1,
    "heat": _build_heat_annulus_b1,
    "wave": _build_wave_annulus_b1,
    "Burgers": _build_burgers_annulus_b1,
}


def _build_burgers_annulus(case: dict[str, Any]) -> np.ndarray:
    """Manufactured u(r, θ, t) = e^(-α t) sin(π(r-r_in)/(r_out-r_in)) cos(θ),
    α=1, ν=0.1, T=0.5, r_in=0.5, r_out=1.
    BC: Dirichlet 0 on both circular boundaries (B0).
    Reference at T=0.5: −e^(-0.5) sin(2πr) cos(θ).  (The minus sign arises
    because sin(2π(r-0.5)) = −sin(2πr) on r ∈ [0.5, 1].)
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params["T_final"])
    alpha = 1.0
    R, Th, meta = _grid_polar_annulus(case)
    r_in, r_out = meta["r_in"], meta["r_out"]
    L = r_out - r_in
    u = np.exp(-alpha * T) * np.sin(np.pi * (R - r_in) / L) * np.cos(Th)
    return u.reshape(-1).astype("float64")


_ANNULUS_B0_BURGERS_BUILDERS = {
    "Burgers": _build_burgers_annulus,
}


def _grid_polar_sector(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = case["case_spec"]["eval_grid"]["spec"]
    n_r = int(spec["n_radial"])
    n_a = int(spec["n_angular"])
    r_max = float(spec["r_max"])
    theta_max = float(spec["theta_max"])
    r = np.linspace(0.0, r_max, n_r)
    theta = np.linspace(0.0, theta_max, n_a)
    R, T = np.meshgrid(r, theta, indexing="ij")
    return R, T, {"n_r": n_r, "n_a": n_a, "r_max": r_max, "theta_max": theta_max}


def _build_poisson_sector(case: dict[str, Any]) -> np.ndarray:
    """MMS Poisson on circular sector with homogeneous Dirichlet on all boundary.

    Domain: Ω = {(r, θ) : 0 ≤ r ≤ r_max, 0 ≤ θ ≤ θ_max}.
    Boundary: three pieces, all set to u=0:
      r = r_max,  θ = 0,  θ = θ_max.   (Apex r=0 lies on θ=0 line.)

    Manufactured u(r, θ) = r² (r_max − r) sin(π θ / θ_max).
      Vanishes at r=0 (via r² factor — also makes Laplacian bounded
      at the apex), at r=r_max (via r_max−r factor), at θ=0 and θ=θ_max
      (via the sine).

    Forcing  f(r, θ) = -Δu  (SymPy-derived, polar):
        f = (θ_max² (9r − 4 r_max) − π² (r − r_max)) sin(πθ/θ_max) / θ_max²
    Specialised at r_max=1, θ_max=3π/4:
        f = (5/9) (13r − 4) sin(4θ/3)
    """
    R, T, meta = _grid_polar_sector(case)
    r_max = meta["r_max"]
    theta_max = meta["theta_max"]
    u = (R ** 2) * (r_max - R) * np.sin(np.pi * T / theta_max)
    return u.reshape(-1).astype("float64")


def _build_helmholtz_sector(case: dict[str, Any]) -> np.ndarray:
    """Helmholtz_low_k on sector + B0.

    Same eigenmode-like u as Poisson_sector:
        u_exact(r, θ) = r²(r_max − r) sin(π θ / θ_max),
    SymPy-derived forcing (k=2 at r_max=1, θ_max=3π/4):
        f = sin(4θ/3) [4 r³ − 4 r² + 65/9 · r − 20/9]
    """
    R, T, meta = _grid_polar_sector(case)
    r_max = meta["r_max"]
    theta_max = meta["theta_max"]
    u = (R ** 2) * (r_max - R) * np.sin(np.pi * T / theta_max)
    return u.reshape(-1).astype("float64")


def _build_heat_sector(case: dict[str, Any]) -> np.ndarray:
    """Heat on sector + B0.

    u(r, θ, t) = e^(−t) r²(r_max − r) sin(π θ / θ_max), α=1.
    Reference at T_final (default 0.5):  e^(−T) · u_spatial.
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 0.5))
    R, Th, meta = _grid_polar_sector(case)
    r_max = meta["r_max"]
    theta_max = meta["theta_max"]
    u = np.exp(-T) * (R ** 2) * (r_max - R) * np.sin(np.pi * Th / theta_max)
    return u.reshape(-1).astype("float64")


def _build_wave_sector(case: dict[str, Any]) -> np.ndarray:
    """Wave on sector + B0.

    u(r, θ, t) = cos(t) r²(r_max − r) sin(π θ / θ_max), c=1.
    IC: u(·,·,0) = r²(r_max-r) sin(πθ/θ_max), ∂_t u(·,·,0) = 0.
    Reference at T_final (default 1.0): cos(T) · u_spatial.
    """
    params = case["case_spec"]["pde"]["parameters"]
    T = float(params.get("T_final", 1.0))
    R, Th, meta = _grid_polar_sector(case)
    r_max = meta["r_max"]
    theta_max = meta["theta_max"]
    u = np.cos(T) * (R ** 2) * (r_max - R) * np.sin(np.pi * Th / theta_max)
    return u.reshape(-1).astype("float64")


_SECTOR_BUILDERS = {
    "Poisson": _build_poisson_sector,
    "Helmholtz_low_k": _build_helmholtz_sector,
    "heat": _build_heat_sector,
    "wave": _build_wave_sector,
}


def manufactured_forcing_sympy() -> sp.Expr:
    """Forcing for the canonical Poisson case (kept for backward compat)."""
    x, y = sp.symbols("x y")
    u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    return sp.simplify(-(u.diff(x, 2) + u.diff(y, 2)))


def forcing_descriptor(
    *,
    operator_family: str,
    parameters: dict[str, Any],
    geometry_shape: str = "unit_square",
    bc_kind: str = "dirichlet_homogeneous",
) -> dict[str, Any] | None:
    """Return a ``case_spec.pde.forcing`` block for the given case configuration.

    Returns ``None`` for combinations not yet supported by manufactured.py.

    Keys (when non-None):
        ``expr_sympy``    — sympy-parseable string for the forcing function
        ``expr_human``    — plain-language rendering for the case statement
        ``u_exact_sympy`` — the manufactured exact solution
        ``is_vector``     — True iff ``expr_sympy`` represents a vector field
        ``coordinate_system`` — "cartesian" (default) or "polar"
    """
    if geometry_shape in {"L_shape_2d", "square_with_hole"}:
        if operator_family == "Poisson" and bc_kind == "dirichlet_homogeneous":
            return {
                "expr_sympy": "1",
                "expr_human": (
                    "f(x, y) = 1  (constant unit forcing); the reference is a "
                    "converged numerical solution — no closed-form u_exact."
                ),
                "u_exact_sympy": None,
                "is_vector": False,
                "reference_type": "high_fidelity",
            }
        return None
    if geometry_shape == "sector":
        if bc_kind != "dirichlet_homogeneous":
            return None
        r_max = float(parameters.get("r_max", 1.0))
        theta_max = float(parameters.get("theta_max", 3 * 3.141592653589793 / 4))
        if operator_family == "Poisson":
            return {
                "expr_sympy": (
                    "(theta_max**2*(9*r - 4*r_max) - pi**2*(r - r_max)) "
                    "* sin(pi*theta/theta_max) / theta_max**2"
                ),
                "expr_human": (
                    f"f(r,θ) = (θ_max²(9r − 4 r_max) − π²(r − r_max)) "
                    f"sin(πθ/θ_max) / θ_max²  with r_max={r_max}, θ_max≈{theta_max:.4f}."
                ),
                "u_exact_sympy": "r**2 * (r_max - r) * sin(pi*theta/theta_max)",
                "is_vector": False,
                "coordinate_system": "polar",
                "r_max": r_max,
                "theta_max": theta_max,
            }
        if operator_family == "Helmholtz_low_k":
            k = float(parameters.get("k", 2.0))
            return {
                "expr_sympy": (
                    "sin(pi*theta/theta_max) * "
                    "((theta_max**2*(9*r - 4*r_max) - pi**2*(r - r_max)) / theta_max**2 "
                    "- k**2 * r**2 * (r_max - r))"
                ),
                "expr_human": (
                    f"f(r,θ) = sin(πθ/θ_max) · "
                    f"[(θ_max²(9r − 4 r_max) − π²(r − r_max))/θ_max² − k²·r²(r_max−r)]  "
                    f"with k={k}, r_max={r_max}, θ_max≈{theta_max:.4f}."
                ),
                "u_exact_sympy": "r**2 * (r_max - r) * sin(pi*theta/theta_max)",
                "is_vector": False,
                "coordinate_system": "polar",
                "r_max": r_max,
                "theta_max": theta_max,
            }
        if operator_family == "heat":
            alpha = float(parameters.get("alpha", 1.0))
            T_final = float(parameters.get("T_final", 0.5))
            return {
                "expr_sympy": (
                    "exp(-t) * sin(pi*theta/theta_max) * "
                    "(-r**2*(r_max - r) "
                    "+ (theta_max**2*(9*r - 4*r_max) - pi**2*(r - r_max)) / theta_max**2)"
                ),
                "expr_human": (
                    f"f(r,θ,t) = e^(-t) · sin(πθ/θ_max) · "
                    f"[-r²(r_max-r) + (θ_max²(9r − 4r_max) − π²(r − r_max))/θ_max²]  "
                    f"with α={alpha}, r_max={r_max}, θ_max≈{theta_max:.4f}."
                ),
                "u_exact_sympy": "exp(-t) * r**2 * (r_max - r) * sin(pi*theta/theta_max)",
                "ic_sympy": "r**2 * (r_max - r) * sin(pi*theta/theta_max)",
                "ic_human": f"u(r,θ,0) = r²(r_max-r) sin(πθ/θ_max); T_final={T_final}.",
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_max": r_max,
                "theta_max": theta_max,
                "bc_human": (
                    "Dirichlet u = 0 on the three boundary pieces of the sector: "
                    "r = r_max,  θ = 0,  θ = θ_max."
                ),
            }
        if operator_family == "wave":
            c_speed = float(parameters.get("c", 1.0))
            T_final = float(parameters.get("T_final", 1.0))
            return {
                "expr_sympy": (
                    "cos(t) * sin(pi*theta/theta_max) * "
                    "(-r**2*(r_max - r) "
                    "+ (theta_max**2*(9*r - 4*r_max) - pi**2*(r - r_max)) / theta_max**2)"
                ),
                "expr_human": (
                    f"f(r,θ,t) = cos(t) · sin(πθ/θ_max) · "
                    f"[-r²(r_max-r) + (θ_max²(9r − 4r_max) − π²(r − r_max))/θ_max²]  "
                    f"with c={c_speed}, r_max={r_max}, θ_max≈{theta_max:.4f}."
                ),
                "u_exact_sympy": "cos(t) * r**2 * (r_max - r) * sin(pi*theta/theta_max)",
                "ic_sympy": "r**2 * (r_max - r) * sin(pi*theta/theta_max)",
                "ic_human": (
                    f"u(r,θ,0) = r²(r_max-r) sin(πθ/θ_max); ∂_t u(·,0) = 0; T_final={T_final}."
                ),
                "v0_sympy": "0",
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_max": r_max,
                "theta_max": theta_max,
                "bc_human": (
                    "Dirichlet u = 0 on the three boundary pieces of the sector: "
                    "r = r_max,  θ = 0,  θ = θ_max."
                ),
            }
        return None
    if geometry_shape == "annulus":
        if operator_family == "Helmholtz_low_k" and bc_kind == "dirichlet_homogeneous":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            k = float(parameters.get("k", 2.0))
            return {
                "expr_sympy": (
                    "(((4 - 4*pi**2) * r**2 * sin(2*pi*r) "
                    "+ 2*pi*r*cos(2*pi*r) - 4*sin(2*pi*r)) * cos(2*theta)) / r**2"
                ),
                "expr_human": (
                    f"f(r,θ) = cos(2θ) · [(k² − π²/L²)·sin(π(r-r_in)/L) "
                    f"+ (1/r²) (π r/L · cos(π(r-r_in)/L) − 4·sin(π(r-r_in)/L))]  "
                    f"with k={k}, r_in={r_in}, r_out={r_out}, L=r_out−r_in;  "
                    f"both circular boundaries Dirichlet 0."
                ),
                "u_exact_sympy": "sin(pi*(r - r_in)/(r_out - r_in)) * cos(2*theta)",
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": "Dirichlet u = 0 on r = r_in AND r = r_out.",
            }
        if operator_family == "heat" and bc_kind == "dirichlet_homogeneous":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            alpha = float(parameters.get("alpha", 1.0))
            T_final = float(parameters.get("T_final", 0.5))
            return {
                "expr_sympy": (
                    "exp(-t) * cos(2*theta) * ((1 - 4*pi**2) * r**2 * sin(2*pi*r) "
                    "+ 2*pi*r*cos(2*pi*r) - 4*sin(2*pi*r)) / r**2"
                ),
                "expr_human": (
                    f"f(r,θ,t) = e^(-t) · cos(2θ) · "
                    f"[(1 − π²/L²)·sin(π(r-r_in)/L) + (π r/L · cos(π(r-r_in)/L) "
                    f"− 4·sin(π(r-r_in)/L))/r²]  "
                    f"with α={alpha}, r_in={r_in}, r_out={r_out}, L=r_out−r_in;  "
                    f"both circular boundaries Dirichlet 0."
                ),
                "u_exact_sympy": (
                    "exp(-t) * sin(pi*(r - r_in)/(r_out - r_in)) * cos(2*theta)"
                ),
                "ic_sympy": "sin(pi*(r - r_in)/(r_out - r_in)) * cos(2*theta)",
                "ic_human": (
                    f"u(r,θ,0) = sin(π(r-r_in)/L) cos(2θ); T_final={T_final}."
                ),
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": "Dirichlet u = 0 on r = r_in AND r = r_out.",
            }
        if operator_family == "wave" and bc_kind == "dirichlet_homogeneous":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            c_speed = float(parameters.get("c", 1.0))
            T_final = float(parameters.get("T_final", 1.0))
            return {
                "expr_sympy": (
                    "cos(t) * cos(2*theta) * ((1 - 4*pi**2) * r**2 * sin(2*pi*r) "
                    "+ 2*pi*r*cos(2*pi*r) - 4*sin(2*pi*r)) / r**2"
                ),
                "expr_human": (
                    f"f(r,θ,t) = cos(t) · cos(2θ) · "
                    f"[(1 − π²/L²)·sin(π(r-r_in)/L) + (π r/L · cos(π(r-r_in)/L) "
                    f"− 4·sin(π(r-r_in)/L))/r²]  "
                    f"with c={c_speed}, r_in={r_in}, r_out={r_out}, L=r_out−r_in;  "
                    f"both circular boundaries Dirichlet 0."
                ),
                "u_exact_sympy": (
                    "cos(t) * sin(pi*(r - r_in)/(r_out - r_in)) * cos(2*theta)"
                ),
                "ic_sympy": "sin(pi*(r - r_in)/(r_out - r_in)) * cos(2*theta)",
                "ic_human": (
                    f"u(r,θ,0) = sin(π(r-r_in)/L) cos(2θ); ∂_t u(·,0)=0; T_final={T_final}."
                ),
                "v0_sympy": "0",
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": "Dirichlet u = 0 on r = r_in AND r = r_out.",
            }
        if operator_family == "Burgers" and bc_kind == "dirichlet_neumann_mixed":
            nu = float(parameters.get("nu", 0.1))
            T_final = float(parameters.get("T_final", 0.5))
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            alpha = 1.0
            return {
                "expr_sympy": (
                    "((alpha*r**2 - nu*pi**2*r**2/(4*L**2) + nu) * exp(alpha*t) "
                    "+ (annulus B1 convection term)) "
                    "* exp(-2*alpha*t) * cos(2*theta) / r**2  "
                    "[symbolic form; numerical form in manufactured.py]"
                ),
                "expr_human": (
                    f"f(r,θ,t) is the MMS forcing for forced 2D Burgers with Cartesian "
                    f"convection ∂_x u + ∂_y u expressed in polar coords via the chain "
                    f"rule:  ∂_x u + ∂_y u = (cos θ + sin θ)∂_r u + ((cos θ − sin θ)/r)∂_θ u. "
                    f"Setting A = π/(2L) and g(r) = sin(A·(r−r_in)), g'(r) = A·cos(A·(r−r_in)):  "
                    f"f = e^(−αt)·cos(2θ)·[−α·g(r) − ν·(g'(r)/r − A²·g(r) − 4·g(r)/r²)] "
                    f"+ e^(−2αt)·g(r)·cos(2θ)·"
                    f"[(cos θ + sin θ)·g'(r)·cos(2θ) − ((cos θ − sin θ)/r)·g(r)·2·sin(2θ)]. "
                    f"Parameters: α=1, ν={nu}, r_in={r_in}, r_out={r_out}, L=r_out−r_in, "
                    f"T_final={T_final}.  "
                    f"BC: u=0 on r=r_in, ∂u/∂r=0 on r=r_out;  IC in the IC block."
                ),
                "u_exact_sympy": (
                    f"exp(-{alpha}*t) * sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)"
                ),
                "ic_sympy": "sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)",
                "ic_human": (
                    f"u(r,θ,0) = sin(π(r-r_in)/(2L)) cos(2θ); T_final={T_final}."
                ),
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": (
                    "Dirichlet u = 0 on r = r_in; "
                    "homogeneous Neumann ∂u/∂r = 0 on r = r_out."
                ),
            }
        if operator_family == "Burgers" and bc_kind == "dirichlet_homogeneous":
            nu = float(parameters.get("nu", 0.1))
            T_final = float(parameters.get("T_final", 0.5))
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            alpha = 1.0
            return {
                "expr_sympy": (
                    "((alpha*r**2 - nu*pi**2*r**2/L**2 + nu) * exp(alpha*t) "
                    "+ sin_radial_part) * exp(-2*alpha*t) * cos(theta) / r**2  "
                    "[symbolic form; numerical form in manufactured.py]"
                ),
                "expr_human": (
                    f"f(r,θ,t) is the MMS forcing for forced 2D Burgers with Cartesian "
                    f"convection ∂_x u + ∂_y u expressed in polar coords via the chain "
                    f"rule:  ∂_x u + ∂_y u = (cos θ + sin θ)∂_r u + ((cos θ − sin θ)/r)∂_θ u. "
                    f"Setting A = π/L and g(r) = sin(A·(r−r_in)), g'(r) = A·cos(A·(r−r_in)):  "
                    f"f = e^(−αt)·cos(θ)·[−α·g(r) − ν·(g'(r)/r − A²·g(r) − g(r)/r²)] "
                    f"+ e^(−2αt)·g(r)·cos(θ)·"
                    f"[(cos θ + sin θ)·g'(r)·cos(θ) − ((cos θ − sin θ)/r)·g(r)·sin(θ)]. "
                    f"Parameters: α=1, ν={nu}, r_in={r_in}, r_out={r_out}, L=r_out−r_in, "
                    f"T_final={T_final}."
                ),
                "u_exact_sympy": (
                    f"exp(-{alpha}*t) * sin(pi*(r - r_in)/(r_out - r_in)) * cos(theta)"
                ),
                "ic_sympy": "sin(pi*(r - r_in)/(r_out - r_in)) * cos(theta)",
                "ic_human": (
                    f"u(r,θ,0) = sin(π(r-r_in)/L) cos(θ); T_final={T_final}."
                ),
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": (
                    "Dirichlet u = 0 on r = r_in AND r = r_out (full B0)."
                ),
            }
        if operator_family == "Poisson" and bc_kind == "dirichlet_homogeneous":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            return {
                "expr_sympy": (
                    "(pi*r*(-pi*r*sin(pi*(r - r_in)/(r_in - r_out)) "
                    "+ (r_in - r_out)*cos(pi*(r - r_in)/(r_in - r_out))) "
                    "- 4*(r_in - r_out)**2*sin(pi*(r - r_in)/(r_in - r_out)))"
                    "*cos(2*theta) / (r**2*(r_in - r_out)**2)"
                ),
                "expr_human": (
                    f"f(r,θ) = cos(2θ) · [(π²/L²)·sin(π(r-r_in)/L) "
                    f"+ (1/r²) · (4·sin(π(r-r_in)/L) − π r/L · cos(π(r-r_in)/L))]  "
                    f"with r_in={r_in}, r_out={r_out}, L=r_out−r_in;  "
                    f"both circular boundaries Dirichlet 0."
                ),
                "u_exact_sympy": "sin(pi*(r - r_in)/(r_out - r_in)) * cos(2*theta)",
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
            }
        if operator_family == "Helmholtz_low_k" and bc_kind == "dirichlet_neumann_mixed":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            k = float(parameters.get("k", 2.0))
            return {
                "expr_sympy": (
                    f"cos(2*theta) * (({k}**2 - pi**2) * r**2 * cos(pi*r) "
                    "- pi*r*sin(pi*r) - 4*cos(pi*r)) / r**2"
                ),
                "expr_human": (
                    f"f(r,θ) = cos(2θ)[(k²−π²)r²cos(πr) − πr sin(πr) − 4cos(πr)]/r²  "
                    f"with k={k}, r_in={r_in}, r_out={r_out};  "
                    f"BC: u=0 on r=r_in, ∂u/∂r=0 on r=r_out."
                ),
                "u_exact_sympy": "sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)",
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": (
                    "Dirichlet u = 0 on inner circle r = r_in; "
                    "homogeneous Neumann ∂u/∂r = 0 on outer circle r = r_out."
                ),
            }
        if operator_family == "heat" and bc_kind == "dirichlet_neumann_mixed":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            alpha = float(parameters.get("alpha", 1.0))
            T_final = float(parameters.get("T_final", 0.5))
            return {
                "expr_sympy": (
                    "exp(-t) * cos(2*theta) * ((1 - pi**2) * r**2 * cos(pi*r) "
                    "- pi*r*sin(pi*r) - 4*cos(pi*r)) / r**2"
                ),
                "expr_human": (
                    f"f(r,θ,t) = e^(-t) cos(2θ)[(1−π²)r²cos(πr) − πr sin(πr) − 4cos(πr)]/r²  "
                    f"with α={alpha}, r_in={r_in}, r_out={r_out};  "
                    f"BC: u=0 on r=r_in, ∂u/∂r=0 on r=r_out."
                ),
                "u_exact_sympy": (
                    "exp(-t) * sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)"
                ),
                "ic_sympy": "sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)",
                "ic_human": (
                    f"u(r,θ,0) = sin(π(r−r_in)/(2L)) cos(2θ); T_final={T_final}."
                ),
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": (
                    "Dirichlet u = 0 on inner circle r = r_in; "
                    "homogeneous Neumann ∂u/∂r = 0 on outer circle r = r_out."
                ),
            }
        if operator_family == "wave" and bc_kind == "dirichlet_neumann_mixed":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            c_speed = float(parameters.get("c", 1.0))
            T_final = float(parameters.get("T_final", 1.0))
            return {
                "expr_sympy": (
                    "cos(t) * cos(2*theta) * ((1 - pi**2) * r**2 * cos(pi*r) "
                    "- pi*r*sin(pi*r) - 4*cos(pi*r)) / r**2"
                ),
                "expr_human": (
                    f"f(r,θ,t) = cos(t) cos(2θ)[(1−π²)r²cos(πr) − πr sin(πr) − 4cos(πr)]/r²  "
                    f"with c={c_speed}, r_in={r_in}, r_out={r_out};  "
                    f"BC: u=0 on r=r_in, ∂u/∂r=0 on r=r_out."
                ),
                "u_exact_sympy": (
                    "cos(t) * sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)"
                ),
                "ic_sympy": "sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)",
                "ic_human": (
                    f"u(r,θ,0) = sin(π(r−r_in)/(2L)) cos(2θ); ∂_t u(·,0) = 0; T_final={T_final}."
                ),
                "v0_sympy": "0",
                "T_final": T_final,
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": (
                    "Dirichlet u = 0 on inner circle r = r_in; "
                    "homogeneous Neumann ∂u/∂r = 0 on outer circle r = r_out."
                ),
            }
        if operator_family == "Poisson" and bc_kind == "dirichlet_neumann_mixed":
            r_in = float(parameters.get("r_in", 0.5))
            r_out = float(parameters.get("r_out", 1.0))
            return {
                "expr_sympy": (
                    "-(pi**2 * r**2 * cos(pi*r) + pi*r*sin(pi*r) + 4*cos(pi*r)) "
                    "* cos(2*theta) / r**2"
                ),
                "expr_human": (
                    f"f(r,θ) = -(π² r² cos(πr) + π r sin(πr) + 4 cos(πr)) · cos(2θ) / r²  "
                    f"with r_in={r_in}, r_out={r_out};  "
                    f"BC: u=0 on r=r_in, ∂u/∂r=0 on r=r_out  "
                    f"(formula specialised to r_in=0.5, r_out=1.0)."
                ),
                "u_exact_sympy": "sin(pi*(r - r_in)/(2*(r_out - r_in))) * cos(2*theta)",
                "is_vector": False,
                "coordinate_system": "polar",
                "r_in": r_in,
                "r_out": r_out,
                "bc_human": (
                    "Dirichlet u = 0 on inner circle r = r_in; "
                    "homogeneous Neumann ∂u/∂r = 0 on outer circle r = r_out."
                ),
            }
        return None

    if bc_kind == "dirichlet_neumann_mixed":
        BC_HUMAN = (
            "Dirichlet u = 0 on {x=0, y=0, y=1}; "
            "Neumann ∂u/∂x = 0 on {x=1} (homogeneous)."
        )
        if operator_family == "Poisson":
            return {
                "expr_sympy": (
                    "sin(pi*y) * ((5*pi**2/4) * x*(2-x) * sin(pi*x/2) "
                    "- 2*pi*(1-x) * cos(pi*x/2) + 2*sin(pi*x/2))"
                ),
                "expr_human": (
                    "f(x,y) = sin(πy) · [(5π²/4)·x(2−x)·sin(πx/2) "
                    "− 2π(1−x)·cos(πx/2) + 2·sin(πx/2)]"
                ),
                "is_vector": False,
                "bc_human": BC_HUMAN,
            }
        if operator_family == "Helmholtz_low_k":
            k = float(parameters.get("k", 2.0))
            return {
                "expr_sympy": (
                    f"sin(pi*y) * (((5*pi**2/4) - ({k})**2)*x*(2-x)*sin(pi*x/2) "
                    f"- 2*pi*(1-x)*cos(pi*x/2) + 2*sin(pi*x/2))"
                ),
                "expr_human": (
                    f"f(x,y) = sin(πy) · [(5π²/4 − k²)·x(2−x)·sin(πx/2) "
                    f"− 2π(1−x)·cos(πx/2) + 2·sin(πx/2)]  with k={k}"
                ),
                "is_vector": False,
                "bc_human": BC_HUMAN,
            }
        if operator_family == "heat":
            alpha = float(parameters.get("alpha", 1.0))
            T_final = float(parameters.get("T_final", 0.5))
            return {
                "expr_sympy": (
                    f"2*t*sin(pi*x/2)*sin(pi*y)*x*(2-x) "
                    f"+ {alpha}*(1 + t**2)*sin(pi*y)*("
                    f"(5*pi**2/4)*x*(2-x)*sin(pi*x/2) "
                    f"- 2*pi*(1-x)*cos(pi*x/2) + 2*sin(pi*x/2))"
                ),
                "expr_human": (
                    f"f(x,y,t) = 2t·sin(πx/2)·sin(πy)·x(2−x) "
                    f"+ α(1+t²)·sin(πy)·[(5π²/4)·x(2−x)·sin(πx/2) "
                    f"− 2π(1−x)·cos(πx/2) + 2·sin(πx/2)]  with α={alpha}"
                ),
                "ic_sympy": "sin(pi*x/2) * sin(pi*y) * x*(2-x)",
                "ic_human": "u(x,y,0) = sin(πx/2)·sin(πy)·x(2−x)",
                "T_final": T_final,
                "is_vector": False,
                "bc_human": BC_HUMAN,
            }
        if operator_family == "wave":
            c = float(parameters.get("c", 1.0))
            T_final = float(parameters.get("T_final", 1.0))
            return {
                "expr_sympy": (
                    f"sin(pi*x/2)*sin(pi*y)*x*(2-x) "
                    f"+ ({c})**2*(1 + t**2/2)*sin(pi*y)*("
                    f"(5*pi**2/4)*x*(2-x)*sin(pi*x/2) "
                    f"- 2*pi*(1-x)*cos(pi*x/2) + 2*sin(pi*x/2))"
                ),
                "expr_human": (
                    f"f(x,y,t) = sin(πx/2)·sin(πy)·x(2−x) "
                    f"+ c²(1+t²/2)·sin(πy)·[(5π²/4)·x(2−x)·sin(πx/2) "
                    f"− 2π(1−x)·cos(πx/2) + 2·sin(πx/2)]  with c={c}"
                ),
                "ic_sympy": "sin(pi*x/2) * sin(pi*y) * x*(2-x)",
                "ic_human": "u(x,y,0) = sin(πx/2)·sin(πy)·x(2−x); ∂_t u(x,y,0) = 0",
                "v0_sympy": "0",
                "T_final": T_final,
                "is_vector": False,
                "bc_human": BC_HUMAN,
            }
        if operator_family == "Burgers":
            nu = float(parameters.get("nu", 0.1))
            T_final = float(parameters.get("T_final", 0.5))
            return {
                "expr_sympy": (
                    f"f is the MMS forcing for forced 2D Burgers with Cartesian "
                    f"convection u(∂_x u + ∂_y u), u = e^(-αt)·sin(πx/2)·sin(πy)·x(2-x), "
                    f"α=1, ν={nu}; closed form is a sum of e^(-αt) and e^(-2αt) terms "
                    f"in sin(πx/2)sin(πy), cos(πx/2)sin(πy), sin(πx/2)cos(πy) — "
                    f"see solvers/burgers_unit_square_b1_imex.py for the numerical form."
                ),
                "expr_human": (
                    f"f(x,y,t) is the Cartesian-Burgers MMS forcing for "
                    f"u(x,y,t) = e^(-αt)·sin(πx/2)·sin(πy)·x(2−x), α=1, ν={nu}. "
                    f"Has a linear e^(-αt) part (∂_t u − νΔu) and a quadratic e^(-2αt) "
                    f"convection part. T_final={T_final}."
                ),
                "ic_sympy": "sin(pi*x/2) * sin(pi*y) * x*(2-x)",
                "ic_human": f"u(x,y,0) = sin(πx/2)·sin(πy)·x(2−x); T_final={T_final}.",
                "T_final": T_final,
                "is_vector": False,
                "bc_human": BC_HUMAN,
            }
        return None

    if operator_family == "Poisson":
        return {
            "expr_sympy": (
                "exp(x + y - 1) * (2*(pi**2 - 1) * sin(pi*x) * sin(pi*y) "
                "- 2*pi * sin(pi*(x + y)))"
            ),
            "expr_human": (
                "f(x,y) = exp(x+y−1) · [2(π² − 1)·sin(πx)sin(πy) − 2π·sin(π(x+y))]"
            ),
            "is_vector": False,
        }
    if operator_family == "Helmholtz_low_k":
        k = float(parameters.get("k", 2.0))
        return {
            "expr_sympy": (
                f"exp(x + y - 1) * ((2*(pi**2 - 1) - ({k})**2) * sin(pi*x) * sin(pi*y) "
                f"- 2*pi * sin(pi*(x + y)))"
            ),
            "expr_human": (
                f"f(x,y) = exp(x+y−1) · [(2(π²−1) − k²)·sin(πx)sin(πy) − 2π·sin(π(x+y))]  "
                f"with k={k}"
            ),
            "is_vector": False,
        }
    if operator_family == "heat":
        alpha = float(parameters.get("alpha", 1.0))
        T_final = float(parameters.get("T_final", 0.5))
        return {
            "expr_sympy": (
                f"exp(x + y - 1) * ("
                f"2*{alpha}*(1 + t**2)*(pi**2 - 1)*sin(pi*x)*sin(pi*y) "
                f"- 2*pi*{alpha}*(1 + t**2)*sin(pi*(x + y)) "
                f"+ 2*t*sin(pi*x)*sin(pi*y))"
            ),
            "expr_human": (
                f"f(x,y,t) = exp(x+y−1) · "
                f"[2α(1+t²)(π²−1)·sin(πx)sin(πy) − 2πα(1+t²)·sin(π(x+y)) "
                f"+ 2t·sin(πx)sin(πy)]  with α={alpha}"
            ),
            "ic_sympy": "sin(pi*x) * sin(pi*y) * exp(x + y - 1)",
            "ic_human": "u(x,y,0) = sin(πx)·sin(πy)·exp(x+y−1)",
            "T_final": T_final,
            "is_vector": False,
        }
    if operator_family == "wave":
        c = float(parameters.get("c", 1.0))
        T_final = float(parameters.get("T_final", 1.0))
        return {
            "expr_sympy": (
                f"exp(x + y - 1) * ("
                f"sin(pi*x)*sin(pi*y) "
                f"+ {c}**2*(2 + t**2)*(pi**2 - 1)*sin(pi*x)*sin(pi*y) "
                f"- pi*{c}**2*(2 + t**2)*sin(pi*(x + y)))"
            ),
            "expr_human": (
                f"f(x,y,t) = exp(x+y−1) · ["
                f"sin(πx)sin(πy) + c²(2+t²)(π²−1)·sin(πx)sin(πy) "
                f"− πc²(2+t²)·sin(π(x+y))]  with c={c}"
            ),
            "ic_sympy": "sin(pi*x) * sin(pi*y) * exp(x + y - 1)",
            "ic_human": "u(x,y,0) = sin(πx)·sin(πy)·exp(x+y−1); ∂_t u(x,y,0) = 0",
            "v0_sympy": "0",
            "T_final": T_final,
            "is_vector": False,
        }
    if operator_family == "Burgers":
        nu = float(parameters.get("nu", 0.1))
        T_final = float(parameters.get("T_final", 0.5))
        alpha = 1.0
        return {
            "expr_sympy": (
                f"exp(-2*{alpha}*t) * sin(pi*x) * sin(pi*y) * "
                f"((-{alpha} + 2*pi**2*{nu}) * exp({alpha}*t) "
                f"+ pi*sin(pi*(x + y)))"
            ),
            "expr_human": (
                f"f(x,y,t) = e^(-2αt) sin(πx) sin(πy) "
                f"[(-α + 2π²ν) e^(αt) + π sin(π(x+y))]  with α=1, ν={nu}."
            ),
            "u_exact_sympy": f"exp(-{alpha}*t) * sin(pi*x) * sin(pi*y)",
            "ic_sympy": "sin(pi*x) * sin(pi*y)",
            "ic_human": f"u(x,y,0) = sin(πx) sin(πy);  evaluate at T_final={T_final}.",
            "T_final": T_final,
            "is_vector": False,
        }
    if operator_family == "linear_elasticity":
        lam = float(parameters.get("lambda", 1.0))
        mu = float(parameters.get("mu", 1.0))
        return {
            "expr_sympy": (
                f"[ exp(x+y-1) * ("
                f"({lam} + 3*{mu})*(pi**2 - 1)*sin(pi*x)*sin(pi*y) "
                f"- 2*pi*({lam} + 2*{mu})*cos(pi*x)*sin(pi*y) "
                f"- 2*pi*{mu}*sin(pi*x)*cos(pi*y)), "
                f"-({lam} + {mu}) * exp(x+y-1) * ("
                f"sin(pi*x)*sin(pi*y) + pi*sin(pi*(x+y)) "
                f"+ pi**2*cos(pi*x)*cos(pi*y)) ]"
            ),
            "expr_human": (
                f"f = ( E·[(λ+3μ)(π²−1)·sin(πx)sin(πy) − 2π(λ+2μ)·cos(πx)sin(πy) "
                f"− 2πμ·sin(πx)cos(πy)],  "
                f"−(λ+μ)·E·[sin(πx)sin(πy) + π·sin(π(x+y)) + π²·cos(πx)cos(πy)] ),  "
                f"with E = exp(x+y−1), λ={lam}, μ={mu}"
            ),
            "is_vector": True,
        }
    return None


__all__ = ["build_reference", "manufactured_forcing_sympy", "forcing_descriptor"]
