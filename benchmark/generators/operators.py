"""Axis-O — Operator standardness."""

from __future__ import annotations

import dataclasses
from typing import Any

OPERATOR_FAMILIES: dict[int, list[str]] = {
    0: [
        "Poisson",
        "Helmholtz_low_k",
        "heat",
        "wave",
        "linear_elasticity",
    ],
    1: [
        "Burgers",
        "semilinear_reaction_diffusion",
        "p_Laplacian",
    ],
    2: [
        "Stokes",
        "Navier_Stokes_steady",
        "thermo_elasticity",
    ],
    3: [
        "fractional_Laplacian",
        "integro_differential",
        "peridynamic",
        "Cahn_Hilliard_degenerate",
    ],
}


@dataclasses.dataclass(frozen=True)
class OperatorSpec:
    """Output of an O-axis generator. Consumed by ``compose.py``."""

    operator_family: str
    statement: str
    parameters: dict[str, Any]
    structure_compatible: tuple[int, ...]
    discretization_min: int


def generate(*, level: int, family: str | None = None, seed: int = 0) -> OperatorSpec:
    """Generate a single operator spec at axis-O ``level``.

    Parameters
    ----------
    level
        Axis-O level (0..3).
    family
        Optional override of the operator family. If None, draws one
        deterministically from ``OPERATOR_FAMILIES[level]`` using ``seed``.
    seed
        Reproducibility seed.
    """
    if level not in OPERATOR_FAMILIES:
        raise ValueError(f"unknown O-level: {level}")
    families = OPERATOR_FAMILIES[level]
    chosen = family if family is not None else families[seed % len(families)]
    if chosen not in families:
        raise ValueError(f"family {chosen!r} not at O-level {level}")

    return _builders[chosen](seed=seed)


def _build_poisson(*, seed: int) -> OperatorSpec:
    return OperatorSpec(
        operator_family="Poisson",
        statement=(
            "Solve -Δu = f on the specified domain, with the boundary conditions "
            "given below. Here Δ is the standard Laplacian and f is the forcing term."
        ),
        parameters={"diffusivity": 1.0},
        structure_compatible=(0,),
        discretization_min=0,
    )


def _build_helmholtz_low_k(*, seed: int) -> OperatorSpec:
    return OperatorSpec(
        operator_family="Helmholtz_low_k",
        statement=(
            "Solve -Δu - k² u = f on the specified domain, where k is a real "
            "wave-number such that k h << 1 on the chosen mesh (k held in the "
            "low-frequency regime)."
        ),
        parameters={"k": 2.0},
        structure_compatible=(0,),
        discretization_min=0,
    )


def _build_heat(*, seed: int) -> OperatorSpec:
    return OperatorSpec(
        operator_family="heat",
        statement=(
            "Solve ∂_t u - α Δu = f on (0,T] with initial condition u(·, 0) = u₀ "
            "and the specified boundary conditions."
        ),
        parameters={"alpha": 1.0, "T_final": 0.5},
        structure_compatible=(0, 1),
        discretization_min=0,
    )


def _build_wave(*, seed: int) -> OperatorSpec:
    return OperatorSpec(
        operator_family="wave",
        statement=(
            "Solve ∂_tt u - c² Δu = f on (0,T] with initial conditions "
            "u(·, 0) = u₀, ∂_t u(·, 0) = v₀, and the specified boundary conditions."
        ),
        parameters={"c": 1.0, "T_final": 1.0},
        structure_compatible=(0, 1),
        discretization_min=0,
    )


def _build_linear_elasticity(*, seed: int) -> OperatorSpec:
    return OperatorSpec(
        operator_family="linear_elasticity",
        statement=(
            "Solve −∇·σ(u) = f with σ(u) = λ tr(ε(u)) I + 2μ ε(u), "
            "ε(u) = ½(∇u + ∇uᵀ), under the specified boundary conditions. "
            "Lamé parameters λ and μ are given below."
        ),
        parameters={"lambda": 1.0, "mu": 1.0},
        structure_compatible=(0,),
        discretization_min=0,
    )


def _build_burgers(*, seed: int) -> OperatorSpec:
    """Scalar viscous Burgers equation.

    PDE form (forced, 2D scalar):
        ∂_t u + u (∂_x u + ∂_y u) − ν Δu = f
    Parameters chosen so the manufactured solution is well-posed and the
    nonlinear term has the same order of magnitude as the diffusion at
    the chosen ν.
    """
    return OperatorSpec(
        operator_family="Burgers",
        statement=(
            "Solve the forced 2D scalar viscous Burgers equation\n"
            "    ∂_t u + u (∂_x u + ∂_y u) − ν Δu = f\n"
            "on (0, T_final] with the specified initial and boundary "
            "conditions. ν is the viscosity; the forcing f is given below."
        ),
        parameters={"nu": 0.1, "T_final": 0.5},
        structure_compatible=(0, 1),
        discretization_min=0,
    )


_builders = {
    "Poisson": _build_poisson,
    "Helmholtz_low_k": _build_helmholtz_low_k,
    "heat": _build_heat,
    "wave": _build_wave,
    "linear_elasticity": _build_linear_elasticity,
    "Burgers": _build_burgers,
}


__all__ = ["OperatorSpec", "OPERATOR_FAMILIES", "generate"]
