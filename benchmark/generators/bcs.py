"""Axis-B — Boundary-condition complexity."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

BC_FAMILIES: dict[int, list[str]] = {
    0: ["dirichlet_homogeneous"],
    1: ["dirichlet_neumann_mixed", "piecewise_constant"],
    2: ["robin_spatially_varying", "sinusoidal_dirichlet"],
    3: ["sommerfeld_radiation", "dtn_map", "periodic_with_flux"],
    4: ["signorini_contact", "slip_friction", "unilateral", "interface_jump"],
}


@dataclasses.dataclass(frozen=True)
class BCSpec:
    kind: str
    detail: dict[str, Any]


def generate(
    *,
    level: int,
    family: str | None = None,
    seed: int = 0,
    geometry_name: str = "unit_square",
) -> BCSpec:
    if level not in BC_FAMILIES:
        raise ValueError(f"unknown B-level: {level}")
    fam = BC_FAMILIES[level]
    chosen = family if family is not None else fam[seed % len(fam)]
    if chosen not in fam:
        raise ValueError(f"family {chosen!r} not at B-level {level}")
    builder = _builders.get(chosen)
    if builder is None:
        raise NotImplementedError(
            f"BC {chosen!r} (B{level}) is not implemented."
        )
    return builder(geometry_name=geometry_name)


def _dirichlet_homogeneous(*, geometry_name: str) -> BCSpec:
    return BCSpec(kind="dirichlet_homogeneous", detail={"value": 0.0})


def _dirichlet_neumann_mixed(*, geometry_name: str) -> BCSpec:
    """Mixed Dirichlet on Γ_D, homogeneous Neumann on Γ_N.

    Per-geometry mappings:

    * **unit_square**: Γ_D = {x=0, y=0, y=1}, Γ_N = {x=1}.
    * **annulus**:     Γ_D = {r = r_in},      Γ_N = {r = r_out}.
    """
    if geometry_name == "unit_square":
        return BCSpec(
            kind="dirichlet_neumann_mixed",
            detail={
                "dirichlet_faces": ["x=0", "y=0", "y=1"],
                "neumann_faces": ["x=1"],
                "dirichlet_value": 0.0,
                "neumann_value": 0.0,
            },
        )
    if geometry_name == "annulus":
        return BCSpec(
            kind="dirichlet_neumann_mixed",
            detail={
                "dirichlet_faces": ["r=r_in"],
                "neumann_faces": ["r=r_out"],
                "dirichlet_value": 0.0,
                "neumann_value": 0.0,
            },
        )
    raise NotImplementedError(
        f"B1 dirichlet_neumann_mixed not yet implemented for geometry={geometry_name!r}; "
        f"supported: unit_square, annulus"
    )


_builders: dict[str, Callable[..., BCSpec]] = {
    "dirichlet_homogeneous": _dirichlet_homogeneous,
    "dirichlet_neumann_mixed": _dirichlet_neumann_mixed,
}


__all__ = ["BCSpec", "BC_FAMILIES", "generate"]
