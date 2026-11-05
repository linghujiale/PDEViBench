"""Axis-D — Discretization specificity."""

from __future__ import annotations

import dataclasses
from typing import Any

D_LEVELS: dict[int, str] = {
    0: "P1/P2 conforming sufficient",
    1: "Higher-order or H(div)/H(curl) needed",
    2: "Mixed/saddle-point pair required",
    3: "Non-conforming / stabilized / DG required",
}


@dataclasses.dataclass(frozen=True)
class DiscretizationHint:
    level: int
    label: str
    suggested_spaces: tuple[str, ...]


def generate(*, level: int, seed: int = 0) -> DiscretizationHint:
    if level not in D_LEVELS:
        raise ValueError(f"unknown D-level: {level}")
    return DiscretizationHint(
        level=level,
        label=D_LEVELS[level],
        suggested_spaces={
            0: ("P1", "P2"),
            1: ("Nedelec", "Raviart-Thomas"),
            2: ("Taylor-Hood", "MINI"),
            3: ("C0IP", "SUPG", "DG"),
        }[level],
    )


__all__ = ["DiscretizationHint", "D_LEVELS", "generate"]
