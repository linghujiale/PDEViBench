"""Axis-C — Coupling / multi-physics."""

from __future__ import annotations

import dataclasses

C_LEVELS: dict[int, str] = {
    0: "Single scalar PDE",
    1: "Vector-valued single PDE",
    2: "Weakly coupled",
    3: "Strongly coupled",
}


@dataclasses.dataclass(frozen=True)
class CouplingSpec:
    level: int
    label: str
    multi_field: bool


def generate(*, level: int, seed: int = 0) -> CouplingSpec:
    if level not in C_LEVELS:
        raise ValueError(f"unknown C-level: {level}")
    return CouplingSpec(level=level, label=C_LEVELS[level], multi_field=level >= 2)


__all__ = ["CouplingSpec", "C_LEVELS", "generate"]
