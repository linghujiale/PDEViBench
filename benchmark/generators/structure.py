"""Axis-S — Structure-preservation requirement."""

from __future__ import annotations

import dataclasses

S_LEVELS: dict[int, str] = {
    0: "None",
    1: "Soft (energy decay, monotonicity, non-negativity preferred)",
    2: "Hard (exact conservation, symplectic, div-free, monotone)",
}


@dataclasses.dataclass(frozen=True)
class StructureRequirement:
    level: int
    label: str
    constraints: tuple[str, ...]


def generate(*, level: int, seed: int = 0) -> StructureRequirement:
    if level not in S_LEVELS:
        raise ValueError(f"unknown S-level: {level}")
    return StructureRequirement(
        level=level,
        label=S_LEVELS[level],
        constraints={
            0: (),
            1: ("approximate_energy_decay",),
            2: ("exact_mass_conservation", "div_free"),
        }[level],
    )


__all__ = ["StructureRequirement", "S_LEVELS", "generate"]
