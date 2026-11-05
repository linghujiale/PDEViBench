"""Stage 3 — Runtime gate."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from evaluator.thresholds import RUNTIME_REPS, compute_tau_time


@dataclasses.dataclass(frozen=True)
class RuntimeVerdict:
    pass_: bool
    wall_seconds_reps: list[float]
    wall_seconds_mean: float
    tau_time_used: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_,
            "wall_seconds_reps": self.wall_seconds_reps,
            "wall_seconds_mean": self.wall_seconds_mean,
            "tau_time_used": self.tau_time_used,
        }


def check(*, wall_seconds_reps: list[float], t_base: float) -> RuntimeVerdict:
    """Apply Stage 3."""
    if len(wall_seconds_reps) != RUNTIME_REPS:
        raise ValueError(
            f"expected {RUNTIME_REPS} reps, got {len(wall_seconds_reps)}; "
            "see evaluator.thresholds.RUNTIME_REPS"
        )
    mean = float(np.mean(wall_seconds_reps))
    tau = compute_tau_time(t_base)
    return RuntimeVerdict(
        pass_=mean <= tau,
        wall_seconds_reps=list(wall_seconds_reps),
        wall_seconds_mean=mean,
        tau_time_used=tau,
    )


__all__ = ["RuntimeVerdict", "check"]
