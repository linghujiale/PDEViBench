"""Evaluation thresholds."""

from __future__ import annotations

from typing import Final

ALPHA_ACC: Final[float] = 10.0
ALPHA_TIME: Final[float] = 3.0
TAU_MIN: Final[float] = 1e-6

RUNTIME_REPS: Final[int] = 3

LLM_JUDGE_SAMPLE_RATE: Final[float] = 0.10


def compute_tau_acc(e_base: float) -> float:
    """Stage-2 accuracy threshold for a case with baseline error ``e_base``."""
    if e_base <= 0:
        raise ValueError(f"e_base must be > 0, got {e_base}")
    return max(ALPHA_ACC * e_base, TAU_MIN)


def compute_tau_time(t_base: float) -> float:
    """Stage-3 runtime threshold for a case with baseline wall-time ``t_base`` (seconds)."""
    if t_base <= 0:
        raise ValueError(f"t_base must be > 0, got {t_base}")
    return ALPHA_TIME * t_base


__all__ = [
    "ALPHA_ACC",
    "ALPHA_TIME",
    "TAU_MIN",
    "RUNTIME_REPS",
    "LLM_JUDGE_SAMPLE_RATE",
    "compute_tau_acc",
    "compute_tau_time",
]
