"""Stage 2 — Accuracy gate."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from evaluator.thresholds import compute_tau_acc


@dataclasses.dataclass(frozen=True)
class AccuracyVerdict:
    pass_: bool
    rel_l2: float
    abs_l2: float
    linf: float
    energy_norm: float | None
    tau_acc_used: float

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pass": self.pass_,
            "rel_l2": self.rel_l2,
            "abs_l2": self.abs_l2,
            "linf": self.linf,
            "tau_acc_used": self.tau_acc_used,
        }
        if self.energy_norm is not None:
            d["energy_norm"] = self.energy_norm
        return d


def check(
    *,
    submitted: np.ndarray,
    reference: np.ndarray,
    e_base: float,
    energy_matrix: np.ndarray | None = None,
) -> AccuracyVerdict:
    """Compute the four error norms and compare to ``tau_acc``.

    Parameters
    ----------
    submitted
        Agent-produced solution sampled on the case's evaluation grid.
    reference
        Reference solution on the same grid (MMS or high-fidelity).
    e_base
        Baseline error magnitude for this case.
    energy_matrix
        Optional symmetric positive-definite operator A such that
        ``energy_norm = sqrt((u - u_ref)^T A (u - u_ref))``. Required for cases
        where the case spec declares an energy-norm metric.
    """
    if submitted.shape != reference.shape:
        raise ValueError(f"shape mismatch: {submitted.shape} vs {reference.shape}")

    sub_nan = np.isnan(submitted)
    ref_nan = np.isnan(reference)
    if not np.array_equal(sub_nan, ref_nan):
        raise ValueError(
            "NaN mask of submitted does not match reference; the agent's "
            "solver is not respecting the case's domain mask."
        )
    valid = ~ref_nan
    diff = submitted[valid] - reference[valid]
    abs_l2 = float(np.linalg.norm(diff))
    ref_norm = float(np.linalg.norm(reference[valid]))
    rel_l2 = abs_l2 / max(ref_norm, 1e-30)
    linf = float(np.max(np.abs(diff)))

    energy_norm: float | None = None
    if energy_matrix is not None:
        if energy_matrix.shape != (submitted.size, submitted.size):
            raise ValueError(
                f"energy_matrix shape {energy_matrix.shape} incompatible with solution {submitted.shape}"
            )
        v = diff.reshape(-1)
        energy_norm = float(np.sqrt(v @ energy_matrix @ v))

    tau = compute_tau_acc(e_base)
    return AccuracyVerdict(
        pass_=rel_l2 <= tau,
        rel_l2=rel_l2,
        abs_l2=abs_l2,
        linf=linf,
        energy_norm=energy_norm,
        tau_acc_used=tau,
    )


__all__ = ["AccuracyVerdict", "check"]
