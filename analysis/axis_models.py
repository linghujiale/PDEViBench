"""Shared deterministic regression helpers for balanced-axis sensitivities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.optimize as opt
from scipy.stats import norm


AXES = ("O", "G", "B", "D", "S", "C")
MECHANISM_INTERACTIONS = (
    ("O", "G"),
    ("O", "B"),
    ("G", "B"),
    ("O", "C"),
    ("G", "C"),
    ("B", "C"),
)


def term_names(
    *,
    axes: Sequence[str] = AXES,
    interactions: Sequence[tuple[str, str]] = (),
) -> list[str]:
    return ["intercept", *axes, *(f"{left}:{right}" for left, right in interactions)]


def design_matrix(
    cases: Sequence[dict[str, Any]],
    *,
    axes: Sequence[str] = AXES,
    interactions: Sequence[tuple[str, str]] = (),
) -> np.ndarray:
    """Effect-coded design: binary zero/one axes map to -0.5/+0.5."""
    rows: list[list[float]] = []
    for case in cases:
        values = {
            axis: float(int(case.get("axes", {}).get(axis, 0))) - 0.5
            for axis in axes
        }
        row = [1.0, *(values[axis] for axis in axes)]
        row.extend(values[left] * values[right] for left, right in interactions)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _cluster_covariance(
    *,
    bread: np.ndarray,
    scores: np.ndarray,
    clusters: Sequence[str],
) -> np.ndarray:
    unique = list(dict.fromkeys(str(cluster) for cluster in clusters))
    grouped = np.zeros((len(unique), scores.shape[1]), dtype=float)
    index = {cluster: idx for idx, cluster in enumerate(unique)}
    for row, cluster in zip(scores, clusters):
        grouped[index[str(cluster)]] += row
    meat = grouped.T @ grouped
    covariance = bread @ meat @ bread
    n = scores.shape[0]
    p = scores.shape[1]
    g = len(unique)
    if g > 1 and n > p:
        covariance *= (g / (g - 1)) * ((n - 1) / (n - p))
    return covariance


def _coefficient_rows(
    beta: np.ndarray,
    covariance: np.ndarray,
    names: Sequence[str],
    *,
    logistic: bool,
) -> dict[str, dict[str, Any]]:
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    rows: dict[str, dict[str, Any]] = {}
    for name, coefficient, standard_error in zip(names, beta, standard_errors):
        coefficient_f = float(coefficient)
        se_f = float(standard_error)
        z = coefficient_f / se_f if se_f > 0 else float("nan")
        p_value = float(2.0 * norm.sf(abs(z))) if math.isfinite(z) else None
        row: dict[str, Any] = {
            "coef": coefficient_f,
            "cluster_se": se_f,
            "ci95": [coefficient_f - 1.96 * se_f, coefficient_f + 1.96 * se_f],
            "p_value": p_value,
        }
        if logistic:
            row["odds_ratio"] = float(np.exp(np.clip(coefficient_f, -700, 700)))
            row["odds_ratio_ci95"] = [
                float(np.exp(np.clip(coefficient_f - 1.96 * se_f, -700, 700))),
                float(np.exp(np.clip(coefficient_f + 1.96 * se_f, -700, 700))),
            ]
        rows[name] = row
    _add_bh_q_values(rows)
    return rows


def _add_bh_q_values(rows: dict[str, dict[str, Any]]) -> None:
    available = [
        (name, float(row["p_value"]))
        for name, row in rows.items()
        if name != "intercept" and row.get("p_value") is not None
    ]
    available.sort(key=lambda item: item[1])
    m = len(available)
    running = 1.0
    for rank in range(m, 0, -1):
        name, p_value = available[rank - 1]
        running = min(running, p_value * m / rank)
        rows[name]["bh_q_value"] = float(min(1.0, running))


def fit_clustered_logistic(
    X: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[str],
    names: Sequence[str],
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    positives = int(np.sum(y == 1.0))
    negatives = int(np.sum(y == 0.0))
    ridge = 1e-4
    penalty = np.ones(X.shape[1], dtype=float)
    penalty[0] = 0.0
    base = {
        "model": "ridge_stabilized_binomial_logit_with_case_clustered_sandwich_covariance",
        "ridge_penalty": ridge,
        "n": int(len(y)),
        "n_clusters": int(len(set(map(str, clusters)))),
        "positives": positives,
        "negatives": negatives,
        "effect_coding": "0 -> -0.5; 1 -> +0.5",
    }
    if positives < 5 or negatives < 5:
        return {
            **base,
            "fit_status": "not_fit_rare_outcome",
            "reason": "fewer than five observations in one outcome class",
            "coefficients": {},
        }

    def objective(beta: np.ndarray) -> float:
        z = np.clip(X @ beta, -40.0, 40.0)
        likelihood = float(np.sum(np.logaddexp(0.0, z) - y * z))
        return likelihood + 0.5 * ridge * float(np.sum(penalty * beta**2))

    def gradient(beta: np.ndarray) -> np.ndarray:
        z = np.clip(X @ beta, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-z))
        return X.T @ (probability - y) + ridge * penalty * beta

    fitted = opt.minimize(
        objective,
        np.zeros(X.shape[1], dtype=float),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 1000, "gtol": 1e-7, "ftol": 1e-12},
    )
    beta = np.asarray(fitted.x, dtype=float)
    z = np.clip(X @ beta, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-z))
    weights = probability * (1.0 - probability)
    hessian = X.T @ (X * weights[:, None]) + ridge * np.diag(penalty)
    bread = np.linalg.pinv(hessian)
    scores = X * (y - probability)[:, None]
    covariance = _cluster_covariance(
        bread=bread, scores=scores, clusters=clusters
    )
    return {
        **base,
        "fit_status": "ok" if fitted.success else "optimizer_warning",
        "optimizer_message": str(fitted.message),
        "log_likelihood": -objective(beta),
        "coefficients": _coefficient_rows(
            beta, covariance, names, logistic=True
        ),
    }


def fit_clustered_ols(
    X: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[str],
    names: Sequence[str],
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    scores = X * residual[:, None]
    covariance = _cluster_covariance(
        bread=bread, scores=scores, clusters=clusters
    )
    return {
        "model": "ols_with_case_clustered_sandwich_covariance",
        "n": int(len(y)),
        "n_clusters": int(len(set(map(str, clusters)))),
        "effect_coding": "0 -> -0.5; 1 -> +0.5",
        "outcome_mean": float(np.mean(y)),
        "outcome_sd": float(np.std(y, ddof=1)) if len(y) > 1 else None,
        "r_squared": (
            float(1.0 - np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2))
            if len(y) > 1 and float(np.sum((y - np.mean(y)) ** 2)) > 0
            else None
        ),
        "coefficients": _coefficient_rows(beta, covariance, names, logistic=False),
    }


__all__ = [
    "AXES",
    "MECHANISM_INTERACTIONS",
    "design_matrix",
    "fit_clustered_logistic",
    "fit_clustered_ols",
    "term_names",
]
