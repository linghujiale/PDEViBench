"""Baseline-normalized accuracy metric for cross-axis fairness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
DEFAULT_BASELINES_PATH = ROOT / "benchmark" / "baselines.json"


def _cell_key(case: dict[str, Any]) -> tuple[str, str, str]:
    op = case["case_spec"]["pde"]["operator_family"]
    geo = case["case_spec"]["domain"]["spec"].get("shape", "")
    bc = case["case_spec"]["bc"]["kind"]
    return op, geo, bc


def load_baselines(path: Path | None = None) -> dict[tuple[str, str, str], float]:
    """Read ``benchmark/baselines.json`` into a (op, geo, bc) → rel_l2_baseline map."""
    src = Path(path) if path else DEFAULT_BASELINES_PATH
    data = json.loads(src.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], float] = {}
    for entry in data["cells"]:
        out[(entry["operator_family"], entry["geometry_shape"], entry["bc_kind"])] = \
            float(entry["rel_l2_baseline"])
    return out


def load_wall_seconds_baselines(path: Path | None = None) -> dict[tuple[str, str, str], float]:
    """Read per-cell wall_seconds_baseline (Q3 runtime-confound mirror).

    Mirrors ``load_baselines`` but for wall-time. Cells without a
    ``wall_seconds_baseline`` key are silently omitted; analysis tools
    can detect missing cells via ``KeyError`` lookup.
    """
    src = Path(path) if path else DEFAULT_BASELINES_PATH
    data = json.loads(src.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str], float] = {}
    for entry in data["cells"]:
        if "wall_seconds_baseline" in entry:
            out[(entry["operator_family"], entry["geometry_shape"], entry["bc_kind"])] = \
                float(entry["wall_seconds_baseline"])
    return out


def normalize_verdict(
    *,
    verdict: dict[str, Any],
    case: dict[str, Any],
    baselines: dict[tuple[str, str, str], float] | None = None,
    wall_baselines: dict[tuple[str, str, str], float] | None = None,
) -> dict[str, Any]:
    """Add ``meta.rel_l2_normalized`` and ``meta.wall_seconds_normalized``
    (with their respective baselines) to a verdict dict.

    Modifies ``verdict`` in place and returns it. Missing baseline →
    None in ``meta`` so analysis tools can detect the gap explicitly.
    """
    if baselines is None:
        baselines = load_baselines()
    if wall_baselines is None:
        try:
            wall_baselines = load_wall_seconds_baselines()
        except FileNotFoundError:
            wall_baselines = {}
    key = _cell_key(case)
    meta = verdict.setdefault("meta", {})

    base = baselines.get(key)
    rel = verdict.get("stages", {}).get("accuracy", {}).get("rel_l2")
    if base is None or rel is None:
        meta["rel_l2_normalized"] = None
        meta["rel_l2_baseline"] = base
    else:
        meta["rel_l2_normalized"] = rel / base
        meta["rel_l2_baseline"] = base
    meta["rel_l2_baseline_cell"] = list(key)

    wall_base = wall_baselines.get(key)
    wall = verdict.get("stages", {}).get("runtime", {}).get("wall_seconds_mean")
    if wall_base is None or wall is None:
        meta["wall_seconds_normalized"] = None
        meta["wall_seconds_baseline"] = wall_base
    else:
        meta["wall_seconds_normalized"] = wall / wall_base
        meta["wall_seconds_baseline"] = wall_base

    return verdict


__all__ = ["load_baselines", "normalize_verdict", "DEFAULT_BASELINES_PATH"]
