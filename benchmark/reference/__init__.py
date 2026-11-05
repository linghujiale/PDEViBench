"""Reference solution pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import high_fidelity, manufactured


def build_reference(case: dict[str, Any], *, out_root: Path) -> Path:
    ref_type = case["reference"]["type"]
    if ref_type == "manufactured":
        return manufactured.build_reference(case, out_root=out_root)
    if ref_type == "high_fidelity":
        return high_fidelity.build_reference(case, out_root=out_root)
    raise ValueError(f"unknown reference.type: {ref_type!r}")


__all__ = ["manufactured", "high_fidelity", "build_reference"]
