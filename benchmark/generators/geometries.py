"""Axis-G — Geometric complexity."""

from __future__ import annotations

import dataclasses
from typing import Any


GEOMETRY_FAMILIES: dict[int, list[str]] = {
    0: ["unit_square", "unit_cube", "disk", "ball"],
    1: ["L_shape_2d", "square_with_hole", "annulus", "sector"],
    2: ["multi_hole_2d", "dumbbell", "gear_2d", "curved_boundary_2d"],
    3: ["re_entrant_3d", "thin_feature_3d", "multiply_connected_3d", "csg_3d"],
    4: ["heat_sink_step", "vascular_bifurcation_stl"],
}


@dataclasses.dataclass(frozen=True)
class GeometrySpec:
    name: str
    type_: str
    spec: dict[str, Any]
    dim: int


def generate(*, level: int, family: str | None = None, seed: int = 0) -> GeometrySpec:
    if level not in GEOMETRY_FAMILIES:
        raise ValueError(f"unknown G-level: {level}")
    fam = GEOMETRY_FAMILIES[level]
    chosen = family if family is not None else fam[seed % len(fam)]
    if chosen not in fam:
        raise ValueError(f"family {chosen!r} not at G-level {level}")
    builder = _builders.get(chosen)
    if builder is None:
        raise NotImplementedError(
            f"geometry {chosen!r} (G{level}) is not implemented."
        )
    return builder()


def _unit_square() -> GeometrySpec:
    return GeometrySpec(name="unit_square", type_="primitive",
                        spec={"shape": "unit_square"}, dim=2)


def _unit_cube() -> GeometrySpec:
    return GeometrySpec(name="unit_cube", type_="primitive",
                        spec={"shape": "unit_cube"}, dim=3)


def _disk() -> GeometrySpec:
    return GeometrySpec(name="disk", type_="primitive",
                        spec={"shape": "disk", "radius": 1.0, "center": [0.0, 0.0]}, dim=2)


def _ball() -> GeometrySpec:
    return GeometrySpec(name="ball", type_="primitive",
                        spec={"shape": "ball", "radius": 1.0, "center": [0.0, 0.0, 0.0]}, dim=3)


def _L_shape_2d() -> GeometrySpec:
    return GeometrySpec(
        name="L_shape_2d",
        type_="parameterized",
        spec={
            "shape": "L_shape_2d",
            "outer": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]],
            "reentrant_corner": [0.5, 0.5],
        },
        dim=2,
    )


def _square_with_hole() -> GeometrySpec:
    return GeometrySpec(
        name="square_with_hole",
        type_="csg",
        spec={
            "shape": "square_with_hole",
            "outer": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "hole_center": [0.5, 0.5],
            "hole_radius": 0.2,
        },
        dim=2,
    )


def _annulus() -> GeometrySpec:
    return GeometrySpec(
        name="annulus",
        type_="parameterized",
        spec={
            "shape": "annulus",
            "r_in": 0.5,
            "r_out": 1.0,
            "center": [0.0, 0.0],
        },
        dim=2,
    )


def _sector() -> GeometrySpec:
    return GeometrySpec(
        name="sector",
        type_="parameterized",
        spec={
            "shape": "sector",
            "r_max": 1.0,
            "theta_max": 2.356194490192345,
            "center": [0.0, 0.0],
        },
        dim=2,
    )


_builders = {
    "unit_square": _unit_square,
    "unit_cube": _unit_cube,
    "disk": _disk,
    "ball": _ball,
    "L_shape_2d": _L_shape_2d,
    "square_with_hole": _square_with_hole,
    "annulus": _annulus,
    "sector": _sector,
}


__all__ = ["GeometrySpec", "GEOMETRY_FAMILIES", "generate"]
