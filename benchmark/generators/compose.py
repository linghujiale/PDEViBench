"""Assemble a full PDE case from per-axis generators."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.generators import bcs, coupling, discretization_hints, geometries, operators, structure


def axis_signature(*, O: int, G: int, B: int, D: int, S: int, C: int) -> str:
    return f"O{O}G{G}B{B}D{D}S{S}C{C}"


def make_case(
    *,
    axes: dict[str, int],
    seed: int,
    seq: int,
    phase: str = "smoke",
    library_support: list[str] | None = None,
    e_base: float = 1e-4,
    t_base: float = 2.0,
    generator_version: str = "<unset-pre-commit>",
    op_family: str | None = None,
    geo_family: str | None = None,
    bc_family: str | None = None,
) -> dict[str, Any]:
    """Compose a schema-conformant case dict.

    The output is *not* yet written to JSONL — see ``benchmark.validate`` for the
    write-and-validate path.

    ``op_family`` / ``geo_family`` / ``bc_family`` override the seed-driven
    deterministic family selection. Necessary when a caller wants the cartesian
    product of (operator, geometry, bc) across families — leaving them all to
    seed-mod-len couples the axes.
    """
    O, G, B, D, S, C = (axes[k] for k in "OGBDSC")

    op = operators.generate(level=O, seed=seed, family=op_family)
    geo = geometries.generate(level=G, seed=seed, family=geo_family)
    bc = bcs.generate(level=B, seed=seed, family=bc_family, geometry_name=geo.name)
    discr = discretization_hints.generate(level=D, seed=seed)
    struct = structure.generate(level=S, seed=seed)
    coup = coupling.generate(level=C, seed=seed)

    from benchmark.reference.manufactured import forcing_descriptor as _fd
    effective_params = dict(op.parameters)
    if geo.name == "annulus":
        effective_params.setdefault("r_in", float(geo.spec.get("r_in", 0.5)))
        effective_params.setdefault("r_out", float(geo.spec.get("r_out", 1.0)))
    manufactured_forcing = _fd(
        operator_family=op.operator_family,
        parameters=effective_params,
        geometry_shape=geo.name,
        bc_kind=bc.kind,
    )
    vector_lifted = False
    if C == 1 and manufactured_forcing is not None and not manufactured_forcing.get("is_vector"):
        manufactured_forcing = _vector_lift_forcing(manufactured_forcing)
        vector_lifted = True

    eval_grid = _default_eval_grid(geo)
    n_grid_points = _grid_npoints(eval_grid)
    if manufactured_forcing is not None and manufactured_forcing.get("is_vector"):
        output_shape: list[int] = [n_grid_points, 2]
    else:
        output_shape = [n_grid_points]
    case_id = f"{axis_signature(O=O, G=G, B=B, D=D, S=S, C=C)}_s{seed}_{seq:03d}"

    if geo.name in {"L_shape_2d", "square_with_hole"}:
        e_base = max(e_base, 0.05)

    tau_acc = max(10.0 * e_base, 1e-6)
    tau_time = 3.0 * t_base

    statement = op.statement
    if vector_lifted:
        statement = (
            "Solve the two-component vector-valued version of the following PDE "
            "componentwise for u = (u1, u2). Each component has the same operator "
            "and boundary assignment; the forcing below gives the two components. "
            + op.statement
        )
    pde_block: dict[str, Any] = {
        "statement": statement if manufactured_forcing is None else (
            statement + f"\n\nFor this case, take f = {manufactured_forcing['expr_human']}."
        ),
        "operator_family": op.operator_family,
        "parameters": dict(op.parameters),
    }
    if manufactured_forcing is not None:
        pde_block["forcing"] = manufactured_forcing
    if vector_lifted:
        pde_block["vector_lift"] = {
            "components": 2,
            "component_2_scale": 0.5,
            "construction": "componentwise scalar PDE lift for binary C=1 benchmark cells",
        }
    ic_block: dict[str, Any] = {}
    if manufactured_forcing is not None:
        if "ic_sympy" in manufactured_forcing:
            ic_block["u0_sympy"] = manufactured_forcing["ic_sympy"]
            ic_block["u0_human"] = manufactured_forcing.get("ic_human", "")
        if "v0_sympy" in manufactured_forcing:
            ic_block["v0_sympy"] = manufactured_forcing["v0_sympy"]
        if "T_final" in manufactured_forcing:
            ic_block["T_final"] = manufactured_forcing["T_final"]

    case_spec_block: dict[str, Any] = {
        "pde": pde_block,
        "domain": {"type": geo.type_, "spec": dict(geo.spec)},
        "bc": {"kind": bc.kind, **bc.detail},
        "eval_grid": eval_grid,
        "output": {"dtype": "float64", "shape": output_shape},
    }
    if ic_block:
        case_spec_block["ic"] = ic_block

    case: dict[str, Any] = {
        "schema_version": "1.0",
        "id": case_id,
        "axes": {"O": O, "G": G, "B": B, "D": D, "S": S, "C": C},
        "case_spec": case_spec_block,
        "reference": _reference_block(
            case_id=case_id,
            geometry_name=geo.name,
            manufactured_forcing=manufactured_forcing,
        ),
        "thresholds": {
            "alpha_acc": 10,
            "alpha_time": 3,
            "tau_min": 1e-6,
            "tau_acc": tau_acc,
            "tau_time": tau_time,
            "e_base": e_base,
            "t_base": t_base,
        },
        "library_support": library_support or ["DOLFINx", "Firedrake"],
        "provenance": {
            "source": "generated",
            "generator_seed": seed,
            "generator_version": generator_version,
            "created_at": _utcnow_iso(),
            "phase": phase,
        },
    }
    case["case_spec"].setdefault("hints", {})
    case["case_spec"]["hints"]["discretization"] = {
        "level": discr.level,
        "label": discr.label,
        "suggested_spaces": list(discr.suggested_spaces),
    }
    case["case_spec"]["hints"]["structure"] = {
        "level": struct.level,
        "label": struct.label,
        "constraints": list(struct.constraints),
    }
    case["case_spec"]["hints"]["coupling"] = {
        "level": coup.level,
        "label": coup.label,
        "multi_field": coup.multi_field,
    }
    return case


def _vector_lift_forcing(desc: dict[str, Any]) -> dict[str, Any]:
    """Turn a scalar manufactured forcing descriptor into a two-component one."""
    lifted = dict(desc)
    expr = str(desc["expr_sympy"])
    human = str(desc["expr_human"])
    lifted["expr_sympy"] = f"({expr}, 0.5*({expr}))"
    lifted["expr_human"] = f"F = (f, 0.5 f), where {human}"
    lifted["is_vector"] = True
    lifted["vector_lift_from_scalar"] = True
    lifted["component_2_scale"] = 0.5
    u_exact = desc.get("u_exact_sympy")
    if u_exact:
        lifted["u_exact_sympy"] = f"({u_exact}, 0.5*({u_exact}))"
    if "ic_sympy" in desc:
        ic = str(desc["ic_sympy"])
        lifted["ic_sympy"] = f"({ic}, 0.5*({ic}))"
        lifted["ic_human"] = (
            "u(r,0) or u(x,0) is the two-component vector lift of the scalar "
            f"initial condition: {desc.get('ic_human', ic)}"
        )
    if "v0_sympy" in desc:
        v0 = str(desc["v0_sympy"])
        lifted["v0_sympy"] = f"({v0}, 0.5*({v0}))"
    return lifted


def _maybe_manufactured_forcing(
    *, op: operators.OperatorSpec, geo: geometries.GeometrySpec, bc: bcs.BCSpec
) -> dict[str, Any] | None:
    """Keys (when non-None):
        ``expr_sympy``   — sympy-parseable string for the forcing function
        ``expr_human``   — plain-language rendering for the case statement
        ``u_exact_sympy`` — the manufactured exact solution
    """
    if (
        op.operator_family == "Poisson"
        and geo.name == "unit_square"
        and bc.kind == "dirichlet_homogeneous"
    ):
        return {
            "expr_sympy": "2*pi**2 * sin(pi*x) * sin(pi*y)",
            "expr_human": "2π² sin(πx) sin(πy)",
            "u_exact_sympy": "sin(pi*x) * sin(pi*y)",
        }
    return None


def _reference_block(
    *, case_id: str, geometry_name: str, manufactured_forcing: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the case.reference block. Cases on geometries where the
    manufactured MMS exists use ``type='manufactured'``; geometries without
    a closed-form (currently only L_shape_2d) use ``type='high_fidelity'``
    with a placeholder convergence_check that the reference builder later
    fills in via the ``_high_fidelity_info`` side-channel.
    """
    needs_high_fidelity = (
        geometry_name == "L_shape_2d"
        or (manufactured_forcing is not None
            and manufactured_forcing.get("reference_type") == "high_fidelity")
    )
    if needs_high_fidelity:
        return {
            "type": "high_fidelity",
            "path": f"benchmark/reference/cases/{case_id}.npz",
            "construction_log": (
                "Numerical reference computed by benchmark/reference/high_fidelity.py. "
                "Multi-level FD convergence check; gap_relative_l2 < 0.05 required."
            ),
            "convergence_check": {
                "levels": [16, 61, 121],
                "gap_relative_l2": 0.0,
                "polynomial_order": 1,
            },
        }
    return {
        "type": "manufactured",
        "path": f"benchmark/reference/cases/{case_id}.npz",
        "construction_log": "Manufactured solution derived in benchmark/reference/manufactured.py; sampling matches eval_grid.",
    }


def _default_eval_grid(geo: geometries.GeometrySpec) -> dict[str, Any]:
    """Default eval grid per geometry."""
    if geo.name == "unit_square":
        return {"type": "uniform", "spec": {"nx": 16, "ny": 16, "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0}}
    if geo.name == "unit_cube":
        return {"type": "uniform", "spec": {"nx": 8, "ny": 8, "nz": 8}}
    if geo.name == "disk":
        return {"type": "structured", "spec": {"n_radial": 16, "n_angular": 32}}
    if geo.name == "ball":
        return {"type": "structured", "spec": {"n_radial": 8, "n_polar": 16, "n_azimuthal": 16}}
    if geo.name == "L_shape_2d":
        return {"type": "uniform_masked",
                "spec": {"nx": 16, "ny": 16, "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,
                         "mask": {"reject_region": [[0.5, 1.0], [0.5, 1.0]]}}}
    if geo.name == "square_with_hole":
        return {"type": "uniform_masked",
                "spec": {"nx": 16, "ny": 16, "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,
                         "mask": {"reject_disk": {"center": [0.5, 0.5], "radius": 0.2}}}}
    if geo.name == "annulus":
        return {"type": "structured_polar",
                "spec": {"n_radial": 8, "n_angular": 32,
                         "r_in": float(geo.spec["r_in"]), "r_out": float(geo.spec["r_out"]),
                         "center": list(geo.spec["center"])}}
    if geo.name == "sector":
        return {"type": "structured_polar_sector",
                "spec": {"n_radial": 16, "n_angular": 16,
                         "r_max": float(geo.spec["r_max"]),
                         "theta_max": float(geo.spec["theta_max"]),
                         "center": list(geo.spec["center"])}}
    return {"type": "uniform", "spec": {"nx": 16, "ny": 16}}


def _grid_npoints(eval_grid: dict[str, Any]) -> int:
    spec = eval_grid["spec"]
    t = eval_grid["type"]
    if t in ("uniform", "uniform_masked"):
        n = 1
        for k in ("nx", "ny", "nz"):
            if k in spec:
                n *= int(spec[k])
        return n
    if t == "structured":
        n = 1
        for v in spec.values():
            if isinstance(v, int):
                n *= v
        return n
    if t in ("structured_polar", "structured_polar_sector"):
        return int(spec["n_radial"]) * int(spec["n_angular"])
    return 1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_cases(cases: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False))
            f.write("\n")


__all__ = ["axis_signature", "make_case", "write_cases"]
