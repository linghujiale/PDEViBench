"""Stage 1 — Execution check."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from evaluator.sandbox import SandboxResult


@dataclasses.dataclass(frozen=True)
class ExecutionVerdict:
    """Schema-conformant Stage-1 verdict (matches result.schema.json#stages.execution)."""

    pass_: bool
    ran_to_completion: bool
    output_shape: list[int] | None
    output_dtype: str | None
    all_finite: bool
    exit_code: int
    stdout_path: str
    stderr_path: str
    solution_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_,
            "ran_to_completion": self.ran_to_completion,
            "output_shape": self.output_shape,
            "output_dtype": self.output_dtype,
            "all_finite": self.all_finite,
            "exit_code": self.exit_code,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            **({"solution_path": self.solution_path} if self.solution_path else {}),
        }


def check(
    *,
    sandbox_result: SandboxResult,
    solution_path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> ExecutionVerdict:
    """Apply the Stage-1 gate.

    Parameters
    ----------
    sandbox_result
        Output of ``evaluator.sandbox.run`` for this submission.
    solution_path
        Where the agent's solver is expected to have written its output.
        We accept ``.npz`` files with a single array named ``solution`` or
        an unnamed array (i.e. ``np.load(...)``\\ -compatible).
    expected_shape
        From ``case_spec.output.shape``.
    expected_dtype
        From ``case_spec.output.dtype``.
    """
    ran = sandbox_result.exit_code == 0 and not sandbox_result.timed_out
    if not ran:
        return ExecutionVerdict(
            pass_=False,
            ran_to_completion=False,
            output_shape=None,
            output_dtype=None,
            all_finite=False,
            exit_code=sandbox_result.exit_code,
            stdout_path=str(sandbox_result.stdout_path),
            stderr_path=str(sandbox_result.stderr_path),
            solution_path=None,
        )
    if not solution_path.is_file():
        return ExecutionVerdict(
            pass_=False,
            ran_to_completion=True,
            output_shape=None,
            output_dtype=None,
            all_finite=False,
            exit_code=sandbox_result.exit_code,
            stdout_path=str(sandbox_result.stdout_path),
            stderr_path=str(sandbox_result.stderr_path),
            solution_path=None,
        )

    arr = _load_solution(solution_path)
    shape_ok = tuple(arr.shape) == tuple(expected_shape)
    dtype_ok = str(arr.dtype) == expected_dtype
    finite_ok = bool(np.all(np.isfinite(arr) | np.isnan(arr)))

    return ExecutionVerdict(
        pass_=shape_ok and dtype_ok and finite_ok,
        ran_to_completion=True,
        output_shape=list(arr.shape),
        output_dtype=str(arr.dtype),
        all_finite=finite_ok,
        exit_code=sandbox_result.exit_code,
        stdout_path=str(sandbox_result.stdout_path),
        stderr_path=str(sandbox_result.stderr_path),
        solution_path=str(solution_path),
    )


def _load_solution(path: Path) -> np.ndarray:
    """Tolerant .npz loader: accepts a single-array ``.npz`` or a ``.npy``.

    The agent may save with ``np.savez`` (named) or ``np.save`` (unnamed).
    We support both rather than mandate one convention — fewer ways to fail
    Stage 1 for accidental shape/encoding reasons.
    """
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"{path}: empty .npz")
            arr = data[keys[0]]
            return np.asarray(arr)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    raise ValueError(f"{path}: unsupported solution format")


__all__ = ["ExecutionVerdict", "check"]
