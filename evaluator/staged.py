"""Four-stage evaluation orchestrator."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluator import accuracy, execution, method_appropriateness, normalize, runtime, sandbox
from evaluator.thresholds import RUNTIME_REPS


@dataclasses.dataclass
class StagedRunContext:
    """All the inputs ``run`` needs but that staged.py should not compute itself."""

    case: dict[str, Any]
    case_id: str
    agent: str
    backbone: dict[str, Any]
    run_idx: int
    experiment_id: str
    git_commit: str
    artifact_dir: Path
    scratch_dir: Path
    image: str
    final_code_path: Path
    solver_command: list[str]
    reference: np.ndarray
    energy_matrix: np.ndarray | None
    rng_seed: int


def run(ctx: StagedRunContext) -> dict[str, Any]:
    """Execute all four stages on a single (case, agent, run) tuple.

    Returns the verdict dict (also written to ``verdict.json``).
    """
    started_at = _utcnow_iso()
    rng = random.Random(ctx.rng_seed)

    solution_path = ctx.scratch_dir / "solution.npz"
    sandbox_result_rep1 = sandbox.run(
        image=ctx.image,
        command=ctx.solver_command,
        scratch_dir=ctx.scratch_dir,
        artifact_dir=ctx.artifact_dir,
        env={"PDEV_SEED": str(ctx.rng_seed), "PDEV_CASE_ID": ctx.case_id},
    )
    out = ctx.case["case_spec"]["output"]
    exec_verdict = execution.check(
        sandbox_result=sandbox_result_rep1,
        solution_path=solution_path,
        expected_shape=tuple(out["shape"]),
        expected_dtype=out["dtype"],
    )
    if not exec_verdict.pass_:
        return _finalize(
            ctx=ctx,
            started_at=started_at,
            stages={
                "execution": exec_verdict.to_dict(),
                "accuracy": _placeholder("F-Exec"),
                "runtime": _placeholder("F-Exec"),
                "method": _placeholder("F-Exec"),
            },
            case_pass=False,
            failure_code="F-Exec",
            meta_extra={"sandbox_rep1": sandbox_result_rep1.as_meta_dict()},
        )

    submitted = _load_array(solution_path)
    acc_verdict = accuracy.check(
        submitted=submitted,
        reference=ctx.reference,
        e_base=ctx.case["thresholds"]["e_base"],
        energy_matrix=ctx.energy_matrix,
    )
    if not acc_verdict.pass_:
        method_verdict = method_appropriateness.check(
            case=ctx.case,
            final_code=_read_text(ctx.final_code_path),
            accuracy_passed=False,
            accuracy_rel_l2=acc_verdict.rel_l2,
            invoke_llm_judge=False,
            llm_judge=None,
            rng_uniform_0_1=rng.random,
        )
        return _finalize(
            ctx=ctx,
            started_at=started_at,
            stages={
                "execution": exec_verdict.to_dict(),
                "accuracy": acc_verdict.to_dict(),
                "runtime": _placeholder("F-Acc"),
                "method": method_verdict.to_dict(),
            },
            case_pass=False,
            failure_code="F-Acc",
            meta_extra={"sandbox_rep1": sandbox_result_rep1.as_meta_dict()},
        )

    reps: list[float] = [sandbox_result_rep1.wall_seconds]
    for rep_i in range(1, RUNTIME_REPS):
        rep_scratch = ctx.scratch_dir.parent / f"{ctx.scratch_dir.name}_rep{rep_i}"
        rep_scratch.mkdir(parents=True, exist_ok=True)
        rep_artifact = ctx.artifact_dir / f"rep{rep_i}"
        rep_artifact.mkdir(parents=True, exist_ok=True)
        rep_result = sandbox.run(
            image=ctx.image,
            command=ctx.solver_command,
            scratch_dir=rep_scratch,
            artifact_dir=rep_artifact,
            env={"PDEV_SEED": str(ctx.rng_seed), "PDEV_CASE_ID": ctx.case_id, "PDEV_REP": str(rep_i)},
        )
        if rep_result.exit_code != 0 or rep_result.timed_out:
            reps.append(rep_result.wall_seconds)
        else:
            reps.append(rep_result.wall_seconds)

    rt_verdict = runtime.check(wall_seconds_reps=reps, t_base=ctx.case["thresholds"]["t_base"])
    if not rt_verdict.pass_:
        method_verdict = method_appropriateness.check(
            case=ctx.case,
            final_code=_read_text(ctx.final_code_path),
            accuracy_passed=True,
            accuracy_rel_l2=acc_verdict.rel_l2,
            invoke_llm_judge=False,
            llm_judge=None,
            rng_uniform_0_1=rng.random,
        )
        return _finalize(
            ctx=ctx,
            started_at=started_at,
            stages={
                "execution": exec_verdict.to_dict(),
                "accuracy": acc_verdict.to_dict(),
                "runtime": rt_verdict.to_dict(),
                "method": method_verdict.to_dict(),
            },
            case_pass=False,
            failure_code="F-Time",
            meta_extra={"sandbox_rep1": sandbox_result_rep1.as_meta_dict()},
        )

    method_verdict = method_appropriateness.check(
        case=ctx.case,
        final_code=_read_text(ctx.final_code_path),
        accuracy_passed=True,
        accuracy_rel_l2=acc_verdict.rel_l2,
        invoke_llm_judge=False,
        llm_judge=None,
        rng_uniform_0_1=rng.random,
    )
    failure = "NONE" if method_verdict.pass_ else "F-Method"
    return _finalize(
        ctx=ctx,
        started_at=started_at,
        stages={
            "execution": exec_verdict.to_dict(),
            "accuracy": acc_verdict.to_dict(),
            "runtime": rt_verdict.to_dict(),
            "method": method_verdict.to_dict(),
        },
        case_pass=method_verdict.pass_,
        failure_code=failure,
        meta_extra={"sandbox_rep1": sandbox_result_rep1.as_meta_dict()},
    )


def _finalize(
    *,
    ctx: StagedRunContext,
    started_at: str,
    stages: dict[str, Any],
    case_pass: bool,
    failure_code: str,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "schema_version": "1.0",
        "case_id": ctx.case_id,
        "agent": ctx.agent,
        "backbone": ctx.backbone,
        "run_idx": ctx.run_idx,
        "experiment_id": ctx.experiment_id,
        "git_commit": ctx.git_commit,
        "started_at": started_at,
        "completed_at": _utcnow_iso(),
        "case_pass": case_pass,
        "failure_code": failure_code,
        "stages": stages,
        "artifacts": {
            "prompt_md": str(ctx.artifact_dir / "prompt.md"),
            "trace_jsonl": str(ctx.artifact_dir / "trace.jsonl"),
            "final_code": str(ctx.final_code_path),
            "stdout_log": str(ctx.artifact_dir / "stdout.log"),
            "stderr_log": str(ctx.artifact_dir / "stderr.log"),
            "meta_json": str(ctx.artifact_dir / "meta.json"),
            "hash_json": str(ctx.artifact_dir / "hash.json"),
        },
        "meta": meta_extra or {},
    }
    try:
        normalize.normalize_verdict(verdict=verdict, case=ctx.case)
    except FileNotFoundError:
        verdict.setdefault("meta", {})["rel_l2_normalized"] = None
    (ctx.artifact_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    _write_hashes(ctx.artifact_dir)
    return verdict


def _placeholder(triggered_by: str) -> dict[str, Any]:
    return {"pass": False, "_not_evaluated_due_to_earlier_failure": triggered_by}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        return np.asarray(data[keys[0]])


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_hashes(artifact_dir: Path) -> None:
    """Write a hash.json mapping every artifact file to its SHA-256 hex digest."""
    out: dict[str, str] = {}
    for p in sorted(artifact_dir.rglob("*")):
        if not p.is_file() or p.name == "hash.json":
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        out[str(p.relative_to(artifact_dir))] = h.hexdigest()
    (artifact_dir / "hash.json").write_text(json.dumps(out, indent=2, sort_keys=True))


__all__ = ["StagedRunContext", "run"]
