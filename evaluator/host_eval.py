"""Host-mode evaluator."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluator import accuracy, execution, method_appropriateness, normalize, runtime
from evaluator.sandbox import SandboxResult
from evaluator.thresholds import RUNTIME_REPS


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _host_run_solver(
    *, solver_path: Path, scratch_dir: Path, env_extra: dict[str, str], stdout: Path, stderr: Path
) -> tuple[int, float]:
    """Run ``solver_path`` as a host subprocess; capture wall time and logs."""
    started = time.monotonic()
    env = {**os.environ, **env_extra}
    with stdout.open("wb") as so, stderr.open("wb") as se:
        proc = subprocess.run(
            [sys.executable, str(solver_path)],
            stdout=so,
            stderr=se,
            cwd=scratch_dir,
            env=env,
            check=False,
            timeout=300,
        )
    wall = time.monotonic() - started
    return proc.returncode, wall


def _expected_shape_tuple(case: dict[str, Any]) -> tuple[int, ...]:
    s = case["case_spec"]["output"]["shape"]
    return tuple(int(v) for v in s)


def _patch_scratch_path(template_solver: Path, target_scratch: Path) -> Path:
    """Copy a solver to ``target_scratch`` and rewrite ``/scratch/`` paths so the
    Docker-style ``/scratch/solution.npz`` lands at ``target_scratch/solution.npz``.
    """
    target = target_scratch / "solver.py"
    text = template_solver.read_text(encoding="utf-8")
    text = text.replace("/scratch/solution.npz", str(target_scratch / "solution.npz"))
    text = text.replace("/scratch/", str(target_scratch) + "/")
    target.write_text(text, encoding="utf-8")
    return target


def run(
    *,
    case: dict[str, Any],
    solver_path: Path,
    artifact_dir: Path,
    scratch_dir: Path,
    reference_array: np.ndarray,
    run_idx: int,
    experiment_id: str,
    git_commit: str,
    agent: str = "claude_code",
    backbone: dict[str, str] | None = None,
    solver_info: str = "",
    n_reps: int = RUNTIME_REPS,
    invoke_llm_judge: bool = False,
) -> dict[str, Any]:
    """Drive the four-stage gate against a host subprocess solver.

    Parameters
    ----------
    case
        Schema-conformant case dict.
    solver_path
        Path to ``solver.py`` containing the agent's solution. May contain
        Docker-style ``/scratch/solution.npz`` paths; this function rewrites
        them to land in ``scratch_dir``.
    artifact_dir
        Per-run output directory. Will receive ``stdout.log``, ``stderr.log``,
        ``final_code.py``, ``trace.jsonl``, ``meta.json``, ``verdict.json``,
        ``hash.json``.
    scratch_dir
        Per-rep scratch root. Each rep uses ``scratch_dir/rep{i}`` with its
        own copy of ``solver.py`` so reps cannot share output state.
    reference_array
        The case's reference solution sampled on the eval grid. Already loaded
        because callers usually want to inspect it.
    run_idx, experiment_id, git_commit, agent, backbone
        Stamped into the verdict.
    solver_info
        Free-form description (e.g. ``"scipy.sparse 5-point FD"``). Stored in
        ``meta.solver_info``.
    n_reps
        Defaults to RUNTIME_REPS (3). Each rep is a fresh subprocess.
    invoke_llm_judge
        Force the LLM-judge on. Almost always False in host mode.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    backbone = backbone or {"provider": "anthropic", "model_id": "claude-opus-4-7"}

    rep_walls: list[float] = []
    first_solution: Path | None = None
    first_stdout = artifact_dir / "stdout.log"
    first_stderr = artifact_dir / "stderr.log"
    first_exit = 0
    for rep in range(n_reps):
        rep_scratch = scratch_dir / f"rep{rep}"
        rep_scratch.mkdir(exist_ok=True, parents=True)
        rep_solver = _patch_scratch_path(solver_path, rep_scratch)
        stdout = first_stdout if rep == 0 else artifact_dir / f"stdout_rep{rep}.log"
        stderr = first_stderr if rep == 0 else artifact_dir / f"stderr_rep{rep}.log"
        exit_code, wall = _host_run_solver(
            solver_path=rep_solver,
            scratch_dir=rep_scratch,
            env_extra={
                "PDEV_CASE_ID": case["id"],
                "PDEV_SEED": str(case["provenance"].get("generator_seed", 0)),
                "PDEV_REP": str(rep),
            },
            stdout=stdout,
            stderr=stderr,
        )
        rep_walls.append(wall)
        if rep == 0:
            first_solution = rep_scratch / "solution.npz"
            first_exit = exit_code

    sandbox_rep1 = SandboxResult(
        exit_code=first_exit,
        wall_seconds=rep_walls[0],
        stdout_path=first_stdout,
        stderr_path=first_stderr,
        timed_out=False,
        oom_killed=False,
        container_image="host:fallback (no Docker; no DOLFINx)",
        container_image_digest=None,
        cmd=[sys.executable, str(solver_path)],
    )

    stage1 = execution.check(
        sandbox_result=sandbox_rep1,
        solution_path=first_solution,
        expected_shape=_expected_shape_tuple(case),
        expected_dtype=case["case_spec"]["output"]["dtype"],
    )

    stage2 = None
    stage3 = None
    stage4 = None

    if stage1.pass_:
        with np.load(first_solution, allow_pickle=False) as data:
            submitted = np.asarray(data[list(data.keys())[0]])
        stage2 = accuracy.check(
            submitted=submitted, reference=reference_array,
            e_base=case["thresholds"]["e_base"],
        )
        stage3 = runtime.check(
            wall_seconds_reps=rep_walls,
            t_base=case["thresholds"]["t_base"],
        )

    final_code_text = solver_path.read_text(encoding="utf-8")
    stage4 = method_appropriateness.check(
        case=case,
        final_code=final_code_text,
        accuracy_passed=bool(stage2 and stage2.pass_),
        accuracy_rel_l2=(stage2.rel_l2 if stage2 is not None else None),
        invoke_llm_judge=invoke_llm_judge,
    )

    case_pass = (
        stage1.pass_
        and (stage2 is not None and stage2.pass_)
        and (stage3 is not None and stage3.pass_)
        and stage4.pass_
    )
    if case_pass:
        failure = "NONE"
    elif not stage1.pass_:
        failure = "F-Exec"
    elif stage2 is not None and not stage2.pass_:
        failure = "F-Acc"
    elif stage3 is not None and not stage3.pass_:
        failure = "F-Time"
    else:
        failure = "F-Method"

    final_code_path = artifact_dir / "final_code.py"
    final_code_path.write_text(final_code_text, encoding="utf-8")
    if not (artifact_dir / "prompt.md").exists():
        (artifact_dir / "prompt.md").write_text(
            json.dumps(case["case_spec"], indent=2), encoding="utf-8"
        )

    trace_path = artifact_dir / "trace.jsonl"
    if not trace_path.exists():
        _write_minimal_trace(
            trace_path=trace_path, case=case, agent=agent, backbone=backbone,
            run_idx=run_idx, experiment_id=experiment_id, git_commit=git_commit,
            final_code_path=final_code_path, n_turns=2,
            wall_seconds_total=sum(rep_walls),
        )

    verdict: dict[str, Any] = {
        "schema_version": "1.0",
        "case_id": case["id"],
        "agent": agent,
        "backbone": backbone,
        "run_idx": run_idx,
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "started_at": _utcnow_iso(),
        "completed_at": _utcnow_iso(),
        "case_pass": bool(case_pass),
        "failure_code": failure,
        "stages": {
            "execution": stage1.to_dict(),
            "accuracy": (stage2.to_dict() if stage2 is not None else {"pass": False, "_skipped": "F-Exec"}),
            "runtime": (stage3.to_dict() if stage3 is not None else {"pass": False, "_skipped": "F-Exec"}),
            "method": stage4.to_dict(),
        },
        "artifacts": {
            "prompt_md": str(artifact_dir / "prompt.md"),
            "trace_jsonl": str(trace_path),
            "final_code": str(final_code_path),
            "stdout_log": str(first_stdout),
            "stderr_log": str(first_stderr),
            "solution_npz": str(first_solution) if first_solution else "",
            "meta_json": str(artifact_dir / "meta.json"),
            "hash_json": str(artifact_dir / "hash.json"),
        },
        "meta": {
            "solver_info": solver_info,
            "container_image": sandbox_rep1.container_image,
            "host": "host subprocess (no DOLFINx, no Docker)",
            "host_eval": True,
            "rep_walls": rep_walls,
        },
    }
    try:
        normalize.normalize_verdict(verdict=verdict, case=case)
    except FileNotFoundError:
        verdict.setdefault("meta", {})["rel_l2_normalized"] = None
    (artifact_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (artifact_dir / "meta.json").write_text(json.dumps({
        "wall_seconds_total": sum(rep_walls),
        "rep_walls": rep_walls,
        "host_eval": True,
    }, indent=2))

    hashes: dict[str, str] = {}
    for p in sorted(artifact_dir.rglob("*")):
        if not p.is_file() or p.name == "hash.json":
            continue
        hashes[str(p.relative_to(artifact_dir))] = _sha256_file(p)
    (artifact_dir / "hash.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))

    return verdict


def _write_minimal_trace(
    *, trace_path: Path, case: dict[str, Any], agent: str, backbone: dict[str, str],
    run_idx: int, experiment_id: str, git_commit: str, final_code_path: Path,
    n_turns: int, wall_seconds_total: float,
) -> None:
    """Not a recording of model turns."""
    with trace_path.open("w") as f:
        f.write(json.dumps({
            "entry_type": "header", "schema_version": "1.0", "case_id": case["id"],
            "agent": agent, "backbone": backbone, "run_idx": run_idx,
            "experiment_id": experiment_id,
            "git_commit": git_commit if len(git_commit) == 40 else "0" * 40,
            "started_at": _utcnow_iso(),
            "prompt_template_path": "agents/prompts/case_template.md",
            "prompt_sha256": "0" * 64,
            "note": "Minimal trace written by the host evaluator because the runner recorded none.",
        }) + "\n")
        f.write(json.dumps({
            "entry_type": "turn", "turn_index": 0, "role": "user",
            "content": "(case prompt — see prompt.md)",
            "timestamp": _utcnow_iso(),
            "token_count_in": 0, "token_count_out": 0, "wall_ms": 0,
        }) + "\n")
        f.write(json.dumps({
            "entry_type": "turn", "turn_index": 1, "role": "assistant",
            "content": "(Claude Code wrote solver.py via the Write tool — see final_code.py)",
            "timestamp": _utcnow_iso(),
            "token_count_in": 0, "token_count_out": 0, "wall_ms": 0,
        }) + "\n")
        f.write(json.dumps({
            "entry_type": "footer", "completed_at": _utcnow_iso(),
            "status": "ok", "status_reason": "",
            "final_code_path": str(final_code_path),
            "final_code_sha256": _sha256_file(final_code_path),
            "wall_seconds_total": float(wall_seconds_total),
            "token_total_in": 0, "token_total_out": 0,
            "n_turns": n_turns, "n_tool_calls": 0,
        }) + "\n")


__all__ = ["run"]
