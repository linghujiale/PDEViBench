#!/usr/bin/env python3
"""Run the 64 single-shot replay program-generation calls on the designated server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentBackbone  # noqa: E402
from agents.codex_runner import BatchCodexCliRunner  # noqa: E402

OUT = ROOT / "analysis/nmi_phase4b_20260811/frozen_program_replay"
CASES = OUT / "case_bank.jsonl"
SCHEDULE = OUT / "generation_schedule.jsonl"
PROMPT = ROOT / "agents/prompts/frozen_program_replay_template.md"
FREEZE = OUT / "execution_freeze_manifest.json"
MODEL_ROOT = OUT / "model_generation"
FROZEN_ROOT = OUT / "frozen_programs"
STARTED = MODEL_ROOT / "calls_started.jsonl"
RESULTS = MODEL_ROOT / "calls_completed.jsonl"
PROGRAM_MANIFEST = OUT / "program_freeze_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_immutable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_server(path: Path) -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if sys.platform != "linux" or socket.gethostname() != payload.get("hostname"):
        raise RuntimeError("model calls are restricted to the designated Linux server")
    required = (
        "operator_confirmed_designated_server",
        "model_calls_authorized",
        "hidden_assets_not_exposed_to_agents",
    )
    if not all(payload.get(key) is True for key in required):
        raise RuntimeError("server designation does not authorize isolated model calls")
    expected = [
        {
            "agent": "codex_gpt55",
            "model_id": "gpt-5.5",
            "reasoning_effort": "high",
        },
        {
            "agent": "codex_gpt56sol",
            "model_id": "gpt-5.6-sol",
            "reasoning_effort": "high",
        },
    ]
    if payload.get("codex_configurations") != expected:
        raise RuntimeError("Codex configuration differs from the two-model design")
    generation_root = Path(str(payload["agent_generation_workspace"])).resolve()
    try:
        generation_root.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("generation-only workspace must be outside the repository")
    generation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return payload, generation_root


def validate_freeze() -> dict[str, Any]:
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    if payload.get("status") != "PROTOCOL_FROZEN_BEFORE_MODEL_CALLS":
        raise RuntimeError("execution protocol is not fixed")
    expected = payload.get("assets", {})
    for path in (CASES, SCHEDULE, PROMPT, OUT / "hidden_replay_specifications.jsonl"):
        relative = str(path.relative_to(ROOT))
        if expected.get(relative) != sha256(path):
            raise RuntimeError(f"asset changed after the design was fixed: {relative}")
    return payload


def placeholder(path: Path, call_order: int, reason: str) -> None:
    message = f"generation call {call_order} failed: {reason}"
    path.write_text(
        "\"\"\"Evaluator-owned placeholder for a failed model-generation call.\"\"\"\n\n"
        "import numpy as np\n\n"
        "def solve(problem_spec: dict) -> np.ndarray:\n"
        f"    raise RuntimeError({message!r})\n",
        encoding="utf-8",
    )


def runner_for(row: dict[str, Any], generation_root: Path) -> Any:
    backbone = AgentBackbone(
        provider=str(row["provider"]),
        model_id=str(row["model_id"]),
        decoding={"temperature": 0.2},
    )
    common = {
        "backbone": backbone,
        "template_path": PROMPT,
        "max_turns": 30,
        "wall_timeout_seconds": 600.0,
    }
    if row["agent"] in {"codex_gpt55", "codex_gpt56sol"}:
        return BatchCodexCliRunner(
            **common,
            name=f"replay_{row['agent']}",
            reasoning_effort=str(row["reasoning_effort"]),
            process_cwd=generation_root,
            filesystem_hide_paths=(ROOT,),
        )
    raise ValueError(f"unknown agent {row['agent']!r}")


def complete_interrupted(
    schedule: list[dict[str, Any]],
    started: dict[int, dict[str, Any]],
    completed: dict[int, dict[str, Any]],
) -> None:
    by_order = {int(row["call_order"]): row for row in schedule}
    for order in sorted(set(started) - set(completed)):
        row = by_order[order]
        frozen = FROZEN_ROOT / f"call_{order:03d}__{row['agent']}__{row['case_id']}.py"
        submitted = frozen.is_file()
        if not submitted:
            placeholder(frozen, order, "orchestrator interrupted after immutable call-start record")
            frozen.chmod(0o444)
        append_immutable(
            RESULTS,
            {
                **row,
                "generation_status": (
                    "orchestrator_interrupted_frozen_artifact"
                    if submitted
                    else "orchestrator_interrupted_unknown_outcome"
                ),
                "status_reason": (
                    "An existing artifact preceded the interruption and was kept."
                    if submitted
                    else "No replacement call permitted after a durable call-start record."
                ),
                "program_path": str(frozen.relative_to(ROOT)),
                "program_sha256": sha256(frozen),
                "submitted_program": submitted,
            },
        )


def run(designation: Path) -> dict[str, Any]:
    _, generation_root = validate_server(designation)
    freeze = validate_freeze()
    if PROGRAM_MANIFEST.is_file():
        existing = json.loads(PROGRAM_MANIFEST.read_text(encoding="utf-8"))
        if existing.get("status") != "PROGRAMS_FROZEN" or existing.get("program_count") != 64:
            raise RuntimeError("existing program manifest is incomplete")
        for row in existing["programs"]:
            if sha256(ROOT / row["program_path"]) != row["program_sha256"]:
                raise RuntimeError(f"program changed: {row['program_id']}")
        return existing
    schedule = read_jsonl(SCHEDULE)
    cases = {row["id"]: row for row in read_jsonl(CASES)}
    if len(schedule) != 64 or len(cases) != 32:
        raise RuntimeError("design cardinality drift")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    FROZEN_ROOT.mkdir(parents=True, exist_ok=True)
    started = {int(row["call_order"]): row for row in read_jsonl(STARTED)}
    completed = {int(row["call_order"]): row for row in read_jsonl(RESULTS)}
    complete_interrupted(schedule, started, completed)
    completed = {int(row["call_order"]): row for row in read_jsonl(RESULTS)}

    for row in schedule:
        order = int(row["call_order"])
        if order in completed:
            continue
        append_immutable(STARTED, {**row, "protocol_freeze_sha256": sha256(FREEZE)})
        artifact_dir = MODEL_ROOT / f"call_{order:03d}" / "artifact"
        scratch_dir = generation_root / f"call_{order:03d}"
        result = runner_for(row, generation_root).run(
            case=cases[str(row["case_id"])],
            run_idx=0,
            artifact_dir=artifact_dir,
            scratch_dir=scratch_dir,
            rng_seed=int(row["rng_seed"]),
        )
        frozen = FROZEN_ROOT / f"call_{order:03d}__{row['agent']}__{row['case_id']}.py"
        submitted = result.status == "ok" and result.final_code_path.is_file()
        if submitted:
            frozen.write_bytes(result.final_code_path.read_bytes())
        else:
            placeholder(frozen, order, result.status_reason or result.status)
        frozen.chmod(0o444)
        completed_row = {
            **row,
            "generation_status": result.status,
            "status_reason": result.status_reason,
            "wall_seconds": result.wall_seconds_total,
            "token_total_in": result.token_total_in,
            "token_total_out": result.token_total_out,
            "n_turns": result.n_turns,
            "n_tool_calls": result.n_tool_calls,
            "artifact_dir": str(artifact_dir.relative_to(ROOT)),
            "prompt_sha256": sha256(result.prompt_path),
            "program_path": str(frozen.relative_to(ROOT)),
            "program_sha256": sha256(frozen),
            "submitted_program": submitted,
        }
        append_immutable(RESULTS, completed_row)
        completed[order] = completed_row

    rows = [completed[index] for index in range(1, 65)]
    programs = [
        {
            "program_id": f"AGENT-{row['agent']}-{row['case_id']}",
            "case_id": row["case_id"],
            "agent": row["agent"],
            "model_id": row["model_id"],
            "call_order": row["call_order"],
            "generation_status": row["generation_status"],
            "submitted_program": row["submitted_program"],
            "program_path": row["program_path"],
            "program_sha256": row["program_sha256"],
        }
        for row in rows
    ]
    manifest = {
        "schema_version": 1,
        "status": "PROGRAMS_FROZEN",
        "protocol_freeze_sha256": sha256(FREEZE),
        "program_count": len(programs),
        "submitted_program_count": sum(bool(row["submitted_program"]) for row in programs),
        "generation_failure_count": sum(not bool(row["submitted_program"]) for row in programs),
        "calls_started_sha256": sha256(STARTED),
        "calls_completed_sha256": sha256(RESULTS),
        "programs": programs,
        "protocol_freeze": freeze["status"],
    }
    PROGRAM_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    PROGRAM_MANIFEST.chmod(0o444)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-designation", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args.server_designation)
    print(
        json.dumps(
            {
                "program_count": manifest["program_count"],
                "generation_failure_count": manifest["generation_failure_count"],
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
