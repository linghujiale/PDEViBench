"""Run a full benchmark panel: every case and repeat for one agent configuration."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    """Load a JSONL case file."""
    cases: list[dict[str, Any]] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    *,
    case_ids: list[str] | None = None,
    limit_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Apply --case-id and --limit-cases selection."""
    selected = cases
    if case_ids:
        by_id = {case["id"]: case for case in cases}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise KeyError(f"case id(s) not found: {', '.join(missing)}")
        selected = [by_id[case_id] for case_id in case_ids]
    if limit_cases is not None:
        selected = selected[:limit_cases]
    return selected


def load_case_id_file(path: Path) -> list[str]:
    """Load case ids from a newline-delimited file.

    Blank lines and ``#`` comments are ignored. If a line has trailing columns,
    the first whitespace-delimited token is used so simple annotated lists are
    accepted.
    """
    case_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        case_ids.append(stripped.split()[0])
    return case_ids


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def case_label(case: dict[str, Any]) -> str:
    """Human-readable label for progress and reports."""
    spec = case.get("case_spec", {})
    pde = spec.get("pde", {})
    domain = spec.get("domain", {})
    bc = spec.get("bc", {})
    shape = domain.get("spec", {}).get("shape", domain.get("type", "?"))
    return f"{pde.get('operator_family', '?')}_{shape}_{bc.get('kind', '?')}"


def artifact_dir_for(
    *,
    results_root: Path,
    experiment_id: str,
    case_id: str,
    agent_name: str,
    backbone_provider: str,
    backbone_model: str,
    run_idx: int,
) -> Path:
    return (
        results_root
        / experiment_id
        / case_id
        / agent_name
        / f"{backbone_provider}-{backbone_model}"
        / f"run{run_idx}"
    )


def existing_record_from_artifacts(
    *,
    artifact_dir: Path,
    case_id: str,
    run_idx: int,
) -> dict[str, Any] | None:
    """Build a minimal resume record from an existing verdict.json, if present."""
    verdict_path = artifact_dir / "verdict.json"
    if not verdict_path.exists():
        return None
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    verdict_case_id = verdict.get("case_id")
    if verdict_case_id != case_id:
        raise ValueError(
            "artifact verdict case identity mismatch: "
            f"expected={case_id!r} observed={verdict_case_id!r} "
            f"path={verdict_path}"
        )
    verdict_run_idx = verdict.get("run_idx")
    if verdict_run_idx is not None and int(verdict_run_idx) != run_idx:
        raise ValueError(
            "artifact verdict run identity mismatch: "
            f"expected={run_idx} observed={verdict_run_idx!r} "
            f"path={verdict_path}"
        )
    method_checks = verdict.get("stages", {}).get("method", {}).get("checks", [])
    method_failed_rule_ids = [
        str(check.get("rule_id"))
        for check in method_checks
        if check.get("passed") is False and check.get("rule_id") is not None
    ]
    near_zero_exact_formula_shortcut = None
    for check in method_checks:
        if (
            check.get("rule_id") == "R-NO-EXACT-FORMULA"
            and check.get("passed") is False
        ):
            evidence = str(check.get("evidence", ""))
            match = re.search(r"classification=([A-Z_]+)", evidence)
            near_zero_exact_formula_shortcut = {
                "label": match.group(1) if match else None,
                "evidence": evidence,
            }
            break
    return {
        "case_id": case_id,
        "run_idx": run_idx,
        "rng_seed": None,
        "subprocess_returncode": 0,
        "subprocess_elapsed_seconds": 0.0,
        "agent_call": {
            "phase": "agent_call",
            "case_id": case_id,
            "status": "ok",
            "status_reason": "resumed_from_existing_verdict_artifact",
            "artifact_dir": str(artifact_dir),
        },
        "verdict_summary": {
            "phase": "verdict",
            "case_id": case_id,
            "case_pass": verdict.get("case_pass"),
            "failure_code": verdict.get("failure_code"),
            "rel_l2": verdict.get("stages", {}).get("accuracy", {}).get("rel_l2"),
            "wall_seconds_mean": verdict.get("stages", {}).get("runtime", {}).get(
                "wall_seconds_mean"
            ),
            "method_failed_rule_ids": method_failed_rule_ids,
            "near_zero_exact_formula_shortcut": near_zero_exact_formula_shortcut,
            "artifact_dir": str(artifact_dir),
        },
        "stderr_tail": "",
        "resumed_from_artifact": True,
    }


def record_has_verdict(record: dict[str, Any]) -> bool:
    """True when a record reached evaluator verdict production."""
    return bool(record.get("verdict_summary"))


def _safe_tail(path: Path, *, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def no_verdict_diagnostic_text(record: dict[str, Any]) -> str:
    """Collect best-effort diagnostic text for a no-verdict record.

    The run ledger stores only a short ``status_reason`` and driver stderr
    tail. Agent CLIs often put the actionable reason in their raw event logs,
    so reports should inspect the artifact directory before collapsing
    everything to a generic AGENT_ERROR.
    """
    parts = [
        str(record.get("stderr_tail") or ""),
        str(record.get("agent_call", {}).get("status_reason") or ""),
    ]
    artifact_dir_s = record.get("agent_call", {}).get("artifact_dir")
    if artifact_dir_s:
        artifact_dir = Path(str(artifact_dir_s))
        if not artifact_dir.is_absolute():
            artifact_dir = ROOT / artifact_dir
        for name in (
            "codex_events.jsonl",
            "codex_stderr.log",
            "claude_cli_events.jsonl",
            "claude_cli_stderr.log",
            "trace.jsonl",
        ):
            parts.append(_safe_tail(artifact_dir / name))
    return "\n".join(part for part in parts if part)


def classify_no_verdict(record: dict[str, Any]) -> dict[str, Any] | None:
    """Classify records that did not produce a verdict.

    Some no-verdict records are retriable infrastructure failures (agent CLI
    crash, driver timeout, no solver). Others are deterministic host-eval
    rejections, where the agent did produce code but the evaluator aborted
    before writing verdict.json. These should be reported explicitly instead
    of being burned repeatedly by ``--rerun-infra-failures``.
    """
    if record_has_verdict(record):
        return None

    stderr_tail = record.get("stderr_tail") or ""
    agent_status = record.get("agent_call", {}).get("status")
    diagnostic = no_verdict_diagnostic_text(record)
    diagnostic_lower = diagnostic.lower()

    if "NaN mask of submitted does not match reference" in stderr_tail:
        return {
            "code": "HOST_EVAL_DOMAIN_MASK_MISMATCH",
            "category": "host_eval_accuracy_rejection",
            "stage": "accuracy",
            "rerun_recommended": False,
            "message": (
                "Submitted solution NaN mask does not match the reference "
                "domain mask; the solver did not respect the case geometry."
            ),
        }
    if (
        "fileexistserror" in diagnostic_lower
        and "artifact_dir.mkdir" in diagnostic_lower
        and "results/batch" in diagnostic_lower
    ):
        return {
            "code": "DRIVER_RESULTS_STORAGE_UNAVAILABLE",
            "category": "infrastructure",
            "stage": "driver",
            "rerun_recommended": True,
            "message": (
                "The results storage target disappeared while the driver was "
                "creating the run artifact directory."
            ),
        }
    if record.get("driver_timeout"):
        return {
            "code": "DRIVER_TIMEOUT",
            "category": "infrastructure",
            "stage": "driver",
            "rerun_recommended": True,
            "message": "Driver subprocess exceeded the configured timeout.",
        }
    if "you've hit your usage limit" in diagnostic_lower or "usage limit" in diagnostic_lower:
        runner = str(record.get("agent_call", {}).get("runner") or "").lower()
        agent_name = str(record.get("agent_call", {}).get("agent") or "").lower()
        is_claude = (
            "claude" in runner
            or "claude" in agent_name
            or "claude" in diagnostic_lower
        )
        return {
            "code": (
                "AGENT_CLAUDE_USAGE_LIMIT"
                if is_claude
                else "AGENT_CODEX_USAGE_LIMIT"
            ),
            "category": "external_quota",
            "stage": "agent",
            "rerun_recommended": True,
            "message": (
                f"{'Claude' if is_claude else 'Codex'} CLI did not produce a solver "
                "because the account hit "
                "a usage limit. Rerun only after quota/reset is available."
            ),
        }
    if (
        "selected model is at capacity" in diagnostic_lower
        or "model is at capacity" in diagnostic_lower
    ):
        return {
            "code": "AGENT_CODEX_MODEL_CAPACITY",
            "category": "external_capacity",
            "stage": "agent",
            "rerun_recommended": True,
            "message": (
                "Codex CLI did not produce a solver because the selected "
                "model was temporarily at capacity. Rerun after a cooldown."
            ),
        }
    if (
        "failed to authenticate" in diagnostic_lower
        and "403" in diagnostic_lower
    ) or "authentication_failed" in diagnostic_lower:
        return {
            "code": "AGENT_CLAUDE_AUTH_403",
            "category": "external_auth",
            "stage": "agent",
            "rerun_recommended": True,
            "message": (
                "Claude Code CLI authentication was rejected with API 403. "
                "Rerun only after CLI/API auth is restored."
            ),
        }
    if (
        (
            "codex_core::session::turn" in diagnostic_lower
            and "stream disconnected" in diagnostic_lower
        )
        or (
            "reconnecting..." in diagnostic_lower
            and "request timed out" in diagnostic_lower
        )
    ):
        return {
            "code": "AGENT_CODEX_SAMPLING_TIMEOUT",
            "category": "external_transport",
            "stage": "agent",
            "rerun_recommended": True,
            "message": "Codex CLI sampling stream repeatedly timed out before producing a solver.",
        }
    if (
        "failed to initialize mcp client" in diagnostic_lower
        and "mcp startup failed" in diagnostic_lower
        and (
            "http/request failed" in diagnostic_lower
            or "error sending request" in diagnostic_lower
        )
    ):
        return {
            "code": "AGENT_CODEX_MCP_HANDSHAKE_TRANSPORT",
            "category": "external_transport",
            "stage": "agent",
            "rerun_recommended": True,
            "message": (
                "Codex CLI could not initialize its built-in MCP client because "
                "the remote MCP handshake transport failed."
            ),
        }
    if (
        "attempt to write a readonly database" in diagnostic_lower
        or (
            "failed to open state db" in diagnostic_lower
            and "readonly database" in diagnostic_lower
        )
    ):
        return {
            "code": "AGENT_CODEX_STATE_DB_READONLY",
            "category": "infrastructure",
            "stage": "agent",
            "rerun_recommended": True,
            "message": (
                "Codex CLI could not initialize its local state database "
                "because the state DB was read-only."
            ),
        }
    if agent_status not in {None, "ok"}:
        return {
            "code": f"AGENT_{str(agent_status).upper()}",
            "category": "infrastructure",
            "stage": "agent",
            "rerun_recommended": True,
            "message": "Agent runner ended before a solver could be evaluated.",
        }
    if "final_code.py missing" in stderr_tail or "did not produce a solver" in stderr_tail:
        return {
            "code": "AGENT_NO_SOLVER",
            "category": "infrastructure",
            "stage": "agent",
            "rerun_recommended": True,
            "message": "Agent completed without producing final_code.py.",
        }
    if "relative_to(ROOT)" in stderr_tail and "is not in the subpath of" in stderr_tail:
        return {
            "code": "DRIVER_ARTIFACT_PATH_ERROR",
            "category": "infrastructure",
            "stage": "driver",
            "rerun_recommended": True,
            "message": "Driver used a relative artifact path where an absolute path was required.",
        }
    if "OPENROUTER_API_KEY environment variable is not set" in stderr_tail:
        return {
            "code": "AGENT_OPENROUTER_API_KEY_MISSING",
            "category": "infrastructure",
            "stage": "agent",
            "rerun_recommended": True,
            "message": "OpenRouter runner could not start because OPENROUTER_API_KEY was unset.",
        }
    if stderr_tail.strip():
        return {
            "code": "HOST_EVAL_UNCLASSIFIED_EXCEPTION",
            "category": "host_eval_exception",
            "stage": "unknown",
            "rerun_recommended": False,
            "message": "Evaluator raised an exception before writing verdict.json.",
        }
    return {
        "code": "NO_VERDICT_UNKNOWN",
        "category": "unknown",
        "stage": "unknown",
        "rerun_recommended": True,
        "message": "No verdict was produced and no diagnostic was captured.",
    }


def is_infra_failure(record: dict[str, Any]) -> bool:
    """Infrastructure failure that is eligible for explicit rerun.

    Accuracy/runtime/method case failures have verdicts and are scientific
    outcomes. Those must not be retried by an infrastructure-rerun command.
    """
    agent_status = record.get("agent_call", {}).get("status")
    if record.get("driver_timeout"):
        return True
    if agent_status not in {None, "ok"}:
        return True
    if not record_has_verdict(record):
        classification = classify_no_verdict(record)
        if classification is not None:
            return bool(classification.get("rerun_recommended", True))
        return True
    return False


def agent_cooldown_wait_seconds(
    cooldown_seconds: float,
    last_cooldown_at: float | None,
    now: float,
    *,
    reset_at: float | None = None,
    reset_buffer_seconds: float = 60.0,
) -> float:
    """Return the delay required after a fresh throttling/capacity signal.

    A response that arrives shortly after the previous cooldown must wait out
    the remainder of that interval; silently skipping it creates a burst of
    immediately repeated capacity failures. If the previous interval has
    already elapsed, the fresh signal starts a new full cooldown.
    """
    if cooldown_seconds <= 0:
        interval_wait = 0.0
    elif last_cooldown_at is None:
        interval_wait = cooldown_seconds
    else:
        elapsed = max(0.0, now - last_cooldown_at)
        interval_wait = cooldown_seconds if elapsed >= cooldown_seconds else cooldown_seconds - elapsed
    reset_wait = 0.0
    if reset_at is not None and reset_at > now:
        reset_wait = reset_at - now + max(0.0, reset_buffer_seconds)
    return max(interval_wait, reset_wait)


def require_claude_cli_version(expected_version: str | None) -> None:
    """Fail before submission if the long-running panel's CLI version drifted."""
    if not expected_version:
        return
    try:
        proc = subprocess.run(
            ["claude", "--version"],
            text=True,
            capture_output=True,
            timeout=10.0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Claude CLI version pre-submit check failed: {type(exc).__name__}: {exc}"
        ) from exc
    version_full = (proc.stdout or proc.stderr or "").strip()
    actual_version = version_full.split(maxsplit=1)[0] if version_full else ""
    if proc.returncode != 0 or actual_version != expected_version:
        raise RuntimeError(
            "Claude CLI version drift before submission: "
            f"expected={expected_version} actual={version_full or 'unavailable'} "
            f"returncode={proc.returncode}"
        )


def claude_rate_limit_reset_epoch(record: dict[str, Any]) -> float | None:
    """Return the latest Claude reset epoch after a >=90% warning or rejection.

    Claude Code emits structured ``rate_limit_event`` rows even when the call
    still succeeds.  Proactively respecting ``resetsAt`` prevents a long panel
    from converting the final allowance into avoidable no-verdict quota calls.
    """
    agent = record.get("agent_call") or {}
    runner = str(agent.get("runner") or "").lower()
    artifact_dir_s = agent.get("artifact_dir")
    if "claude" not in runner or not artifact_dir_s:
        return None
    artifact_dir = Path(str(artifact_dir_s))
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir
    events = artifact_dir / "claude_cli_events.jsonl"
    if not events.is_file():
        return None
    reset_epochs: list[float] = []
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "rate_limit_event":
            continue
        info = event.get("rate_limit_info") or {}
        status = str(info.get("status") or "")
        utilization = info.get("utilization")
        warning = (
            status == "allowed_warning"
            and isinstance(utilization, (int, float))
            and float(utilization) >= 0.9
        )
        rejected = status in {"rejected", "blocked", "denied"}
        if not (warning or rejected):
            continue
        reset_at = info.get("resetsAt")
        if isinstance(reset_at, (int, float)):
            reset_epochs.append(float(reset_at))
    return max(reset_epochs, default=None)


def parse_json_blobs(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse the agent_call and verdict JSON objects printed by run_case_batch."""
    agent_call: dict[str, Any] = {}
    verdict_summary: dict[str, Any] = {}
    buf = ""
    depth = 0
    for line in stdout.splitlines():
        stripped = line.lstrip()
        if not buf and stripped.startswith("{"):
            buf = line + "\n"
            depth = line.count("{") - line.count("}")
            if depth == 0:
                blob = _loads_or_empty(buf)
                agent_call, verdict_summary = _classify_blob(
                    blob, agent_call, verdict_summary
                )
                buf = ""
        elif buf:
            buf += line + "\n"
            depth += line.count("{") - line.count("}")
            if depth == 0:
                blob = _loads_or_empty(buf)
                agent_call, verdict_summary = _classify_blob(
                    blob, agent_call, verdict_summary
                )
                buf = ""
    return agent_call, verdict_summary


def _loads_or_empty(text: str) -> dict[str, Any]:
    try:
        blob = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return blob if isinstance(blob, dict) else {}


def _classify_blob(
    blob: dict[str, Any],
    agent_call: dict[str, Any],
    verdict_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if blob.get("phase") == "agent_call":
        agent_call = blob
    elif blob.get("phase") == "verdict":
        verdict_summary = blob
    return agent_call, verdict_summary


def run_one_call(
    *,
    case_id: str,
    run_idx: int,
    experiment_id: str,
    cases_path: Path,
    wall_timeout: float,
    max_turns: int,
    library: str,
    library_guide_root: Path | None = None,
    eval_mode: str,
    container_image: str | None,
    results_root: Path,
    runner: str,
    agent_name: str,
    backbone_provider: str,
    backbone_model: str,
    skip_eval: bool,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Drive one SDK call through run_case_batch.py."""
    rng_seed = 70000 + run_idx * 1000
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_case_batch.py"),
        "--case-id",
        case_id,
        "--cases",
        str(cases_path),
        "--experiment-id",
        experiment_id,
        "--runner",
        runner,
        "--agent-name",
        agent_name,
        "--backbone-provider",
        backbone_provider,
        "--backbone-model",
        backbone_model,
        "--library",
        library,
        "--eval-mode",
        eval_mode,
        "--max-turns",
        str(max_turns),
        "--wall-timeout",
        str(wall_timeout),
        "--run-idx",
        str(run_idx),
        "--rng-seed",
        str(rng_seed),
        "--results-root",
        str(results_root),
    ]
    if container_image:
        cmd.extend(["--container-image", container_image])
    if library_guide_root is not None:
        cmd.extend(["--library-guide-root", str(library_guide_root)])
    if reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", reasoning_effort])
    if skip_eval:
        cmd.append("--skip-eval")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=wall_timeout + 120.0,
        )
        elapsed = time.monotonic() - t0
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - t0
        stdout = _decode_timeout_stream(e.stdout)
        stderr = _decode_timeout_stream(e.stderr)
        agent_call, verdict_summary = parse_json_blobs(stdout)
        if not agent_call:
            agent_call = {
                "phase": "agent_call",
                "case_id": case_id,
                "status": "timeout",
                "status_reason": f"driver subprocess exceeded {wall_timeout + 120.0}s",
            }
        return {
            "case_id": case_id,
            "run_idx": run_idx,
            "rng_seed": rng_seed,
            "subprocess_returncode": None,
            "subprocess_elapsed_seconds": elapsed,
            "agent_call": agent_call,
            "verdict_summary": verdict_summary,
            "stderr_tail": stderr[-1000:] if stderr else "",
            "driver_timeout": True,
        }

    agent_call, verdict_summary = parse_json_blobs(proc.stdout)
    return {
        "case_id": case_id,
        "run_idx": run_idx,
        "rng_seed": rng_seed,
        "subprocess_returncode": proc.returncode,
        "subprocess_elapsed_seconds": elapsed,
        "agent_call": agent_call,
        "verdict_summary": verdict_summary,
        "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
    }


def _decode_timeout_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def load_existing_runs(runs_path: Path) -> list[dict[str, Any]]:
    """Load an existing append-only run ledger."""
    if not runs_path.exists():
        return []
    runs: list[dict[str, Any]] = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            runs.append(json.loads(line))
    return runs


def aggregate_runs(
    *,
    experiment_id: str,
    cases_path: Path,
    selected_cases: list[dict[str, Any]],
    run_idx_start: int,
    n_runs: int,
    library: str,
    library_guide_root: Path | None = None,
    eval_mode: str,
    container_image: str | None,
    skip_eval: bool,
    runs: list[dict[str, Any]],
    skipped_resume: int,
    started_at: str,
    completed_at: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    selected_ids = {case["id"] for case in selected_cases}
    per_case: list[dict[str, Any]] = []
    latest_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        if run.get("case_id") in selected_ids:
            latest_by_run[(run["case_id"], int(run.get("run_idx", -1)))] = run

    grouped: dict[str, list[dict[str, Any]]] = {case["id"]: [] for case in selected_cases}
    for (case_id, _run_idx), run in latest_by_run.items():
        grouped.setdefault(case_id, []).append(run)

    for case in selected_cases:
        case_id = case["id"]
        group = sorted(grouped.get(case_id, []), key=lambda r: int(r.get("run_idx", -1)))
        rel_l2_vals = [
            float(r["verdict_summary"]["rel_l2"])
            for r in group
            if r.get("verdict_summary", {}).get("rel_l2") is not None
        ]
        case_passes = [
            bool(r.get("verdict_summary", {}).get("case_pass"))
            for r in group
            if r.get("verdict_summary")
        ]
        agent_statuses = [
            r.get("agent_call", {}).get("status")
            for r in group
            if r.get("agent_call", {}).get("status") is not None
        ]
        no_verdict_classes = Counter(
            classification["code"]
            for r in group
            if (classification := classify_no_verdict(r)) is not None
        )
        rel_l2_cv = None
        if len(rel_l2_vals) >= 2:
            mean = statistics.mean(rel_l2_vals)
            rel_l2_cv = statistics.stdev(rel_l2_vals) / mean if mean > 0 else None
        rounded = [float(f"{v:.4g}") for v in rel_l2_vals]
        per_case.append(
            {
                "case_id": case_id,
                "label": case_label(case),
                "axes": case.get("axes", {}),
                "n_expected": n_runs,
                "n_records": len(group),
                "n_verdicts": len(case_passes),
                "n_passing": sum(case_passes),
                "n_no_verdict": sum(1 for r in group if not record_has_verdict(r)),
                "no_verdict_classifications": dict(no_verdict_classes),
                "agent_statuses": agent_statuses,
                "rel_l2_runs": rel_l2_vals,
                "rel_l2_median": statistics.median(rel_l2_vals) if rel_l2_vals else None,
                "rel_l2_iqr": _iqr(rel_l2_vals),
                "rel_l2_cv": rel_l2_cv,
                "n_distinct_rel_l2": len(set(rounded)),
                "is_bimodal": (
                    len(set(rounded)) <= 2
                    and rel_l2_cv is not None
                    and rel_l2_cv > 0.20
                ),
                "subprocess_returncodes": [
                    r.get("subprocess_returncode") for r in group
                ],
            }
        )

    cv_vals = [
        item["rel_l2_cv"]
        for item in per_case
        if item["rel_l2_cv"] is not None and item["n_passing"] >= 2
    ]
    all_records = [run for run in latest_by_run.values() if run.get("case_id") in selected_ids]
    expected_total = len(selected_cases) * n_runs
    n_agent_ok = sum(
        1 for run in all_records if run.get("agent_call", {}).get("status") == "ok"
    )
    n_missing_verdict = sum(1 for run in all_records if not run.get("verdict_summary"))
    n_case_pass = sum(
        1 for run in all_records if run.get("verdict_summary", {}).get("case_pass") is True
    )
    n_case_fail = sum(
        1 for run in all_records if run.get("verdict_summary", {}).get("case_pass") is False
    )
    failure_code_counts = Counter(
        str(run.get("verdict_summary", {}).get("failure_code"))
        for run in all_records
        if run.get("verdict_summary", {}).get("failure_code") is not None
    )
    method_failed_rule_counts = Counter(
        rule_id
        for run in all_records
        for rule_id in run.get("verdict_summary", {}).get("method_failed_rule_ids", [])
    )
    n_near_zero_exact_formula_method_failures = sum(
        1
        for run in all_records
        if run.get("verdict_summary", {}).get("near_zero_exact_formula_shortcut")
    )
    no_verdict_classifications = Counter(
        classification["code"]
        for run in all_records
        if (classification := classify_no_verdict(run)) is not None
    )
    n_retriable_infra_failures = sum(1 for run in all_records if is_infra_failure(run))
    n_host_eval_domain_mask = no_verdict_classifications.get(
        "HOST_EVAL_DOMAIN_MASK_MISMATCH", 0
    )
    summary = {
        "experiment_id": experiment_id,
        "cases_path": str(cases_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "n_cases": len(selected_cases),
        "n_runs_per_case": n_runs,
        "run_idx_start": run_idx_start,
        "expected_total_calls": expected_total,
        "n_records_total": len(all_records),
        "n_ledger_records_total": sum(1 for run in runs if run.get("case_id") in selected_ids),
        "n_skipped_resume": skipped_resume,
        "library": library,
        "library_guide_root": (
            str(library_guide_root.resolve()) if library_guide_root is not None else None
        ),
        "eval_mode": eval_mode,
        "container_image": container_image,
        "skip_eval": skip_eval,
        "reasoning_effort": reasoning_effort,
        "all_expected_runs_recorded": len(all_records) >= expected_total,
        "n_agent_ok": n_agent_ok,
        "n_missing_verdict": n_missing_verdict,
        "no_verdict_classification_counts": dict(no_verdict_classifications),
        "n_host_eval_domain_mask_mismatch": n_host_eval_domain_mask,
        "n_retriable_infra_failures": n_retriable_infra_failures,
        "n_nonpass_including_classified_host_eval": n_case_fail
        + n_host_eval_domain_mask,
        "n_case_pass": n_case_pass,
        "n_case_fail": n_case_fail,
        "failure_code_counts": dict(failure_code_counts),
        "method_failed_rule_counts": dict(method_failed_rule_counts),
        "n_near_zero_exact_formula_method_failures": n_near_zero_exact_formula_method_failures,
        "max_rel_l2_cv": max(cv_vals) if cv_vals else None,
        "median_rel_l2_cv": statistics.median(cv_vals) if cv_vals else None,
        "n_bimodal_cases": sum(1 for item in per_case if item["is_bimodal"]),
    }
    return {"summary": summary, "per_case": per_case}


def _iqr(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        lower = ordered[:mid]
        upper = ordered[mid + 1 :]
    else:
        lower = ordered[:mid]
        upper = ordered[mid:]
    if not lower or not upper:
        return None
    return statistics.median(upper) - statistics.median(lower)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "benchmark" / "cases" / "pilot_phase1_8.jsonl",
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help="Default: phase2_wave1_<runner>_<library>_<UTC timestamp>",
    )
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--run-idx-start", type=int, default=0)
    p.add_argument("--wall-timeout", type=float, default=600.0)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--library", default="scipy")
    p.add_argument(
        "--library-guide-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing <library>.md prompt guides. "
            "Recorded in wave1_report.json and forwarded unchanged."
        ),
    )
    p.add_argument("--eval-mode", choices=("host", "docker"), default="host")
    p.add_argument("--container-image", default=None)
    p.add_argument("--limit-cases", type=int, default=None)
    p.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only this case id. May be supplied multiple times.",
    )
    p.add_argument(
        "--case-id-file",
        type=Path,
        default=None,
        help=(
            "Newline-delimited case ids to run. Blank lines and # comments "
            "are ignored. Values are combined with --case-id."
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Rebuild wave1_report.json from an existing runs.jsonl without "
            "launching any agent calls. Useful for interrupted runs."
        ),
    )
    p.add_argument(
        "--rerun-infra-failures",
        action="store_true",
        help=(
            "With --resume, rerun ledger records that did not reach a verdict "
            "because the agent/driver infrastructure failed. Does not rerun "
            "scientific case_pass=false verdicts."
        ),
    )
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--runner", choices=("claude", "claude_cli", "codex", "openrouter"), default="codex")
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of concurrent run_case_batch subprocesses. Default keeps legacy sequential behavior.",
    )
    p.add_argument(
        "--usage-limit-cooldown",
        type=float,
        default=float(os.environ.get("USAGE_LIMIT_COOLDOWN_SECONDS", "300")),
        help=(
            "Seconds to pause before submitting more work after an agent usage-limit "
            "failure. Set to 0 to disable. Default: 300 or USAGE_LIMIT_COOLDOWN_SECONDS."
        ),
    )
    p.add_argument(
        "--capacity-cooldown",
        type=float,
        default=float(os.environ.get("CAPACITY_COOLDOWN_SECONDS", "60")),
        help=(
            "Seconds to pause before submitting more work after a Codex model-capacity "
            "failure. Set to 0 to disable. Default: 60 or CAPACITY_COOLDOWN_SECONDS."
        ),
    )
    p.add_argument(
        "--max-new-calls",
        type=int,
        default=None,
        help=(
            "Cap the number of newly launched agent calls after resume/skip "
            "selection. Existing ledger/artifact records still count in the "
            "aggregate report. Useful for controlled recovery after quota/auth failures."
        ),
    )
    p.add_argument(
        "--halt-on-external-limit",
        action="store_true",
        help=(
            "With a sequential driver, stop after recording the first usage-limit "
            "or model-capacity outcome instead of burning later keys. Resume after "
            "the external limit clears with --resume --rerun-infra-failures."
        ),
    )
    p.add_argument("--agent-name", default=None)
    p.add_argument("--backbone-provider", default=None)
    p.add_argument("--backbone-model", default=None)
    p.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Explicit Codex model reasoning effort, forwarded to every call.",
    )
    p.add_argument(
        "--expected-claude-cli-version",
        default=os.environ.get("EXPECTED_CLAUDE_CLI_VERSION"),
        help=(
            "For --runner=claude_cli, verify this exact CLI version immediately "
            "before every task submission and abort on drift."
        ),
    )
    p.add_argument("--results-root", type=Path, default=ROOT / "results" / "batch")
    p.add_argument("--analysis-root", type=Path, default=ROOT / "analysis" / "phase2")
    args = p.parse_args(argv)

    if args.n_runs <= 0:
        print("ERROR: --n-runs must be positive.", file=sys.stderr)
        return 2
    if args.run_idx_start < 0:
        print("ERROR: --run-idx-start must be non-negative.", file=sys.stderr)
        return 2
    if args.parallel_workers <= 0:
        print("ERROR: --parallel-workers must be positive.", file=sys.stderr)
        return 2
    if args.halt_on_external_limit and args.parallel_workers != 1:
        print(
            "ERROR: --halt-on-external-limit currently requires --parallel-workers=1.",
            file=sys.stderr,
        )
        return 2
    if args.usage_limit_cooldown < 0:
        print("ERROR: --usage-limit-cooldown must be non-negative.", file=sys.stderr)
        return 2
    if args.capacity_cooldown < 0:
        print("ERROR: --capacity-cooldown must be non-negative.", file=sys.stderr)
        return 2
    if args.max_new_calls is not None and args.max_new_calls <= 0:
        print("ERROR: --max-new-calls must be positive when provided.", file=sys.stderr)
        return 2
    if args.eval_mode == "docker" and not args.skip_eval and shutil.which("docker") is None:
        print(
            "ERROR: --eval-mode=docker requested but the docker command is not available. "
            "Run scripts/check_dolfinx_container.py on the target host first.",
            file=sys.stderr,
        )
        return 2

    exp_id = args.experiment_id or (
        f"phase2_wave1_{args.runner}_{str(args.library).lower()}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    default_agent_names = {
        "codex": "codex_cli_batch",
        "claude": "claude_code_batch",
        "claude_cli": "claude_code_cli_batch",
        "openrouter": "openrouter_batch",
    }
    agent_name = args.agent_name or default_agent_names[args.runner]
    default_provider = {
        "codex": "openai",
        "claude": "anthropic",
        "claude_cli": "anthropic",
        "openrouter": "openrouter",
    }[args.runner]
    default_model = {
        "codex": "gpt-5.5",
        "claude": "claude-opus-4-7",
        "claude_cli": "claude-opus-4-7",
        "openrouter": "deepseek/deepseek-v3.2",
    }[args.runner]
    backbone_provider = args.backbone_provider or default_provider
    backbone_model = args.backbone_model or default_model
    if args.reasoning_effort is not None and args.runner != "codex":
        p.error("--reasoning-effort is currently supported only with --runner=codex")
    out_root = args.analysis_root / exp_id
    runs_path = out_root / "runs.jsonl"
    report_path = out_root / "wave1_report.json"

    if runs_path.exists() and not (args.resume or args.report_only):
        print(
            f"ERROR: {runs_path} already exists; use --resume to continue this experiment.",
            file=sys.stderr,
        )
        return 2

    try:
        case_ids = []
        if args.case_id_file is not None:
            case_ids.extend(load_case_id_file(args.case_id_file))
        if args.case_id:
            case_ids.extend(args.case_id)
        case_ids = unique_preserving_order(case_ids)
        cases = select_cases(
            load_cases(args.cases),
            case_ids=case_ids or None,
            limit_cases=args.limit_cases,
        )
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not cases:
        print("ERROR: no cases selected.", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    if args.report_only and not runs_path.exists():
        print(f"ERROR: --report-only requested but {runs_path} does not exist.", file=sys.stderr)
        return 2
    existing_runs = load_existing_runs(runs_path) if (args.resume or args.report_only) else []
    latest_recorded: dict[tuple[str, int], dict[str, Any]] = {}
    for run in existing_runs:
        latest_recorded[(run.get("case_id"), int(run.get("run_idx", -1)))] = run
    runs = list(existing_runs)
    run_indices = list(range(args.run_idx_start, args.run_idx_start + args.n_runs))
    total = len(cases) * len(run_indices)
    skipped_resume = 0
    started_at = datetime.now(timezone.utc).isoformat()

    print(f"experiment_id={exp_id}")
    print(f"plan: {len(cases)} cases × {len(run_indices)} runs = {total} agent calls")
    print(f"run_idx range: {run_indices[0]}..{run_indices[-1]}")
    print(f"cases: {args.cases}")
    print(
        f"runner={args.runner} agent={agent_name} "
        f"backbone={backbone_provider}-{backbone_model}"
    )
    print(f"reasoning_effort={args.reasoning_effort}")
    print(f"library={args.library} skip_eval={args.skip_eval} resume={args.resume}")
    print(f"library_guide_root={args.library_guide_root}")
    print(f"eval_mode={args.eval_mode} container_image={args.container_image}")
    print(f"analysis → {out_root}")
    print()

    if args.report_only:
        completed_at = datetime.now(timezone.utc).isoformat()
        report = aggregate_runs(
            experiment_id=exp_id,
            cases_path=args.cases,
            selected_cases=cases,
            run_idx_start=args.run_idx_start,
            n_runs=args.n_runs,
            library=args.library,
            library_guide_root=args.library_guide_root,
            eval_mode=args.eval_mode,
            container_image=args.container_image,
            skip_eval=args.skip_eval,
            reasoning_effort=args.reasoning_effort,
            runs=runs,
            skipped_resume=0,
            started_at=started_at,
            completed_at=completed_at,
        )
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print("report_only=true; no agent calls launched")
        print()
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
        print()
        print(f"runs trace: {runs_path}")
        print(f"report:     {report_path}")
        return 0

    halted_external_code: str | None = None
    mode = "a" if args.resume else "w"
    with runs_path.open(mode, encoding="utf-8") as fh:
        pending: list[dict[str, Any]] = []
        ordinal = 0
        for case in cases:
            case_id = case["id"]
            label = case_label(case)
            for run_idx in run_indices:
                ordinal += 1
                key = (case_id, run_idx)
                artifact_dir = artifact_dir_for(
                    results_root=args.results_root,
                    experiment_id=exp_id,
                    case_id=case_id,
                    agent_name=agent_name,
                    backbone_provider=backbone_provider,
                    backbone_model=backbone_model,
                    run_idx=run_idx,
                )
                if args.resume and key in latest_recorded:
                    latest = latest_recorded[key]
                    if not (args.rerun_infra_failures and is_infra_failure(latest)):
                        skipped_resume += 1
                        print(
                            f"[{ordinal}/{total}] SKIP ledger {label} run{run_idx} ({case_id})"
                        )
                        continue
                    print(
                        f"[{ordinal}/{total}] RERUN infra failure {label} "
                        f"run{run_idx} ({case_id})"
                    )
                if args.resume:
                    existing = existing_record_from_artifacts(
                        artifact_dir=artifact_dir, case_id=case_id, run_idx=run_idx
                    )
                    if existing is not None:
                        skipped_resume += 1
                        runs.append(existing)
                        latest_recorded[key] = existing
                        fh.write(json.dumps(existing, ensure_ascii=False) + "\n")
                        fh.flush()
                        print(
                            f"[{ordinal}/{total}] SKIP artifact {label} run{run_idx} ({case_id})"
                        )
                        continue

                pending.append(
                    {
                        "ordinal": ordinal,
                        "case_id": case_id,
                        "run_idx": run_idx,
                        "label": label,
                        "axes": case.get("axes", {}),
                        "key": key,
                        "previous": latest_recorded.get(key),
                    }
                )

        if args.max_new_calls is not None and len(pending) > args.max_new_calls:
            print(
                f"limiting new launches: {len(pending)} pending → "
                f"{args.max_new_calls} due to --max-new-calls",
                flush=True,
            )
            pending = pending[: args.max_new_calls]

        def execute_task(task: dict[str, Any]) -> dict[str, Any]:
            result = run_one_call(
                case_id=task["case_id"],
                run_idx=task["run_idx"],
                experiment_id=exp_id,
                cases_path=args.cases,
                wall_timeout=args.wall_timeout,
                max_turns=args.max_turns,
                library=args.library,
                library_guide_root=args.library_guide_root,
                eval_mode=args.eval_mode,
                container_image=args.container_image,
                results_root=args.results_root,
                runner=args.runner,
                agent_name=agent_name,
                backbone_provider=backbone_provider,
                backbone_model=backbone_model,
                skip_eval=args.skip_eval,
                reasoning_effort=args.reasoning_effort,
            )
            result["label"] = task["label"]
            result["axes"] = task["axes"]
            previous = task.get("previous")
            if previous is not None and is_infra_failure(previous):
                result["rerun_of_infra_failure"] = True
                result["previous_status"] = previous.get("agent_call", {}).get("status")
            return result

        def record_result(task: dict[str, Any], result: dict[str, Any]) -> None:
            key = task["key"]
            runs.append(result)
            latest_recorded[key] = result
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()

            agent = result.get("agent_call", {})
            verdict = result.get("verdict_summary", {})
            print(
                f"   → [{task['ordinal']}/{total}] returncode={result.get('subprocess_returncode')} "
                f"status={agent.get('status', '?'):<9s} "
                f"case_pass={verdict.get('case_pass', '?')} "
                f"rel_l2={verdict.get('rel_l2', 'n/a')} "
                f"wall={agent.get('wall_seconds_total', 0):.0f}s",
                flush=True,
            )

        def result_code(result: dict[str, Any]) -> str:
            agent = result.get("agent_call", {})
            verdict = result.get("verdict_summary", {})
            if agent.get("status") == "ok" and verdict:
                return "OK"
            classification = classify_no_verdict(result)
            return str((classification or {}).get("code", "UNCLASSIFIED"))

        if args.parallel_workers == 1:
            for task in pending:
                if args.runner == "claude_cli":
                    require_claude_cli_version(args.expected_claude_cli_version)
                print(
                    f"[{task['ordinal']}/{total}] {task['label']} run{task['run_idx']} "
                    f"(case_id={task['case_id']})",
                    flush=True,
                )
                result = execute_task(task)
                record_result(task, result)
                code = result_code(result)
                if args.halt_on_external_limit and code in {
                    "AGENT_CODEX_USAGE_LIMIT",
                    "AGENT_CLAUDE_USAGE_LIMIT",
                    "AGENT_CODEX_MODEL_CAPACITY",
                }:
                    halted_external_code = code
                    print(
                        f"HALT external limit after complete record: {code}",
                        flush=True,
                    )
                    break
        else:
            print(
                f"launching {len(pending)} pending calls with "
                f"{args.parallel_workers} parallel workers",
                flush=True,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel_workers
            ) as pool:
                pending_iter = iter(pending)
                future_to_task: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
                last_agent_cooldown_at: dict[str, float] = {}

                def submit_task(task: dict[str, Any]) -> None:
                    if args.runner == "claude_cli":
                        require_claude_cli_version(args.expected_claude_cli_version)
                    print(
                        f"[{task['ordinal']}/{total}] QUEUE {task['label']} "
                        f"run{task['run_idx']} (case_id={task['case_id']})",
                        flush=True,
                    )
                    future_to_task[pool.submit(execute_task, task)] = task

                for _ in range(min(args.parallel_workers, len(pending))):
                    submit_task(next(pending_iter))

                while future_to_task:
                    done, _ = concurrent.futures.wait(
                        future_to_task,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        task = future_to_task.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # pragma: no cover
                            result = {
                                "case_id": task["case_id"],
                                "run_idx": task["run_idx"],
                                "rng_seed": None,
                                "subprocess_returncode": None,
                                "subprocess_elapsed_seconds": 0.0,
                                "agent_call": {
                                    "phase": "agent_call",
                                    "case_id": task["case_id"],
                                    "status": "driver_exception",
                                    "status_reason": repr(exc),
                                },
                                "verdict_summary": {},
                                "stderr_tail": repr(exc),
                                "driver_exception": True,
                                "label": task["label"],
                                "axes": task["axes"],
                            }
                        record_result(task, result)

                        try:
                            next_task = next(pending_iter)
                        except StopIteration:
                            continue

                        cooldown_code = result_code(result)
                        quota_reset_at = None
                        if cooldown_code in {"OK", "AGENT_CLAUDE_USAGE_LIMIT"}:
                            quota_reset_at = claude_rate_limit_reset_epoch(result)
                            if quota_reset_at is not None and quota_reset_at > time.time():
                                cooldown_code = "AGENT_CLAUDE_USAGE_WARNING"
                        cooldown_seconds = {
                            "AGENT_CODEX_USAGE_LIMIT": args.usage_limit_cooldown,
                            "AGENT_CLAUDE_USAGE_LIMIT": args.usage_limit_cooldown,
                            "AGENT_CLAUDE_USAGE_WARNING": 0.0,
                            "AGENT_CODEX_MODEL_CAPACITY": args.capacity_cooldown,
                        }.get(cooldown_code, 0.0)
                        if cooldown_seconds > 0 or quota_reset_at is not None:
                            now = time.time()
                            wait_seconds = agent_cooldown_wait_seconds(
                                cooldown_seconds,
                                last_agent_cooldown_at.get(cooldown_code),
                                now,
                                reset_at=quota_reset_at,
                            )
                            if wait_seconds > 0:
                                print(
                                    f"agent cooldown ({cooldown_code}): sleeping "
                                    f"{wait_seconds:.0f}s before submitting more work",
                                    flush=True,
                                )
                                time.sleep(wait_seconds)
                                last_agent_cooldown_at[cooldown_code] = time.time()
                                print(
                                    "quota_resumed "
                                    f"cooldown_code={cooldown_code} "
                                    f"waited_seconds={wait_seconds:.0f} "
                                    f"reset_epoch={quota_reset_at}",
                                    flush=True,
                                )

                        submit_task(next_task)

    completed_at = datetime.now(timezone.utc).isoformat()
    report = aggregate_runs(
        experiment_id=exp_id,
        cases_path=args.cases,
        selected_cases=cases,
        run_idx_start=args.run_idx_start,
        n_runs=args.n_runs,
        library=args.library,
        library_guide_root=args.library_guide_root,
        eval_mode=args.eval_mode,
        container_image=args.container_image,
        skip_eval=args.skip_eval,
        reasoning_effort=args.reasoning_effort,
        runs=runs,
        skipped_resume=skipped_resume,
        started_at=started_at,
        completed_at=completed_at,
    )
    report["summary"]["halted_on_external_limit"] = halted_external_code
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print()
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print()
    print(f"runs trace: {runs_path}")
    print(f"report:     {report_path}")
    print()
    print("Next analysis command:")
    print(
        "python3 -m analysis.frontier.slopes "
        f"--verdicts {args.results_root / exp_id} "
        f"--cases {args.cases} "
        "--aggregation per_case --aggregator median --bootstrap 1000 "
        f"--out paper/figures/regen/fig_axis_slopes_{exp_id}.json "
        f"--per-case-out paper/figures/regen/fig_per_case_cv_{exp_id}.json"
    )
    return 75 if halted_external_code is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
