"""Batch Claude Code CLI runner."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from agents.base import AgentBackbone, AgentResult, AgentRunner
from agents.sdk_runner import _patch_footer_wall
from agents.trace_logger import TraceLogger

ROOT = Path(__file__).parent.parent

DEFAULT_TEMPLATE = ROOT / "agents" / "prompts" / "case_template.md"
DEFAULT_LIBRARY_GUIDES = ROOT / "agents" / "prompts" / "library_guides"


@dataclasses.dataclass
class BatchClaudeCodeCliRunner:
    """Headless Claude Code runner driven by ``claude -p``."""

    backbone: AgentBackbone = dataclasses.field(
        default_factory=lambda: AgentBackbone(
            provider="anthropic",
            model_id="claude-opus-4-7",
            decoding={"temperature": 0.2},
        )
    )
    name: str = "claude_code_cli_batch"
    max_turns: int = 30
    wall_timeout_seconds: float = 600.0
    template_path: Path = dataclasses.field(default_factory=lambda: DEFAULT_TEMPLATE)
    library_guide_root: Path = dataclasses.field(default_factory=lambda: DEFAULT_LIBRARY_GUIDES)
    claude_bin: str = "claude"
    permission_mode: str = "bypassPermissions"
    tools: tuple[str, ...] = ("Bash", "Read", "Write")
    no_session_persistence: bool = True
    extra_args: tuple[str, ...] = ()

    def run(
        self,
        *,
        case: dict[str, Any],
        run_idx: int,
        artifact_dir: Path,
        scratch_dir: Path,
        rng_seed: int,
    ) -> AgentResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = self._render_and_persist_prompt(case, artifact_dir)
        trace_path = artifact_dir / "trace.jsonl"
        raw_events_path = artifact_dir / "claude_cli_events.jsonl"
        stderr_path = artifact_dir / "claude_cli_stderr.log"
        claude_version = self._claude_version()

        command = self._command()
        header = {
            "agent": self.name,
            "case_id": case["id"],
            "run_idx": run_idx,
            "rng_seed": rng_seed,
            "backbone": self.backbone.as_dict(),
            "max_turns": self.max_turns,
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "claude_bin": self.claude_bin,
            "claude_version": claude_version,
            "claude_command": command,
            "auth_mode": "local Claude Code CLI auth/session",
            "permission_mode": self.permission_mode,
            "tools": list(self.tools),
            "no_session_persistence": self.no_session_persistence,
        }

        wall_start = time.monotonic()
        status = "ok"
        status_reason = ""
        token_in = 0
        token_out = 0
        with TraceLogger(trace_path, header=header) as log:
            log.turn(role="user", content=prompt_text)
            try:
                proc = subprocess.run(
                    command,
                    cwd=scratch_dir,
                    input=prompt_text,
                    text=True,
                    capture_output=True,
                    timeout=self.wall_timeout_seconds,
                )
                raw_events_path.write_text(proc.stdout, encoding="utf-8")
                stderr_path.write_text(proc.stderr, encoding="utf-8")
                token_in, token_out = self._log_claude_events(proc.stdout, log)
                if proc.returncode != 0:
                    status = "agent_error"
                    diagnostic = proc.stderr[-500:].strip() or _claude_error_summary(
                        proc.stdout
                    )
                    status_reason = (
                        f"claude -p exited {proc.returncode}: "
                        f"{diagnostic or 'no diagnostic emitted'}"
                    )
                    log.set_status(status, status_reason)
            except subprocess.TimeoutExpired as e:
                status = "timeout"
                status_reason = f"Exceeded wall_timeout_seconds={self.wall_timeout_seconds}"
                stdout = _decode_timeout_stream(e.stdout)
                stderr = _decode_timeout_stream(e.stderr)
                raw_events_path.write_text(stdout, encoding="utf-8")
                stderr_path.write_text(stderr, encoding="utf-8")
                token_in, token_out = self._log_claude_events(stdout, log)
                log.set_status(status, status_reason)
            except OSError as e:
                status = "agent_error"
                status_reason = f"failed to launch claude CLI: {e}"
                raw_events_path.write_text("", encoding="utf-8")
                stderr_path.write_text(status_reason, encoding="utf-8")
                log.set_status(status, status_reason)

            final_code_path = self._capture_final_code(scratch_dir, artifact_dir)
            if final_code_path is None and status == "ok":
                status = "agent_error"
                status_reason = "Claude Code CLI completed without producing solver.py"
                log.set_status(status, status_reason)
            elif final_code_path is not None:
                log.set_final_code(final_code_path)

            log._token_in = token_in
            log._token_out = token_out
            n_turns = log._n_turns
            n_tool_calls = log._n_tool_calls

        wall_total = time.monotonic() - wall_start
        _patch_footer_wall(trace_path, wall_total)

        return AgentResult(
            case_id=case["id"],
            agent=self.name,
            backbone=self.backbone,
            run_idx=run_idx,
            final_code_path=artifact_dir / "final_code.py",
            trace_path=trace_path,
            prompt_path=artifact_dir / "prompt.md",
            status=status,
            status_reason=status_reason,
            wall_seconds_total=wall_total,
            token_total_in=token_in,
            token_total_out=token_out,
            n_turns=n_turns,
            n_tool_calls=n_tool_calls,
        )

    def _command(self) -> list[str]:
        cmd = [
            self.claude_bin,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--input-format",
            "text",
            "--model",
            self.backbone.model_id,
            "--permission-mode",
            self.permission_mode,
            "--tools",
            ",".join(self.tools),
        ]
        if self.no_session_persistence:
            cmd.append("--no-session-persistence")
        cmd.extend(self.extra_args)
        return cmd

    def _render_and_persist_prompt(self, case: dict[str, Any], artifact_dir: Path) -> str:
        from agents.cc_runner import render_prompt

        library = (case.get("library_support") or ["DOLFINx"])[0]
        guide_path = self.library_guide_root / f"{library.lower()}.md"
        prompt_text = render_prompt(
            case,
            template_path=self.template_path,
            library_guide_path=guide_path if guide_path.exists() else None,
        )
        (artifact_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")
        return prompt_text

    def _claude_version(self) -> str:
        try:
            proc = subprocess.run(
                [self.claude_bin, "--version"],
                text=True,
                capture_output=True,
                timeout=10.0,
            )
        except Exception as e:
            return f"unavailable: {type(e).__name__}: {e}"
        version = (proc.stdout or proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"unavailable: exit {proc.returncode}: {version}"
        return version

    def _log_claude_events(self, stdout: str, log: TraceLogger) -> tuple[int, int]:
        assistant_token_in = 0
        assistant_token_out = 0
        result_token_in = 0
        result_token_out = 0
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "assistant":
                usage = self._log_assistant_message(event.get("message") or {}, log)
                assistant_token_in += usage[0]
                assistant_token_out += usage[1]
            elif event_type == "user":
                self._log_user_tool_results(event.get("message") or {}, log)
            elif event_type == "result":
                usage = event.get("usage") or {}
                result_token_in += _usage_input_tokens(usage)
                result_token_out += _usage_output_tokens(usage)
                if event.get("is_error"):
                    log.set_status(
                        "agent_error",
                        str(event.get("subtype") or event.get("error") or "result error"),
                    )
        token_in = assistant_token_in or result_token_in
        token_out = assistant_token_out or result_token_out
        return token_in, token_out

    def _log_assistant_message(
        self, message: dict[str, Any], log: TraceLogger
    ) -> tuple[int, int]:
        text_parts: list[str] = []
        usage = message.get("usage") or {}
        for idx, block in enumerate(message.get("content") or []):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "tool_use":
                call_id = str(block.get("id") or f"tool_use_{idx}")
                log.tool_call(
                    call_id=call_id,
                    tool=str(block.get("name") or "claude.tool_use"),
                    arguments=block.get("input") or {},
                )
        if text_parts:
            log.turn(
                role="assistant",
                content="\n".join(text_parts),
                token_count_in=_usage_input_tokens(usage),
                token_count_out=_usage_output_tokens(usage),
            )
        return _usage_input_tokens(usage), _usage_output_tokens(usage)

    def _log_user_tool_results(self, message: dict[str, Any], log: TraceLogger) -> None:
        for idx, block in enumerate(message.get("content") or []):
            if block.get("type") != "tool_result":
                continue
            content = block.get("content")
            if isinstance(content, str):
                result = content
            else:
                result = json.dumps(content, ensure_ascii=False)
            log.tool_result(
                call_id=str(block.get("tool_use_id") or f"tool_result_{idx}"),
                result=result,
                exit_code=1 if block.get("is_error") else 0,
            )

    def _capture_final_code(self, scratch_dir: Path, artifact_dir: Path) -> Path | None:
        candidates: list[Path] = []
        explicit = scratch_dir / "solver.py"
        if explicit.exists():
            candidates.append(explicit)
        else:
            for p in scratch_dir.rglob("*.py"):
                if p.stat().st_size > 32:
                    candidates.append(p)
        if not candidates:
            return None
        dst = artifact_dir / "final_code.py"
        shutil.copyfile(candidates[0], dst)
        return dst


def _usage_input_tokens(usage: dict[str, Any]) -> int:
    return int(
        (usage.get("input_tokens", 0) or 0)
        + (usage.get("cache_creation_input_tokens", 0) or 0)
        + (usage.get("cache_read_input_tokens", 0) or 0)
    )


def _usage_output_tokens(usage: dict[str, Any]) -> int:
    return int(usage.get("output_tokens", 0) or 0)


def _decode_timeout_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _claude_error_summary(stdout: str) -> str:
    """Extract a self-contained diagnostic from Claude's stream-json output."""

    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)

    for event in reversed(events):
        if event.get("type") == "result" and event.get("is_error"):
            for key in ("result", "error", "subtype"):
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[-500:]

    for event in reversed(events):
        if event.get("type") != "rate_limit_event":
            continue
        info = event.get("rate_limit_info") or {}
        status = str(info.get("status") or "unknown")
        reset_at = info.get("resetsAt")
        return f"rate limit {status}; resetsAt={reset_at}"[-500:]

    return ""


_runner: AgentRunner = BatchClaudeCodeCliRunner()
del _runner


__all__ = ["BatchClaudeCodeCliRunner"]
