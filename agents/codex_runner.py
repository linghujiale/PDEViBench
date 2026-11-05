"""Batch Codex CLI runner."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agents.base import AgentBackbone, AgentResult, AgentRunner
from agents.sdk_runner import _patch_footer_wall
from agents.trace_logger import TraceLogger

ROOT = Path(__file__).parent.parent

DEFAULT_TEMPLATE = ROOT / "agents" / "prompts" / "case_template.md"
DEFAULT_LIBRARY_GUIDES = ROOT / "agents" / "prompts" / "library_guides"


@dataclasses.dataclass
class BatchCodexCliRunner:
    """Headless Codex runner driven by ``codex exec``."""

    backbone: AgentBackbone = dataclasses.field(
        default_factory=lambda: AgentBackbone(
            provider="openai",
            model_id="gpt-5.5",
            decoding={"temperature": 0.2},
        )
    )
    name: str = "codex_cli_batch"
    max_turns: int = 30
    wall_timeout_seconds: float = 600.0
    template_path: Path = dataclasses.field(default_factory=lambda: DEFAULT_TEMPLATE)
    library_guide_root: Path = dataclasses.field(default_factory=lambda: DEFAULT_LIBRARY_GUIDES)
    codex_bin: str = "codex"
    sandbox_mode: str = "danger-full-access"
    bypass_approvals_and_sandbox: bool = True
    ignore_user_config: bool = True
    ignore_rules: bool = True
    ephemeral: bool = True
    disable_plugins: bool = True
    isolate_codex_home: bool = True
    codex_home_source: Path | None = None
    process_cwd: Path = dataclasses.field(default_factory=lambda: ROOT)
    codex_home_copy_files: tuple[str, ...] = (
        "auth.json",
        "installation_id",
        "version.json",
    )
    extra_args: tuple[str, ...] = ()
    reasoning_effort: str | None = None
    filesystem_hide_paths: tuple[Path, ...] = ()
    mount_namespace_helper: Path = dataclasses.field(
        default_factory=lambda: ROOT / "scripts" / "nmi_hidden_mount_namespace.py"
    )

    def __post_init__(self) -> None:
        if self.reasoning_effort is None:
            return
        allowed = {"none", "low", "medium", "high", "xhigh", "max"}
        if self.reasoning_effort not in allowed:
            raise ValueError(
                "reasoning_effort must be one of "
                + ", ".join(sorted(allowed))
            )
        decoding = dict(self.backbone.decoding)
        recorded = decoding.get("reasoning_effort")
        if recorded is not None and recorded != self.reasoning_effort:
            raise ValueError(
                "backbone decoding reasoning_effort conflicts with runner setting: "
                f"{recorded!r} != {self.reasoning_effort!r}"
            )
        decoding["reasoning_effort"] = self.reasoning_effort
        self.backbone = dataclasses.replace(self.backbone, decoding=decoding)

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
        raw_events_path = artifact_dir / "codex_events.jsonl"
        stderr_path = artifact_dir / "codex_stderr.log"
        last_message_path = artifact_dir / "codex_last_message.txt"
        runtime_last_message_path = scratch_dir / ".codex_last_message.txt"

        header = {
            "agent": self.name,
            "case_id": case["id"],
            "run_idx": run_idx,
            "rng_seed": rng_seed,
            "backbone": self.backbone.as_dict(),
            "max_turns": self.max_turns,
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "codex_bin": self.codex_bin,
            "sandbox_mode": self.sandbox_mode,
            "ignore_user_config": self.ignore_user_config,
            "ignore_rules": self.ignore_rules,
            "ephemeral": self.ephemeral,
            "disable_plugins": self.disable_plugins,
            "isolate_codex_home": self.isolate_codex_home,
            "reasoning_effort": self.reasoning_effort,
        }

        wall_start = time.monotonic()
        status = "ok"
        status_reason = ""
        token_in = 0
        token_out = 0
        with TraceLogger(trace_path, header=header) as log:
            log.turn(role="user", content=prompt_text)
            try:
                with self._codex_env() as env:
                    proc = subprocess.run(
                        self._command(
                            scratch_dir=scratch_dir,
                            last_message_path=runtime_last_message_path,
                        ),
                        cwd=self.process_cwd,
                        input=prompt_text,
                        text=True,
                        capture_output=True,
                        timeout=self.wall_timeout_seconds,
                        env=env,
                    )
                raw_events_path.write_text(proc.stdout, encoding="utf-8")
                stderr_path.write_text(proc.stderr, encoding="utf-8")
                if runtime_last_message_path.is_file():
                    shutil.copyfile(runtime_last_message_path, last_message_path)
                token_in, token_out = self._log_codex_events(proc.stdout, log)
                if proc.returncode != 0:
                    status = "agent_error"
                    status_reason = (
                        f"codex exec exited {proc.returncode}: "
                        f"{proc.stderr[-500:].strip()}"
                    )
                    log.set_status(status, status_reason)
            except subprocess.TimeoutExpired as e:
                status = "timeout"
                status_reason = f"Exceeded wall_timeout_seconds={self.wall_timeout_seconds}"
                stdout = _decode_timeout_stream(e.stdout)
                stderr = _decode_timeout_stream(e.stderr)
                raw_events_path.write_text(stdout, encoding="utf-8")
                stderr_path.write_text(stderr, encoding="utf-8")
                token_in, token_out = self._log_codex_events(stdout, log)
                log.set_status(status, status_reason)

            final_code_path = self._capture_final_code(scratch_dir, artifact_dir)
            if final_code_path is None and status == "ok":
                status = "agent_error"
                status_reason = "Codex completed without producing solver.py"
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

    def _command(self, *, scratch_dir: Path, last_message_path: Path) -> list[str]:
        cmd = [
            self.codex_bin,
            "exec",
            "--json",
            "--cd",
            str(scratch_dir),
            "--skip-git-repo-check",
            "--output-last-message",
            str(last_message_path),
            "-m",
            self.backbone.model_id,
        ]
        if self.ignore_user_config:
            cmd.append("--ignore-user-config")
        if self.ignore_rules:
            cmd.append("--ignore-rules")
        if self.ephemeral:
            cmd.append("--ephemeral")
        if self.disable_plugins:
            cmd.extend(["--disable", "plugins"])
        if self.reasoning_effort is not None:
            cmd.extend(
                ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
            )
        if self.bypass_approvals_and_sandbox:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", self.sandbox_mode])
        cmd.extend(self.extra_args)
        cmd.append("-")
        if not self.filesystem_hide_paths:
            return cmd
        if not self.mount_namespace_helper.is_file():
            raise FileNotFoundError(
                f"missing mount-namespace helper: {self.mount_namespace_helper}"
            )
        mask_root = self.process_cwd / ".nmi_hidden_mount_masks"
        mask_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        wrapped = [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--mount-proc",
            "python3",
            str(self.mount_namespace_helper),
            "--mask-root",
            str(mask_root),
        ]
        for path in self.filesystem_hide_paths:
            resolved = path.resolve()
            if not resolved.is_dir():
                raise FileNotFoundError(f"filesystem hide target is not a directory: {resolved}")
            wrapped.extend(["--hide", str(resolved)])
        return [*wrapped, "--", *cmd]

    @contextmanager
    def _codex_env(self) -> Iterator[dict[str, str]]:
        env = os.environ.copy()
        if not self.isolate_codex_home:
            yield env
            return

        source = self.codex_home_source
        if source is None:
            source = Path(env.get("CODEX_HOME") or (Path.home() / ".codex"))
        with tempfile.TemporaryDirectory(prefix="pdev_codex_home_") as tmp:
            runtime_home = Path(tmp)
            runtime_home.chmod(0o700)
            names = list(self.codex_home_copy_files)
            if not self.ignore_user_config:
                names.append("config.toml")
            for name in names:
                src = source / name
                if not src.is_file():
                    continue
                dst = runtime_home / name
                shutil.copy2(src, dst)
                dst.chmod(0o600)
            env["CODEX_HOME"] = str(runtime_home)
            yield env

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

    def _log_codex_events(self, stdout: str, log: TraceLogger) -> tuple[int, int]:
        token_in = 0
        token_out = 0
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            item = event.get("item") or {}
            item_type = item.get("type")
            item_id = str(item.get("id") or event_type or "codex_event")
            if event_type == "item.completed" and item_type in {"agent_message", "reasoning"}:
                role = "assistant" if item_type == "agent_message" else "thinking"
                log.turn(role=role, content=str(item.get("text") or ""))
            elif event_type == "item.started" and item_type in {"file_change", "command_execution"}:
                log.tool_call(call_id=item_id, tool=f"codex.{item_type}", arguments=item)
            elif event_type == "item.completed" and item_type in {"file_change", "command_execution"}:
                log.tool_result(
                    call_id=item_id,
                    result=json.dumps(item, ensure_ascii=False),
                    exit_code=0 if item.get("status") in {None, "completed"} else 1,
                )
            elif event_type == "turn.completed":
                usage = event.get("usage") or {}
                token_in += int(usage.get("input_tokens", 0) or 0)
                token_out += int(usage.get("output_tokens", 0) or 0)
        return token_in, token_out

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


def _decode_timeout_stream(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


_runner: AgentRunner = BatchCodexCliRunner()
del _runner


__all__ = ["BatchCodexCliRunner"]
