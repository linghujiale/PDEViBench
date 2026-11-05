"""Batched Claude Code agent runner backed by the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from agents.base import AgentBackbone, AgentResult, AgentRunner
from agents.trace_logger import TraceLogger

ROOT = Path(__file__).parent.parent

DEFAULT_TEMPLATE = ROOT / "agents" / "prompts" / "case_template.md"
DEFAULT_LIBRARY_GUIDES = ROOT / "agents" / "prompts" / "library_guides"

DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = ("Bash", "Read", "Write")


@dataclasses.dataclass
class BatchClaudeCodeRunner:
    """Headless Claude Code runner driven by the Claude Agent SDK.

    Each call to :meth:`run` spawns one SDK ``query()`` session scoped to
    the per-rep scratch directory and returns a single :class:`AgentResult`.

    Constructor parameters
    ----------------------
    backbone
        :class:`AgentBackbone` identifying the LLM. The default uses
        ``provider="anthropic", model_id="claude-opus-4-7"`` which matches
        the ``BatchClaudeCodeRunner.model`` SDK option below.
    name
        Canonical agent name (stamped into the verdict + trace).
    max_turns
        Hard cap on assistant turns per case. Reached → ``status="timeout"``.
        Default 30; tune up for harder cells.
    wall_timeout_seconds
        Wall-clock timeout per case (in real seconds; the runner uses
        ``asyncio.wait_for`` to enforce it). Default 600s (10 minutes).
    allowed_tools
        Tools the agent is permitted to use. Defaults to Bash/Read/Write.
    template_path / library_guide_root
        Where the prompt template + per-library guides live. Defaults point
        at ``agents/prompts/case_template.md`` and ``library_guides/``.
    extra_options
        Passthrough for ``ClaudeAgentOptions`` fields not exposed as
        first-class attributes (e.g. ``effort``, ``thinking``,
        ``permission_mode``). Merged into the options dict at call time.
    """

    backbone: AgentBackbone = dataclasses.field(
        default_factory=lambda: AgentBackbone(
            provider="anthropic",
            model_id="claude-opus-4-7",
            decoding={"temperature": 0.2},
        )
    )
    name: str = "claude_code_batch"
    max_turns: int = 30
    wall_timeout_seconds: float = 600.0
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    template_path: Path = dataclasses.field(default_factory=lambda: DEFAULT_TEMPLATE)
    library_guide_root: Path = dataclasses.field(default_factory=lambda: DEFAULT_LIBRARY_GUIDES)
    permission_mode: str = "bypassPermissions"
    extra_options: dict[str, Any] = dataclasses.field(default_factory=dict)

    def run(
        self,
        *,
        case: dict[str, Any],
        run_idx: int,
        artifact_dir: Path,
        scratch_dir: Path,
        rng_seed: int,
    ) -> AgentResult:
        """Drive one Claude Code SDK session for ``case``.

        Side effects:
        * Writes ``artifact_dir/prompt.md`` (rendered prompt).
        * Writes ``artifact_dir/trace.jsonl`` (turn-by-turn).
        * Writes ``artifact_dir/final_code.py`` (copy of solver.py from scratch).
        * Leaves the agent's working tree intact at ``scratch_dir/`` so the
          evaluator can find ``/scratch/solution.npz`` (created when the
          evaluator later runs solver.py).
        """
        artifact_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = self._render_and_persist_prompt(case, artifact_dir)
        trace_path = artifact_dir / "trace.jsonl"

        header = {
            "agent": self.name,
            "case_id": case["id"],
            "run_idx": run_idx,
            "rng_seed": rng_seed,
            "backbone": self.backbone.as_dict(),
            "max_turns": self.max_turns,
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "allowed_tools": list(self.allowed_tools),
        }

        wall_start = time.monotonic()
        status: str = "ok"
        status_reason: str = ""
        with TraceLogger(trace_path, header=header) as log:
            log.turn(role="user", content=prompt_text)
            try:
                summary = asyncio.run(
                    self._drive_session(
                        prompt_text=prompt_text,
                        scratch_dir=scratch_dir,
                        case_id=case["id"],
                        run_idx=run_idx,
                        rng_seed=rng_seed,
                        log=log,
                    )
                )
                status = summary["status"]
                status_reason = summary["status_reason"]
            except asyncio.TimeoutError:
                status = "timeout"
                status_reason = f"Exceeded wall_timeout_seconds={self.wall_timeout_seconds}"
                log.set_status(status, status_reason)
            except KeyboardInterrupt:
                status = "interrupted"
                status_reason = "KeyboardInterrupt"
                log.set_status(status, status_reason)
                raise
            except Exception as e:  # noqa: BLE001
                status = "agent_error"
                status_reason = f"{type(e).__name__}: {e}"
                log.set_status(status, status_reason)
            n_turns = log._n_turns
            n_tool_calls = log._n_tool_calls
            tok_in = log._token_in
            tok_out = log._token_out

            final_code_path = self._capture_final_code(scratch_dir, artifact_dir)
            if final_code_path is not None:
                log.set_final_code(final_code_path)

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
            token_total_in=tok_in,
            token_total_out=tok_out,
            n_turns=n_turns,
            n_tool_calls=n_tool_calls,
        )


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

    async def _drive_session(
        self,
        *,
        prompt_text: str,
        scratch_dir: Path,
        case_id: str,
        run_idx: int,
        rng_seed: int,
        log: TraceLogger,
    ) -> dict[str, Any]:
        """Run one SDK query and stream messages into the trace.

        Raises ``asyncio.TimeoutError`` on wall timeout. Caller catches.
        """
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            query,
            AssistantMessage,
            UserMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            ToolResultBlock,
            ThinkingBlock,
        )

        options_kwargs: dict[str, Any] = dict(
            allowed_tools=list(self.allowed_tools),
            cwd=str(scratch_dir),
            model=self.backbone.model_id,
            max_turns=self.max_turns,
            permission_mode=self.permission_mode,
            env={
                "PDEV_CASE_ID": case_id,
                "PDEV_RUN_IDX": str(run_idx),
                "PDEV_SEED": str(rng_seed),
            },
        )
        options_kwargs.update(self.extra_options)
        options = ClaudeAgentOptions(**options_kwargs)

        async def stream() -> dict[str, Any]:
            tok_in = 0
            tok_out = 0
            n_turns = 0
            n_tool_calls = 0
            status = "ok"
            status_reason = ""
            async for msg in query(prompt=prompt_text, options=options):
                if isinstance(msg, AssistantMessage):
                    text_parts: list[str] = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ThinkingBlock):
                            log.turn(role="thinking",
                                     content=getattr(block, "thinking", "") or "")
                        elif isinstance(block, ToolUseBlock):
                            log.tool_call(
                                call_id=block.id,
                                tool=block.name,
                                arguments=block.input,
                            )
                            n_tool_calls += 1
                    usage = msg.usage or {}
                    in_tokens = int(usage.get("input_tokens", 0) or 0)
                    out_tokens = int(usage.get("output_tokens", 0) or 0)
                    tok_in += in_tokens
                    tok_out += out_tokens
                    log.turn(
                        role="assistant",
                        content="\n".join(text_parts),
                        token_count_in=in_tokens,
                        token_count_out=out_tokens,
                    )
                    n_turns += 1
                    if msg.error:
                        status = "agent_error"
                        status_reason = f"AssistantMessage.error={msg.error}"
                elif isinstance(msg, UserMessage):
                    if isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, ToolResultBlock):
                                result_str = _normalize_tool_result(block.content)
                                log.tool_result(
                                    call_id=block.tool_use_id,
                                    result=result_str,
                                    exit_code=1 if block.is_error else 0,
                                )
                elif isinstance(msg, ResultMessage):
                    if msg.is_error:
                        status = "agent_error"
                        status_reason = (msg.stop_reason
                                         or "ResultMessage reports is_error=True")
                    break
            log.set_status(status, status_reason)
            return {
                "n_turns": n_turns,
                "n_tool_calls": n_tool_calls,
                "tok_in": tok_in,
                "tok_out": tok_out,
                "status": status,
                "status_reason": status_reason,
            }

        return await asyncio.wait_for(stream(), timeout=self.wall_timeout_seconds)

    def _capture_final_code(self, scratch_dir: Path, artifact_dir: Path) -> Path | None:
        """Copy the agent's solver.py from scratch into artifact_dir/final_code.py.

        Filename preference: ``solver.py`` at scratch root, then any *.py
        with non-trivial size. Returns the destination path on success.
        """
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
        src = candidates[0]
        dst = artifact_dir / "final_code.py"
        shutil.copyfile(src, dst)
        return dst


def _normalize_tool_result(content: Any) -> str:
    """Render a ToolResultBlock.content into a single string for the trace."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(repr(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return repr(content)


def _patch_footer_wall(trace_path: Path, wall_seconds: float) -> None:
    """Stamp the runner's wall_seconds_total onto the trace's footer.

    The TraceLogger writes the footer with wall_seconds_total=0.0 because
    the logger has no view of the runner's own wall clock. We rewrite the
    last line (the footer) with the correct value.
    """
    if not trace_path.exists():
        return
    import json as _json
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    try:
        footer = _json.loads(lines[-1])
    except _json.JSONDecodeError:
        return
    if footer.get("entry_type") != "footer":
        return
    footer["wall_seconds_total"] = float(wall_seconds)
    lines[-1] = _json.dumps(footer, ensure_ascii=False)
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_runner: AgentRunner = BatchClaudeCodeRunner()
del _runner


__all__ = ["BatchClaudeCodeRunner", "DEFAULT_ALLOWED_TOOLS"]
