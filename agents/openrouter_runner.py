"""Batch OpenRouter runner with a minimal tool loop (bash / read / write)."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from agents.base import AgentBackbone, AgentResult, AgentRunner
from agents.sdk_runner import _patch_footer_wall
from agents.trace_logger import TraceLogger

ROOT = Path(__file__).parent.parent

DEFAULT_TEMPLATE = ROOT / "agents" / "prompts" / "case_template.md"
DEFAULT_LIBRARY_GUIDES = ROOT / "agents" / "prompts" / "library_guides"

SYSTEM_PROMPT = (
    "You are an expert numerical-methods engineer solving a PDE coding task.\n"
    "You have three tools: bash, read, and write. All paths are relative to "
    "your scratch directory, which is your current working directory.\n"
    "Workflow: use write to create solver.py, optionally use bash to test it "
    "(e.g. `python3 solver.py`), and stop emitting tool calls when solver.py "
    "is final. The grading harness will execute solver.py and evaluate its "
    "solution.npz output."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the scratch directory. Use to inspect files, "
                "run intermediate computations, or test your solver. Returns "
                "combined stdout+stderr truncated to 50000 chars and the exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a text file from the scratch directory. Returns up to 50000 chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to scratch."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "Write a text file to the scratch directory (overwrites if exists). "
                "Use to emit your final solver as solver.py."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to scratch."},
                    "content": {"type": "string", "description": "Full file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
]


@dataclasses.dataclass
class BatchOpenRouterRunner:
    """Headless OpenRouter chat-completion runner with a bash/read/write tool loop."""

    backbone: AgentBackbone = dataclasses.field(
        default_factory=lambda: AgentBackbone(
            provider="openrouter",
            model_id="deepseek/deepseek-v3.2",
            decoding={"temperature": 0.2},
        )
    )
    name: str = "openrouter_batch"
    max_turns: int = 60
    wall_timeout_seconds: float = 600.0
    request_timeout_seconds: float = 180.0
    max_completion_tokens: int = 16_000
    template_path: Path = dataclasses.field(default_factory=lambda: DEFAULT_TEMPLATE)
    library_guide_root: Path = dataclasses.field(default_factory=lambda: DEFAULT_LIBRARY_GUIDES)
    api_base: str = "https://openrouter.ai/api/v1"
    tool_bash_timeout_seconds: float = 60.0
    tool_output_max_chars: int = 50_000
    http_referer: str = "https://github.com/pdevibench-nmi"
    http_title: str = "PDEViBench-NMI"

    def run(
        self,
        *,
        case: dict[str, Any],
        run_idx: int,
        artifact_dir: Path,
        scratch_dir: Path,
        rng_seed: int,
    ) -> AgentResult:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = self._render_and_persist_prompt(case, artifact_dir)
        trace_path = artifact_dir / "trace.jsonl"
        raw_log_path = artifact_dir / "openrouter_events.jsonl"

        header = {
            "agent": self.name,
            "case_id": case["id"],
            "run_idx": run_idx,
            "rng_seed": rng_seed,
            "backbone": self.backbone.as_dict(),
            "max_turns": self.max_turns,
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "api_base": self.api_base,
            "tools": ["bash", "read", "write"],
        }

        wall_start = time.monotonic()
        status = "ok"
        status_reason = ""
        token_in = 0
        token_out = 0
        n_turns_total = 0
        n_tool_calls_total = 0

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        raw_log_fh = raw_log_path.open("w", encoding="utf-8")
        try:
            with TraceLogger(trace_path, header=header) as log:
                log.turn(role="system", content=SYSTEM_PROMPT)
                log.turn(role="user", content=prompt_text)

                for turn_idx in range(self.max_turns):
                    elapsed = time.monotonic() - wall_start
                    if elapsed >= self.wall_timeout_seconds:
                        status = "timeout"
                        status_reason = (
                            f"Exceeded wall_timeout_seconds={self.wall_timeout_seconds} "
                            f"after {turn_idx} model turns"
                        )
                        log.set_status(status, status_reason)
                        break

                    response, err_status, err_reason = self._chat_request(
                        api_key=api_key,
                        messages=messages,
                        turn_idx=turn_idx,
                        remaining_seconds=max(1.0, self.wall_timeout_seconds - elapsed),
                    )
                    if response is None:
                        status = err_status
                        status_reason = err_reason
                        log.set_status(status, status_reason)
                        break

                    raw_log_fh.write(json.dumps(response) + "\n")
                    raw_log_fh.flush()

                    usage = response.get("usage") or {}
                    token_in += int(usage.get("prompt_tokens", 0) or 0)
                    token_out += int(usage.get("completion_tokens", 0) or 0)

                    choices = response.get("choices") or []
                    if not choices:
                        status = "agent_error"
                        status_reason = f"empty choices at turn {turn_idx}"
                        log.set_status(status, status_reason)
                        break

                    msg = choices[0].get("message") or {}
                    text = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls") or []

                    if text:
                        log.turn(role="assistant", content=text)

                    if not tool_calls:
                        messages.append({"role": "assistant", "content": text})
                        break

                    messages.append(
                        {
                            "role": "assistant",
                            "content": text or "",
                            "tool_calls": tool_calls,
                        }
                    )

                    for tc in tool_calls:
                        call_id = tc.get("id") or f"tc_{turn_idx}_{n_tool_calls_total}"
                        fn = tc.get("function") or {}
                        tool_name = fn.get("name") or "unknown"
                        raw_args = fn.get("arguments") or "{}"
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {"_parse_error": str(raw_args)[:1000]}

                        log.tool_call(call_id=call_id, tool=tool_name, arguments=args)
                        result_text, exit_code = self._exec_tool(tool_name, args, scratch_dir)
                        log.tool_result(call_id=call_id, result=result_text, exit_code=exit_code)

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": result_text[: self.tool_output_max_chars],
                            }
                        )
                else:
                    status = "max_turns_with_solver"
                    status_reason = (
                        f"max_turns={self.max_turns} reached without final "
                        f"assistant message; will downgrade to agent_error "
                        f"only if no substantive solver.py exists"
                    )

                final_code_path = self._capture_final_code(scratch_dir, artifact_dir)
                if final_code_path is None:
                    if status in ("ok", "max_turns_with_solver"):
                        status = "agent_error"
                        status_reason = (
                            "Agent completed without producing solver.py in scratch"
                            if status_reason == "" or "max_turns" not in status_reason
                            else status_reason + "; and no solver.py was found"
                        )
                        log.set_status(status, status_reason)
                else:
                    log.set_final_code(final_code_path)
                    if status == "max_turns_with_solver":
                        status = "ok"
                        status_reason = f"max_turns={self.max_turns} reached but substantive solver.py was produced"
                        log.set_status(status, status_reason)

                log._token_in = token_in
                log._token_out = token_out
                n_turns_total = log._n_turns
                n_tool_calls_total = log._n_tool_calls
        finally:
            raw_log_fh.close()

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
            n_turns=n_turns_total,
            n_tool_calls=n_tool_calls_total,
        )

    def _chat_request(
        self,
        *,
        api_key: str,
        messages: list[dict[str, Any]],
        turn_idx: int,
        remaining_seconds: float,
    ) -> tuple[dict[str, Any] | None, str, str]:
        """POST to OpenRouter with exponential-backoff retry on transient errors.

        Retriable: requests.RequestException (connection pool exhaustion,
        SSL handshake blip, read timeout), HTTP 5xx, HTTP 429.
        Non-retriable: HTTP 4xx other than 429, JSON decode error.

        Total attempts: 4 (initial + 3 retries) with sleeps 5 / 15 / 45 seconds.
        Aborts retry loop if the cumulative sleep would exceed `remaining_seconds`.
        """
        import time as _time
        max_completion_tokens = int(
            os.environ.get("OPENROUTER_MAX_TOKENS", str(self.max_completion_tokens))
        )
        payload = {
            "model": self.backbone.model_id,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": self.backbone.decoding.get("temperature", 0.2),
            "max_tokens": max_completion_tokens,
        }
        provider_order = os.environ.get("OPENROUTER_PROVIDER_ORDER")
        if provider_order:
            payload["provider"] = {
                "order": [p.strip() for p in provider_order.split(",") if p.strip()],
                "allow_fallbacks": True,
                "require_parameters": True,
            }
        timeout = min(self.request_timeout_seconds, max(5.0, remaining_seconds))
        backoff_schedule = (5.0, 15.0, 45.0)
        last_reason = "no attempts"
        for attempt_idx in range(1 + len(backoff_schedule)):
            try:
                resp = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": self.http_referer,
                        "X-Title": self.http_title,
                    },
                    timeout=timeout,
                )
            except requests.exceptions.Timeout:
                last_reason = f"request timeout at turn {turn_idx} (attempt {attempt_idx + 1})"
                resp = None
            except requests.exceptions.RequestException as e:
                last_reason = f"request error at turn {turn_idx} (attempt {attempt_idx + 1}): {e}"
                resp = None
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json(), "ok", ""
                    except ValueError as e:
                        return None, "agent_error", f"non-JSON response at turn {turn_idx}: {e}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    return None, "agent_error", (
                        f"HTTP {resp.status_code} at turn {turn_idx}: {resp.text[:500]}"
                    )
                last_reason = (
                    f"HTTP {resp.status_code} at turn {turn_idx} (attempt {attempt_idx + 1}): "
                    f"{resp.text[:300]}"
                )
            if attempt_idx >= len(backoff_schedule):
                break
            sleep_s = backoff_schedule[attempt_idx]
            if sleep_s + 5.0 > remaining_seconds:
                break
            _time.sleep(sleep_s)
        return None, "agent_error", last_reason

    def _exec_tool(
        self, name: str, args: dict[str, Any], scratch_dir: Path
    ) -> tuple[str, int]:
        try:
            if name == "bash":
                cmd = args.get("command", "")
                if not isinstance(cmd, str) or not cmd:
                    return "ERROR: bash requires a non-empty 'command' string", 1
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(scratch_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.tool_bash_timeout_seconds,
                )
                combined = (proc.stdout or "") + (proc.stderr or "")
                return combined[: self.tool_output_max_chars], int(proc.returncode)
            if name == "read":
                rel = args.get("path", "")
                path = (scratch_dir / rel).resolve()
                if not str(path).startswith(str(scratch_dir.resolve())):
                    return "ERROR: path outside scratch directory", 1
                if not path.exists():
                    return f"ERROR: file does not exist: {rel}", 1
                return path.read_text(errors="replace")[: self.tool_output_max_chars], 0
            if name == "write":
                rel = args.get("path", "")
                content = args.get("content", "")
                if not isinstance(content, str):
                    return "ERROR: write 'content' must be a string", 1
                path = (scratch_dir / rel).resolve()
                if not str(path).startswith(str(scratch_dir.resolve())):
                    return "ERROR: path outside scratch directory", 1
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                return f"wrote {rel} ({len(content)} bytes)", 0
            return f"unknown tool: {name}", 1
        except subprocess.TimeoutExpired:
            return f"tool {name} timed out after {self.tool_bash_timeout_seconds}s", 1
        except Exception as e:
            return f"{type(e).__name__}: {e}", 1

    def _render_and_persist_prompt(
        self, case: dict[str, Any], artifact_dir: Path
    ) -> str:
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

    def _capture_final_code(
        self, scratch_dir: Path, artifact_dir: Path
    ) -> Path | None:
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


_runner: AgentRunner = BatchOpenRouterRunner()
del _runner


__all__ = ["BatchOpenRouterRunner"]
