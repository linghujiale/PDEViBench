"""Uniform trace capture."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TraceLogger:
    """Append-only JSONL writer for an agent trace."""

    def __init__(self, path: Path, *, header: dict[str, Any]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")
        self._n_turns = 0
        self._n_tool_calls = 0
        self._token_in = 0
        self._token_out = 0
        self._started = _utcnow_iso()
        self._status = "ok"
        self._status_reason = ""

        header_entry = {"entry_type": "header", "schema_version": "1.0", "started_at": self._started, **header}
        self._emit(header_entry)


    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is not None:
            self.set_status("agent_error", f"{exc_type.__name__ if exc_type else 'Exception'}: {exc}")
        self._emit_footer()
        self._fh.close()
        return False


    def turn(
        self,
        *,
        role: str,
        content: str,
        token_count_in: int = 0,
        token_count_out: int = 0,
        wall_ms: int = 0,
    ) -> None:
        self._emit({
            "entry_type": "turn",
            "turn_index": self._n_turns,
            "role": role,
            "content": content,
            "timestamp": _utcnow_iso(),
            "token_count_in": token_count_in,
            "token_count_out": token_count_out,
            "wall_ms": wall_ms,
        })
        self._n_turns += 1
        self._token_in += token_count_in
        self._token_out += token_count_out

    def tool_call(self, *, call_id: str, tool: str, arguments: dict[str, Any]) -> None:
        self._emit({
            "entry_type": "tool_call",
            "turn_index": self._n_turns,
            "call_id": call_id,
            "tool": tool,
            "arguments": arguments,
            "timestamp": _utcnow_iso(),
        })
        self._n_tool_calls += 1

    def tool_result(
        self,
        *,
        call_id: str,
        result: str,
        exit_code: int = 0,
        truncate_at: int = 64 * 1024,
    ) -> None:
        truncated = len(result) > truncate_at
        payload = result[:truncate_at] if truncated else result
        self._emit({
            "entry_type": "tool_result",
            "call_id": call_id,
            "result": payload,
            "truncated": truncated,
            "exit_code": exit_code,
            "timestamp": _utcnow_iso(),
        })

    def set_status(self, status: str, reason: str = "") -> None:
        self._status = status
        self._status_reason = reason


    def _emit(self, entry: dict[str, Any]) -> None:
        self._fh.write(json.dumps(entry, ensure_ascii=False))
        self._fh.write("\n")
        self._fh.flush()

    def _emit_footer(self) -> None:
        footer = {
            "entry_type": "footer",
            "completed_at": _utcnow_iso(),
            "status": self._status,
            "status_reason": self._status_reason,
            "final_code_path": getattr(self, "_final_code_path", ""),
            "final_code_sha256": getattr(self, "_final_code_sha256", "0" * 64),
            "wall_seconds_total": 0.0,
            "token_total_in": self._token_in,
            "token_total_out": self._token_out,
            "n_turns": self._n_turns,
            "n_tool_calls": self._n_tool_calls,
        }
        self._emit(footer)

    def set_final_code(self, path: Path) -> None:
        self._final_code_path = str(path)
        self._final_code_sha256 = _sha256(path.read_text(encoding="utf-8"))


__all__ = ["TraceLogger"]
