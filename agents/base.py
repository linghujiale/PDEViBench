"""AgentRunner protocol."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Protocol


@dataclasses.dataclass(frozen=True)
class AgentBackbone:
    """Identifies the LLM behind the agent."""

    provider: str
    model_id: str
    model_snapshot: str | None = None
    decoding: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"provider": self.provider, "model_id": self.model_id}
        if self.model_snapshot:
            d["model_snapshot"] = self.model_snapshot
        if self.decoding:
            d["decoding"] = self.decoding
        return d


@dataclasses.dataclass(frozen=True)
class AgentResult:
    """What every runner returns to the orchestrator."""

    case_id: str
    agent: str
    backbone: AgentBackbone
    run_idx: int
    final_code_path: Path
    trace_path: Path
    prompt_path: Path
    status: str
    status_reason: str
    wall_seconds_total: float
    token_total_in: int
    token_total_out: int
    n_turns: int
    n_tool_calls: int


class AgentRunner(Protocol):
    """Protocol every concrete runner satisfies."""

    name: str
    backbone: AgentBackbone

    def run(
        self,
        *,
        case: dict[str, Any],
        run_idx: int,
        artifact_dir: Path,
        scratch_dir: Path,
        rng_seed: int,
    ) -> AgentResult:
        """Solve ``case`` once and return an ``AgentResult``.

        Implementations MUST:
        * Load and instantiate ``agents/prompts/case_template.md``.
        * Capture every turn via ``agents.trace_logger.TraceLogger``.
        * Write ``artifact_dir / "final_code.py"`` (or ``.cc`` for deal.II cases).
        * Write ``artifact_dir / "prompt.md"`` with the full instantiated prompt.
        * Never retry silently on agent error.
        """
        ...
