"""Claude Code agent runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_STRIP_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)


from agents.sdk_runner import BatchClaudeCodeRunner  # noqa: E402, F401


def render_prompt(case: dict[str, Any], *, template_path: Path,
                  library_guide_path: Path | None = None) -> str:
    """Materialise the prompt template against a case.

    Keeping it here avoids prompt drift between modes.

    Template uses ``str.format``-style braces. Dotted attribute access
    (e.g. ``{output_spec.dtype}``) resolves against a SimpleNamespace.
    Jinja-style ``{# ... #}`` comment blocks are stripped before
    formatting so the same template can also be consumed by a Jinja2
    pipeline.
    """
    from types import SimpleNamespace

    template = template_path.read_text(encoding="utf-8")
    library_guide = ""
    if library_guide_path and library_guide_path.exists():
        library_guide = library_guide_path.read_text(encoding="utf-8")

    template = _STRIP_JINJA_COMMENT.sub("", template)

    spec = case["case_spec"]
    output_spec = SimpleNamespace(
        dtype=spec["output"]["dtype"],
        shape=spec["output"]["shape"],
    )

    class _Defaulting(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Defaulting({
        "case_id": case.get("prompt_case_id", case["id"]),
        "library": (case.get("library_support") or ["DOLFINx"])[0],
        "pde_statement": spec["pde"]["statement"],
        "case_spec_json": json.dumps(spec, indent=2, sort_keys=True),
        "operator_family": spec["pde"].get("operator_family", "unknown"),
        "parameters_block": _bullet_block(spec["pde"].get("parameters", {})),
        "domain_block": _json_block(spec.get("domain", {})),
        "bc_block": _json_block(spec.get("bc", {})),
        "ic_block": _json_block(spec.get("ic", {})) if spec.get("ic") else "_(steady problem; no initial conditions)_",
        "eval_grid_block": _json_block(spec.get("eval_grid", {})),
        "output_spec": output_spec,
        "library_guide": library_guide,
        "seed": case["provenance"].get("generator_seed", 0),
    }))


def _bullet_block(d: dict[str, Any]) -> str:
    if not d:
        return "_(no parameters)_"
    return "\n".join(f"* `{k}` = `{v}`" for k, v in d.items())


def _json_block(d: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(d, indent=2) + "\n```"


__all__ = [
    "BatchClaudeCodeRunner",
    "INTERACTIVE_DESCRIPTION",
    "render_prompt",
]
