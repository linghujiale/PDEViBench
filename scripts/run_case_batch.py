"""Drive a single case via a batch agent runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.base import AgentBackbone  # noqa: E402
from evaluator import host_eval, sandbox, staged  # noqa: E402


def _load_case(cases_path: Path, case_id: str) -> dict:
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c["id"] == case_id:
            return c
    raise KeyError(f"case {case_id!r} not found in {cases_path}")


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "0" * 40


def _display_path(path: Path) -> str:
    """Prefer a repo-relative path, but support external result volumes."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-id", required=True)
    p.add_argument("--cases", type=Path,
                   default=ROOT / "benchmark" / "cases" / "pilot_phase1_6.jsonl")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--runner", choices=("claude", "claude_cli", "codex", "openrouter"), default="claude",
                   help="Agent backend to invoke. Default preserves legacy Claude path.")
    p.add_argument("--agent-name", default=None)
    p.add_argument("--backbone-provider", default=None)
    p.add_argument("--backbone-model", default=None)
    p.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Explicit Codex model reasoning effort; recorded in trace provenance.",
    )
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--wall-timeout", type=float, default=600.0,
                   help="Wall-clock seconds before the SDK call is cancelled.")
    p.add_argument("--run-idx", type=int, default=0)
    p.add_argument("--rng-seed", type=int, default=None,
                   help="Overrides the case's generator_seed if given.")
    p.add_argument("--results-root", type=Path,
                   default=ROOT / "results" / "batch")
    p.add_argument("--skip-eval", action="store_true",
                   help="Stop after the SDK call; don't run host_eval.")
    p.add_argument("--eval-mode", choices=("host", "docker"), default="host",
                   help="host = evaluator.host_eval fallback; docker = evaluator.staged via Docker.")
    p.add_argument("--container-image", default=None,
                   help="Override Docker image when --eval-mode=docker.")
    p.add_argument("--library", default=None,
                   help=("Override the library the agent is told to use. "
                         "Default uses the case's library_support[0] "
                         "(typically 'DOLFINx'). Pass 'scipy' for host-mode "
                         "smoke runs where DOLFINx isn't installed."))
    p.add_argument(
        "--library-guide-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing <library>.md prompt guides. This "
            "changes only the injected guide while preserving the --library "
            "label, which is required for controlled prompt ablations."
        ),
    )
    args = p.parse_args(argv)

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

    if (
        args.runner == "claude"
        and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"))
    ):
        print("NOTE: no ANTHROPIC_API_KEY in env — SDK will fall back to Claude CLI auth.",
              file=sys.stderr)

    case = _load_case(args.cases, args.case_id)
    rng_seed = (args.rng_seed if args.rng_seed is not None
                else int(case["provenance"].get("generator_seed", 0)))

    if args.library is not None:
        case = dict(case)
        case["library_support"] = [args.library]

    guide_kwargs = (
        {"library_guide_root": args.library_guide_root.resolve()}
        if args.library_guide_root is not None
        else {}
    )

    results_root = (
        args.results_root
        if args.results_root.is_absolute()
        else ROOT / args.results_root
    )
    artifact_dir = (
        results_root / args.experiment_id / case["id"]
        / agent_name / f"{backbone_provider}-{backbone_model}"
        / f"run{args.run_idx}"
    )
    scratch_dir = Path(tempfile.gettempdir()) / (
        f"pdev_scratch_{args.experiment_id}_{case['id']}_run{args.run_idx}"
    )

    if args.runner == "codex":
        from agents.codex_runner import BatchCodexCliRunner

        runner = BatchCodexCliRunner(
            backbone=AgentBackbone(
                provider=backbone_provider,
                model_id=backbone_model,
                decoding={"temperature": 0.2},
            ),
            name=agent_name,
            max_turns=args.max_turns,
            wall_timeout_seconds=args.wall_timeout,
            reasoning_effort=args.reasoning_effort,
            **guide_kwargs,
        )
    elif args.runner == "claude_cli":
        from agents.claude_cli_runner import BatchClaudeCodeCliRunner

        runner = BatchClaudeCodeCliRunner(
            backbone=AgentBackbone(
                provider=backbone_provider,
                model_id=backbone_model,
                decoding={"temperature": 0.2},
            ),
            name=agent_name,
            max_turns=args.max_turns,
            wall_timeout_seconds=args.wall_timeout,
            **guide_kwargs,
        )
    elif args.runner == "openrouter":
        from agents.openrouter_runner import BatchOpenRouterRunner

        runner = BatchOpenRouterRunner(
            backbone=AgentBackbone(
                provider=backbone_provider,
                model_id=backbone_model,
                decoding={"temperature": 0.2},
            ),
            name=agent_name,
            max_turns=args.max_turns,
            wall_timeout_seconds=args.wall_timeout,
            **guide_kwargs,
        )
    else:
        from agents.sdk_runner import BatchClaudeCodeRunner

        runner = BatchClaudeCodeRunner(
            backbone=AgentBackbone(
                provider=backbone_provider,
                model_id=backbone_model,
                decoding={"temperature": 0.2},
            ),
            name=agent_name,
            max_turns=args.max_turns,
            wall_timeout_seconds=args.wall_timeout,
            **guide_kwargs,
        )

    agent_result = runner.run(
        case=case,
        run_idx=args.run_idx,
        artifact_dir=artifact_dir,
        scratch_dir=scratch_dir,
        rng_seed=rng_seed,
    )

    print(json.dumps({
        "phase": "agent_call",
        "case_id": agent_result.case_id,
        "status": agent_result.status,
        "status_reason": agent_result.status_reason,
        "wall_seconds_total": agent_result.wall_seconds_total,
        "n_turns": agent_result.n_turns,
        "n_tool_calls": agent_result.n_tool_calls,
        "token_total_in": agent_result.token_total_in,
        "token_total_out": agent_result.token_total_out,
        "artifact_dir": _display_path(artifact_dir),
        "runner": args.runner,
        "backbone": agent_result.backbone.as_dict(),
    }, indent=2))

    if args.skip_eval:
        return 0 if agent_result.status == "ok" else 1
    if agent_result.status != "ok":
        return 1

    final_code_path = artifact_dir / "final_code.py"
    if not final_code_path.exists():
        print("ERROR: agent did not produce a solver — final_code.py missing.",
              file=sys.stderr)
        return 1

    ref_path = ROOT / case["reference"]["path"]
    with np.load(ref_path, allow_pickle=False) as data:
        reference = np.asarray(data[list(data.keys())[0]])

    if args.eval_mode == "docker":
        eval_scratch = (
            ROOT
            / "analysis"
            / "tmp"
            / "docker_eval"
            / f"{args.experiment_id}_{case['id']}_run{args.run_idx}"
        )
        if eval_scratch.exists():
            shutil.rmtree(eval_scratch)
        eval_scratch.mkdir(parents=True, exist_ok=True)
        library_for_image = (case.get("library_support") or [args.library or "DOLFINx"])[0]
        image = args.container_image or sandbox.select_image(library_for_image)
        verdict = staged.run(
            staged.StagedRunContext(
                case=case,
                case_id=case["id"],
                agent=agent_name,
                backbone=agent_result.backbone.as_dict(),
                run_idx=args.run_idx,
                experiment_id=args.experiment_id,
                git_commit=_git_head(),
                artifact_dir=artifact_dir,
                scratch_dir=eval_scratch,
                image=image,
                final_code_path=final_code_path,
                solver_command=["/artifact/final_code.py"],
                reference=reference,
                energy_matrix=None,
                rng_seed=rng_seed,
            )
        )
    else:
        verdict = host_eval.run(
            case=case,
            solver_path=final_code_path,
            artifact_dir=artifact_dir,
            scratch_dir=scratch_dir,
            reference_array=reference,
            run_idx=args.run_idx,
            experiment_id=args.experiment_id,
            git_commit=_git_head(),
            agent=agent_name,
            backbone=agent_result.backbone.as_dict(),
            solver_info=f"{type(runner).__name__} @ max_turns={args.max_turns}, "
                        f"wall_timeout={args.wall_timeout}s",
        )

    print(json.dumps({
        "phase": "verdict",
        "case_id": case["id"],
        "case_pass": verdict["case_pass"],
        "failure_code": verdict["failure_code"],
        "rel_l2": verdict["stages"]["accuracy"].get("rel_l2"),
        "wall_seconds_mean": verdict["stages"]["runtime"].get("wall_seconds_mean"),
        "method_failed_rule_ids": _method_failed_rule_ids(verdict),
        "near_zero_exact_formula_shortcut": _near_zero_exact_formula_shortcut(verdict),
        "artifact_dir": _display_path(artifact_dir),
    }, indent=2))
    return 0 if verdict["case_pass"] else 1


def _method_failed_rule_ids(verdict: dict) -> list[str]:
    checks = verdict.get("stages", {}).get("method", {}).get("checks", [])
    return [
        str(check.get("rule_id"))
        for check in checks
        if check.get("passed") is False and check.get("rule_id") is not None
    ]


def _near_zero_exact_formula_shortcut(verdict: dict) -> dict | None:
    checks = verdict.get("stages", {}).get("method", {}).get("checks", [])
    for check in checks:
        if (
            check.get("rule_id") == "R-NO-EXACT-FORMULA"
            and check.get("passed") is False
        ):
            evidence = str(check.get("evidence", ""))
            match = re.search(r"classification=([A-Z_]+)", evidence)
            return {
                "label": match.group(1) if match else None,
                "evidence": evidence,
            }
    return None


if __name__ == "__main__":
    raise SystemExit(main())
