#!/usr/bin/env python3
"""Run the four-call experiment on the Helmholtz task with the corrected operator line."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "analysis" / "matched_visibility_corrected_case_20260825"
CASES = DESIGN / "scheduled_observations.jsonl"
RESULTS = DESIGN / "results_batch"
EXPERIMENT_ID = "matched_visibility_corrected_case_codex_gpt56sol_scipy_20260825"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if not (DESIGN / "design_manifest.json").is_file() or not CASES.is_file():
        raise SystemExit("fix the corrected-case design before launching calls")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_phase2_wave1.py"),
        "--cases",
        str(CASES),
        "--experiment-id",
        EXPERIMENT_ID,
        "--n-runs",
        "1",
        "--runner",
        "codex",
        "--agent-name",
        "codex_cli_batch",
        "--backbone-provider",
        "openai",
        "--backbone-model",
        "gpt-5.6-sol",
        "--reasoning-effort",
        "high",
        "--library",
        "scipy",
        "--eval-mode",
        "host",
        "--parallel-workers",
        "1",
        "--wall-timeout",
        "600",
        "--max-turns",
        "30",
        "--results-root",
        str(RESULTS),
        "--analysis-root",
        str(ROOT / "analysis" / "phase2"),
        "--halt-on-external-limit",
    ]
    if args.resume:
        command.extend(["--resume", "--rerun-infra-failures"])
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        return completed.returncode
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_matched_visibility_corrected_case.py")],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
