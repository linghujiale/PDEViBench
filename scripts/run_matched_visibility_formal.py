#!/usr/bin/env python3
"""Run the analytic-field disclosure experiment (256 calls)."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "analysis" / "matched_visibility_20260825"
CASES = DESIGN / "scheduled_observations.jsonl"
RESULTS = DESIGN / "results_batch"
EXPERIMENT_ID = "matched_visibility_codex_gpt56sol_scipy_20260825"
ANALYSIS = ROOT / "analysis" / "phase2" / EXPERIMENT_ID
RUNS = ANALYSIS / "runs.jsonl"
LOG = DESIGN / "formal_driver.log"
LOCK = DESIGN / "formal_driver.lock"


def stamp(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_logged(command: list[str], *, redact_scientific: bool) -> int:
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    with LOG.open("a", encoding="utf-8") as fh:
        for line in proc.stdout:
            fh.write(line)
            if redact_scientific and any(
                token in line
                for token in (
                    "case_pass=",
                    '"n_case_pass"',
                    '"n_case_fail"',
                    '"failure_code_counts"',
                    '"method_failed_rule_counts"',
                    '"max_rel_l2_cv"',
                    '"median_rel_l2_cv"',
                )
            ):
                print("[scientific outcome withheld until completion]", flush=True)
            else:
                print(line, end="", flush=True)
    return proc.wait()


def batch_command(resume: bool) -> list[str]:
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
        "--usage-limit-cooldown",
        "300",
        "--capacity-cooldown",
        "60",
    ]
    if resume:
        command.extend(["--resume", "--rerun-infra-failures"])
    return command


def status_command() -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "analyze_matched_visibility.py"),
        "--design-dir",
        str(DESIGN),
        "--analysis-dir",
        str(ANALYSIS),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-wait", type=float, default=300.0)
    parser.add_argument("--max-cycles", type=int, default=100)
    args = parser.parse_args(argv)
    if args.retry_wait < 0 or args.max_cycles <= 0:
        parser.error("retry wait must be non-negative and max cycles positive")

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another formal visibility driver already holds the lock", file=sys.stderr)
        return 2

    stamp("formal driver acquired exclusive lock")
    verify = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_matched_visibility_design.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(verify.stdout)
        fh.write(verify.stderr)
    if verify.returncode != 0:
        stamp("design verification failed; formal calls not launched")
        return verify.returncode
    stamp("design verification PASS; launching schedule")

    for cycle in range(1, args.max_cycles + 1):
        resume = RUNS.exists()
        stamp(f"batch cycle {cycle}; resume={resume}")
        batch_rc = run_logged(batch_command(resume), redact_scientific=True)
        stamp(f"batch cycle {cycle} exited rc={batch_rc}; checking completion without interim outcomes")
        status_rc = run_logged(status_command(), redact_scientific=False)
        status_path = ANALYSIS / "matched_visibility_status.json"
        if status_rc == 0:
            stamp("all formal observations finalized; blinded packet and provisional paired analysis written")
            return 0
        if not status_path.is_file():
            stamp("status file missing; stopping for operator diagnosis")
            return 4
        status = json.loads(status_path.read_text(encoding="utf-8"))
        remaining = int(status.get("remaining_retriable_or_unlaunched", 256))
        stamp(f"formal observations still open={remaining}; retrying unchanged keys after {args.retry_wait:.0f}s")
        time.sleep(args.retry_wait)

    stamp("maximum recovery cycles reached before completion")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
