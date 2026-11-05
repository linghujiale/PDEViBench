"""Run the 72-task panel with the DOLFINx finite-element backend."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts import run_phase2_wave1


def _has_option(argv: list[str], *names: str) -> bool:
    for arg in argv:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    defaults: list[str] = []
    if not _has_option(args, "--library"):
        defaults.extend(["--library", "DOLFINx"])
    if not _has_option(args, "--eval-mode"):
        defaults.extend(["--eval-mode", "docker"])
    if not _has_option(args, "--experiment-id"):
        exp_id = (
            "phase2_wave2_codex_dolfinx_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        defaults.extend(["--experiment-id", exp_id])
    return run_phase2_wave1.main(defaults + args)


if __name__ == "__main__":
    raise SystemExit(main())
