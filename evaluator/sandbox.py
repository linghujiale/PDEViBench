"""Container-isolated subprocess runner."""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS: int = 300
DEFAULT_MEMORY_LIMIT: str = "8g"

DEFAULT_IMAGE_BY_LIBRARY: Mapping[str, str] = {
    "DOLFINx": "pdevibench/dolfinx:0.9.0",
    "Firedrake": "pdevibench/firedrake:2024.10",
    "deal.II": "pdevibench/dealii:9.6.0",
    "evaluator": "pdevibench/evaluator:0.1",
}


@dataclasses.dataclass(frozen=True)
class SandboxResult:
    """The outcome of a single sandboxed subprocess invocation."""

    exit_code: int
    wall_seconds: float
    stdout_path: Path
    stderr_path: Path
    timed_out: bool
    oom_killed: bool
    container_image: str
    container_image_digest: str | None
    cmd: list[str]

    def as_meta_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "wall_seconds": self.wall_seconds,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "timed_out": self.timed_out,
            "oom_killed": self.oom_killed,
            "container_image": self.container_image,
            "container_image_digest": self.container_image_digest,
            "cmd": self.cmd,
        }


def select_image(library: str | None) -> str:
    """Choose the container image for a given library track. Falls back to the
    evaluator image when ``library`` is None or unknown.
    """
    if library is None:
        return DEFAULT_IMAGE_BY_LIBRARY["evaluator"]
    return DEFAULT_IMAGE_BY_LIBRARY.get(library, DEFAULT_IMAGE_BY_LIBRARY["evaluator"])


def resolve_image_digest(image: str) -> str | None:
    """Resolve the immutable digest of ``image`` so the artifact records the
    exact bits used, not just the tag.
    """
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image],
            capture_output=True,
            check=False,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except FileNotFoundError:
        return None
    return None


def run(
    *,
    image: str,
    command: Sequence[str],
    scratch_dir: Path,
    artifact_dir: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    cpus: str = "2",
    extra_mounts: Sequence[tuple[Path, str, str]] = (),
) -> SandboxResult:
    """Run ``command`` inside ``image`` with PDEViBench's sandbox conventions.

    Parameters
    ----------
    image
        Container image, e.g. ``"pdevibench/dolfinx:0.9.0"``.
    command
        The command to execute *inside* the container. Argv-style, no shell.
    scratch_dir
        Host directory mounted at ``/scratch`` inside the container; the agent
        and the solver may read/write here freely. Created fresh per case.
    artifact_dir
        Host directory where ``stdout.log`` and ``stderr.log`` are captured.
    env
        Environment variables to pass into the container (e.g. ``PDEV_SEED``).
    timeout_seconds
        Wall-clock cap. Hard 300 s.1; the caller must justify any
    memory_limit
        Docker memory string (e.g. ``"8g"``). Default 8 GB.1.
    cpus
        Docker ``--cpus`` argument. Defaulted to "2" so the same node can host
        multiple parallel sandboxes without thrashing.
    extra_mounts
        Iterable of ``(host_path, container_path, mode)`` tuples for read-only
        case inputs (e.g. ``cases.jsonl``, reference NPZ files).

    Returns
    -------
    SandboxResult
        Captures everything the evaluator needs to short-circuit subsequent
        stages: exit code, wall time, log paths, OOM/timeout flags.
    """
    if not scratch_dir.is_dir():
        raise FileNotFoundError(f"scratch_dir does not exist: {scratch_dir}")
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact_dir does not exist: {artifact_dir}")

    stdout_path = artifact_dir / "stdout.log"
    stderr_path = artifact_dir / "stderr.log"
    cache_dirs = [
        scratch_dir,
        scratch_dir / ".cache",
        scratch_dir / ".ffcx-cache",
        scratch_dir / ".home_cache",
    ]
    for cache_dir in cache_dirs:
        cache_dir.mkdir(exist_ok=True)
        cache_dir.chmod(0o777)
    home_cache_dir = scratch_dir / ".home_cache"

    docker_cmd: list[str] = [
        "docker", "run",
        "--rm",
        "--network", "none",
        "--memory", memory_limit,
        "--memory-swap", memory_limit,
        "--cpus", cpus,
        "--pids-limit", "1024",
        "--read-only",
        "--tmpfs", "/tmp:size=512m",
        "-v", f"{scratch_dir}:/scratch",
        "-v", f"{home_cache_dir}:/home/runner/.cache",
        "-v", f"{artifact_dir}:/artifact",
        "-w", "/scratch",
    ]
    for host, container, mode in extra_mounts:
        docker_cmd += ["-v", f"{host}:{container}:{mode}"]

    cache_env = {
        "XDG_CACHE_HOME": "/scratch/.cache",
        "FFCX_CACHE_DIR": "/scratch/.ffcx-cache",
    }
    merged_env = {**cache_env, **dict(env or {})}
    if merged_env:
        for k, v in merged_env.items():
            docker_cmd += ["-e", f"{k}={v}"]
    docker_cmd += ["-e", "PYTHONHASHSEED=0"]

    docker_cmd += [image, *command]

    started = _now_seconds()
    timed_out = False
    oom = False
    try:
        with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
            proc = subprocess.run(
                docker_cmd,
                stdout=so,
                stderr=se,
                timeout=timeout_seconds,
                check=False,
            )
        exit_code = proc.returncode
        if exit_code == 137:
            oom = True
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = 124
    wall = _now_seconds() - started

    digest = resolve_image_digest(image)
    return SandboxResult(
        exit_code=exit_code,
        wall_seconds=wall,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timed_out=timed_out,
        oom_killed=oom,
        container_image=image,
        container_image_digest=digest,
        cmd=docker_cmd,
    )


def _now_seconds() -> float:
    """Monotonic seconds, never wall-clock — wall-clock can jump on NTP."""
    return _MONOTONIC()


import time as _time  # noqa: E402

_MONOTONIC = _time.monotonic


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test the sandbox runner.")
    parser.add_argument("--image", default=DEFAULT_IMAGE_BY_LIBRARY["evaluator"])
    parser.add_argument("--cmd", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()

    scratch = Path("/tmp/pdev-sandbox-smoke-scratch")
    artifact = Path("/tmp/pdev-sandbox-smoke-artifact")
    scratch.mkdir(exist_ok=True)
    artifact.mkdir(exist_ok=True)

    result = run(image=args.image, command=args.cmd, scratch_dir=scratch, artifact_dir=artifact)
    print(json.dumps(result.as_meta_dict(), indent=2))
    sys.exit(0 if result.exit_code == 0 else 1)
