"""Evaluator-owned placeholder for a failed model-generation call."""

import numpy as np

def solve(problem_spec: dict) -> np.ndarray:
    raise RuntimeError('generation call 15 failed: Exceeded wall_timeout_seconds=600.0')
