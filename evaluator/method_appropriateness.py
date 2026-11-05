"""Stage 4 — Method appropriateness."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

from evaluator.exact_formula import extract_evidence_lines, is_exact_formula_shortcut
from evaluator.thresholds import LLM_JUDGE_SAMPLE_RATE

NEAR_ZERO_REL_L2_THRESHOLD = 1e-14


@dataclasses.dataclass(frozen=True)
class CheckResult:
    rule_id: str
    rule_description: str
    passed: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
            "passed": self.passed,
            "evidence": self.evidence,
        }


@dataclasses.dataclass(frozen=True)
class MethodVerdict:
    pass_: bool
    checks: list[CheckResult]
    llm_judge_invoked: bool
    llm_judge_verdict: str | None
    llm_judge_rationale: str | None
    llm_judge_model: str | None
    human_verified: bool

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pass": self.pass_,
            "checks": [c.to_dict() for c in self.checks],
            "llm_judge_invoked": self.llm_judge_invoked,
            "human_verified": self.human_verified,
        }
        if self.llm_judge_invoked:
            d["llm_judge_verdict"] = self.llm_judge_verdict
            d["llm_judge_rationale"] = self.llm_judge_rationale
            d["llm_judge_model"] = self.llm_judge_model
        return d


@dataclasses.dataclass(frozen=True)
class MethodRule:
    rule_id: str
    description: str
    applies_when: Callable[[dict[str, Any]], bool]
    is_satisfied: Callable[[str], tuple[bool, str]]


def _operator_is(family: str) -> Callable[[dict[str, Any]], bool]:
    return lambda case: case.get("case_spec", {}).get("pde", {}).get("operator_family") == family


def _has_any(*patterns: str) -> Callable[[str], tuple[bool, str]]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def fn(code: str) -> tuple[bool, str]:
        for p in compiled:
            m = p.search(code)
            if m:
                return True, f"matched pattern '{p.pattern}'"
        return False, f"none of {[p.pattern for p in compiled]} matched"

    return fn


RULES: list[MethodRule] = [
    MethodRule(
        rule_id="R-BIHARM-SPACE",
        description="Biharmonic problems require C^0-IP, mixed, or H^2-conforming spaces.",
        applies_when=_operator_is("biharmonic"),
        is_satisfied=_has_any(r"C0IP|Argyris|Hsieh.*Clough.*Tocher|HHJ|Morley|mixed"),
    ),
    MethodRule(
        rule_id="R-STOKES-INFSUP",
        description="Stokes requires an inf-sup-stable velocity/pressure pair.",
        applies_when=_operator_is("Stokes"),
        is_satisfied=_has_any(
            r"Taylor.?Hood",
            r"\bP2\b.*\bP1\b",
            r"MINI",
            r"Crouzeix.?Raviart.*\bP0\b",
            r"BDM.*\bDG\b",
        ),
    ),
    MethodRule(
        rule_id="R-NS-INFSUP",
        description="Steady Navier–Stokes requires inf-sup-stable pair or equal-order + stabilization.",
        applies_when=_operator_is("Navier_Stokes_steady"),
        is_satisfied=_has_any(
            r"Taylor.?Hood",
            r"\bP2\b.*\bP1\b",
            r"GLS|PSPG|SUPG",
        ),
    ),
    MethodRule(
        rule_id="R-CONV-DOM-STAB",
        description=(
            "Convection-dominated transport requires SUPG, GLS, upwind, or DG. "
            "Applies to: (a) the explicit advection_diffusion_convection_dominated "
            "operator family, (b) Burgers ONLY when viscosity ν is low enough that "
            "the typical cell Péclet exceeds 2 (heuristic ν < 0.01 at h ≤ 1/15). "
            "Viscous Burgers (ν ≥ 0.01) at moderate Re is in "
            "the diffusion-comparable regime where centered FD + IMEX is "
            "method-appropriate."
        ),
        applies_when=lambda case: (
            case.get("case_spec", {}).get("pde", {}).get("operator_family") == "advection_diffusion_convection_dominated"
            or (
                case.get("case_spec", {}).get("pde", {}).get("operator_family") == "Burgers"
                and float(case.get("case_spec", {}).get("pde", {}).get("parameters", {}).get("nu", 1.0)) < 0.01
            )
        ),
        is_satisfied=_has_any(r"SUPG|GLS|upwind|DG", r"streamline.?diffusion"),
    ),
    MethodRule(
        rule_id="R-SADDLE-NULLSPACE",
        description="Pure-Neumann or otherwise saddle-point problems must handle the null space.",
        applies_when=lambda case: case.get("case_spec", {}).get("bc", {}).get("kind") == "pure_neumann",
        is_satisfied=_has_any(r"nullspace|pin.+point|projection.+constant"),
    ),
    MethodRule(
        rule_id="R-MAXWELL-NEDELEC",
        description="Curl–curl (Maxwell) requires Nédélec / edge elements.",
        applies_when=_operator_is("Maxwell"),
        is_satisfied=_has_any(r"Nedelec|N1curl|edge.element"),
    ),
    MethodRule(
        rule_id="R-DARCY-HDIV",
        description="Mixed Darcy requires H(div)-conforming spaces (Raviart–Thomas, BDM).",
        applies_when=_operator_is("Darcy_mixed"),
        is_satisfied=_has_any(r"Raviart.?Thomas|RT[0-9]+|BDM"),
    ),
]


def check(
    *,
    case: dict[str, Any],
    final_code: str,
    accuracy_passed: bool,
    accuracy_rel_l2: float | None = None,
    exact_formula_near_zero_threshold: float = NEAR_ZERO_REL_L2_THRESHOLD,
    invoke_llm_judge: bool = False,
    llm_judge: Callable[[dict[str, Any], str], tuple[str, str, str]] | None = None,
    rng_uniform_0_1: Callable[[], float] | None = None,
) -> MethodVerdict:
    """Apply Stage 4.

    Parameters
    ----------
    case
        Case dict (schema-validated).
    final_code
        UTF-8 string of the agent-produced solver code.
    accuracy_passed
        Whether Stage 2 passed. Drives the LLM-judge sampling rule.
    accuracy_rel_l2
        Stage-2 relative L2. Near-zero passing outputs are additionally checked
        for direct manufactured/analytic formula shortcuts.
    exact_formula_near_zero_threshold
        Threshold below which a passing output is audited for exact-formula
    invoke_llm_judge
        Force the LLM-judge on. Used by ad-hoc tests; production callers leave
        this False and let the sampling rule decide.
    llm_judge
        Injection point: a callable that takes (case, final_code) and returns
        (verdict, rationale, model_id). The default is "not invoked".
    rng_uniform_0_1
        Injection point for sampling. Default ``random.random`` is used inside
    """
    checks: list[CheckResult] = []
    for rule in RULES:
        if not rule.applies_when(case):
            continue
        ok, evidence = rule.is_satisfied(final_code)
        checks.append(CheckResult(
            rule_id=rule.rule_id,
            rule_description=rule.description,
            passed=ok,
            evidence=evidence,
        ))

    if (
        accuracy_passed
        and accuracy_rel_l2 is not None
        and accuracy_rel_l2 < exact_formula_near_zero_threshold
    ):
        is_shortcut, classification = is_exact_formula_shortcut(final_code)
        evidence_lines = extract_evidence_lines(final_code, max_lines=4)
        evidence = (
            f"classification={classification['label']}; "
            f"rel_l2={accuracy_rel_l2:.3e}; "
            f"features={classification}; "
            f"evidence_lines={evidence_lines}"
        )
        checks.append(CheckResult(
            rule_id="R-NO-EXACT-FORMULA",
            rule_description=(
                "Near-zero outputs must come from a numerical PDE solve, not "
                "from directly evaluating the manufactured/analytic solution "
                "on the evaluation grid."
            ),
            passed=not is_shortcut,
            evidence=evidence,
        ))

    heuristic_pass = all(c.passed for c in checks)
    judge_invoked = False
    judge_verdict: str | None = None
    judge_rationale: str | None = None
    judge_model: str | None = None

    needs_judge = invoke_llm_judge or (not accuracy_passed)
    if not needs_judge and rng_uniform_0_1 is not None:
        needs_judge = rng_uniform_0_1() < LLM_JUDGE_SAMPLE_RATE

    if needs_judge and llm_judge is not None:
        judge_invoked = True
        judge_verdict, judge_rationale, judge_model = llm_judge(case, final_code)

    overall = heuristic_pass and (not judge_invoked or judge_verdict != "inappropriate")

    return MethodVerdict(
        pass_=overall,
        checks=checks,
        llm_judge_invoked=judge_invoked,
        llm_judge_verdict=judge_verdict,
        llm_judge_rationale=judge_rationale,
        llm_judge_model=judge_model,
        human_verified=False,
    )


__all__ = ["MethodRule", "MethodVerdict", "CheckResult", "RULES", "check"]
