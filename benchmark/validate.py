"""Validate generated cases against ``schemas/pde_case.schema.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
import numpy as np

ROOT = Path(__file__).parent.parent


def validate_file(*, cases_path: Path, schema_path: Path, check_references: bool) -> int:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = 0
    n = 0
    for ln, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        n += 1
        case_errors = list(validator.iter_errors(case))
        if case_errors:
            errors += len(case_errors)
            for e in case_errors:
                print(f"  [line {ln}] schema: {e.message} (path: {list(e.absolute_path)})", file=sys.stderr)
            continue
        if check_references:
            ref_path = ROOT / case["reference"]["path"]
            if not ref_path.is_file():
                print(f"  [line {ln}] reference missing: {ref_path}", file=sys.stderr)
                errors += 1
                continue
            with np.load(ref_path, allow_pickle=False) as data:
                keys = list(data.keys())
                if not keys:
                    print(f"  [line {ln}] reference empty: {ref_path}", file=sys.stderr)
                    errors += 1
                    continue
                arr = np.asarray(data[keys[0]])
                if np.any(np.isinf(arr)):
                    print(f"  [line {ln}] reference has Inf values: {ref_path}", file=sys.stderr)
                    errors += 1
                elif not np.any(np.isfinite(arr)):
                    print(f"  [line {ln}] reference is entirely non-finite: {ref_path}", file=sys.stderr)
                    errors += 1
    print(f"validated {n} cases; {errors} errors")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--schema", type=Path, default=ROOT / "schemas" / "pde_case.schema.json")
    p.add_argument("--check-references", action="store_true")
    args = p.parse_args(argv)
    return 1 if validate_file(
        cases_path=args.cases,
        schema_path=args.schema,
        check_references=args.check_references,
    ) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
