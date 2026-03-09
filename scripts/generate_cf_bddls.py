#!/usr/bin/env python3
"""Batch-generate counterfactual (CF) BDDL files for all LIBERO task suites.

For each task whose goal is ``(On X Y)`` or ``(In X Y)``, we generate one CF
variant per *other* graspable object in the scene.  The CF variant keeps the
exact same scene layout and destination, but re-targets the language instruction
to a different object — forcing the policy to demonstrate genuine language
understanding rather than canonical-object bias.

Example
-------
Original task: "Put the bowl on the plate"
CF variant 1:  "Put the cream cheese on the plate"   (cream_cheese_1 → plate_1)
CF variant 2:  "Put the wine bottle on the plate"    (wine_bottle_1  → plate_1)

Usage
-----
    python scripts/generate_cf_bddls.py                      # default output dir
    python scripts/generate_cf_bddls.py --out-root /tmp/cf   # custom output dir
    python scripts/generate_cf_bddls.py --suite libero_goal  # single suite
    python scripts/generate_cf_bddls.py --dry-run            # count only
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Add src/ so we can import without installing
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from libero_infinity.bddl_preprocessor import generate_cf_bddls

from libero_infinity.runtime import get_bddl_dir  # noqa: E402

LIBERO_BDDL_ROOT = get_bddl_dir()

SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]


def process_suite(
    suite_dir: pathlib.Path,
    out_root: pathlib.Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Generate CF variants for all BDDLs in *suite_dir*.

    Returns (n_tasks, n_generated, n_skipped).
    """
    out_dir = out_root / suite_dir.name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = n_generated = n_skipped = 0
    for bddl_path in sorted(suite_dir.glob("*.bddl")):
        n_tasks += 1
        content = bddl_path.read_text()
        variants = generate_cf_bddls(content)
        if not variants:
            n_skipped += 1
            continue

        for suffix, cf_text in variants:
            out_path = out_dir / (bddl_path.stem + suffix + ".bddl")
            if not dry_run:
                out_path.write_text(cf_text)
            n_generated += 1

    return n_tasks, n_generated, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent / "bddl_cf",
        help="Root directory for generated CF BDDL files (default: bddl_cf/)",
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default=None,
        help="Generate only for a specific suite (default: all suites)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count variants without writing files",
    )
    args = parser.parse_args()

    suites = [args.suite] if args.suite else SUITES
    total_tasks = total_generated = total_skipped = 0

    for suite_name in suites:
        suite_dir = LIBERO_BDDL_ROOT / suite_name
        if not suite_dir.exists():
            print(f"  [SKIP] {suite_name}: directory not found at {suite_dir}")
            continue

        n_tasks, n_gen, n_skip = process_suite(suite_dir, args.out_root, args.dry_run)
        total_tasks += n_tasks
        total_generated += n_gen
        total_skipped += n_skip
        print(
            f"  {suite_name:20s}  {n_tasks:3d} tasks  "
            f"{n_gen:4d} CF variants generated  "
            f"{n_skip:3d} skipped (non-On/In goal)"
        )

    print()
    action = "Would generate" if args.dry_run else "Generated"
    print(f"{action} {total_generated} CF BDDL files from {total_tasks} tasks "
          f"({total_skipped} skipped) → {args.out_root}")


if __name__ == "__main__":
    main()
