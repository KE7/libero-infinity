#!/usr/bin/env python3
"""Batch-generate reversed BDDL files from a directory of forward tasks.

Usage::

    python scripts/generate_reversed_bddls.py \
        --input src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/ \
        --output data/reversed_bddls/libero_goal/
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate reversed BDDL files")
    p.add_argument(
        "--input",
        required=True,
        help="Directory containing forward-task .bddl files",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output directory for reversed .bddl files",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Return-region half-width in metres (default: 0.05)",
    )
    args = p.parse_args(argv)

    from libero_infinity.task_reverser import reverse_bddl

    in_dir = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bddl_files = sorted(in_dir.glob("*.bddl"))
    if not bddl_files:
        print(f"No .bddl files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    skipped = 0
    for bddl_path in bddl_files:
        try:
            original = bddl_path.read_text()
            reversed_content = reverse_bddl(original, return_region_margin=args.margin)
            out_path = out_dir / bddl_path.name
            out_path.write_text(reversed_content)
            print(f"  OK  {bddl_path.name}")
            ok += 1
        except Exception as e:
            print(f"  SKIP {bddl_path.name}: {e}")
            skipped += 1

    print(f"\nDone: {ok} reversed, {skipped} skipped")


if __name__ == "__main__":
    main()
