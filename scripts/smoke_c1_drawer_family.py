"""Smoke verification for campaign caveat c.1 (drawer-family G3 failures).

Generates Scenic programs for the 10 `libero_goal/*` tasks identified in
`worst_50_rca.md` and tries `scenario.generate(maxIterations=25000)` across
multiple axis-subsets x seeds. Prints a per-task / per-condition pass/fail
table and aggregate stats.
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback
from pathlib import Path

import scenic as sc
from scenic.core.distributions import RejectionException

from libero_infinity.compiler import generate_scenic_file
from libero_infinity.task_config import TaskConfig

REPO = Path(__file__).resolve().parents[1]
BDDL_DIR = REPO / "libero-pro" / "libero" / "libero" / "bddl_files" / "libero_goal"
if not BDDL_DIR.exists():
    BDDL_DIR = REPO / "vendor" / "libero" / "libero" / "libero" / "bddl_files" / "libero_goal"

TASKS = [
    "turn_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "put_the_wine_bottle_on_the_rack",
    "push_the_plate_to_the_front_of_the_stove",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_bowl_on_the_plate",
]

# 5 axis-subsets representative of the cardinality-monotone failure mode
# (low to high cardinality, including the worst — cardinality 8 — case
# identified in stage4_c1_addendum_10_task_footprint.md).
AXIS_SUBSETS = [
    "position,distractor",
    "position,distractor,camera",
    "position,distractor,lighting,object",
    "position,distractor,lighting,object,background,robot",
    "combined",  # cardinality 7, includes all object-fixture interactions
]

SEEDS = [13, 17, 23, 29, 31]
MAX_ITER = 25_000


def run_one(bddl_path: Path, axes: str, seed: int) -> tuple[bool, int, str]:
    cfg = TaskConfig.from_bddl(str(bddl_path))
    scenic_path = generate_scenic_file(cfg, perturbation=axes)
    random.seed(seed)
    scenario = sc.scenarioFromFile(
        scenic_path, params={"bddl_path": str(bddl_path)}
    )
    t0 = time.time()
    try:
        scene, iters_used = scenario.generate(maxIterations=MAX_ITER, verbosity=0)
        return True, iters_used, ""
    except RejectionException as exc:
        return False, MAX_ITER, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, -1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    total = 0
    passed = 0
    fail_records: list[str] = []
    max_iters_seen = 0
    rows: list[tuple[str, str, int, bool, int]] = []

    for task in TASKS:
        bddl = BDDL_DIR / f"{task}.bddl"
        if not bddl.exists():
            print(f"MISSING: {bddl}", file=sys.stderr)
            return 2
        for axes in AXIS_SUBSETS:
            for seed in SEEDS:
                total += 1
                ok, iters, err = run_one(bddl, axes, seed)
                rows.append((task, axes, seed, ok, iters))
                if ok:
                    passed += 1
                    max_iters_seen = max(max_iters_seen, iters)
                else:
                    fail_records.append(
                        f"FAIL {task} | {axes} | seed={seed} | iters={iters} | {err[:120]}"
                    )
                print(
                    f"{'PASS' if ok else 'FAIL'} {task:55s} {axes:60s} seed={seed:>3} iters={iters:>6}"
                )

    print()
    print("=" * 78)
    print(f"Total conditions:   {total}")
    print(f"Passed:             {passed}  ({100.0 * passed / total:.3f}%)")
    print(f"Failed:             {total - passed}")
    print(f"Max iters observed: {max_iters_seen} (cap={MAX_ITER})")
    if fail_records:
        print()
        print("Failure detail:")
        for r in fail_records:
            print("  " + r)
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
