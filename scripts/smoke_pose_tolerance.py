"""Non-scenic-only smoke through G6 that tallies G4 family-C pose_tolerance.

Runs the real validation ``run_condition`` (G0..G6 incl. env reset) over a set
of (task, axis_subset) pairs and counts, across all movables, how many pass the
``pose_tolerance`` invariant. Writes one JSONL row per condition.

Usage:
    PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/smoke_pose_tolerance.py \
        --out /tmp/smoke_pose.jsonl [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys

from libero_infinity.validation.sweep import run_condition

TASKS = [
    "libero_goal/put_the_bowl_on_the_stove.bddl",            # bowl, plate, bottle, box
    "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl",
    "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",   # cabinet
    "libero_goal/put_the_wine_bottle_on_the_rack.bddl",      # bottle
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",  # pot, frypan
    "libero_90/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.bddl",  # cabinet+contained
]

SUBSETS = [
    ("position",),
    ("position", "object"),
    ("position", "camera"),
    ("object",),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/smoke_pose.jsonl")
    ap.add_argument("--quick", action="store_true", help="2 tasks x 1 subset only")
    args = ap.parse_args()

    if args.quick:
        tasks, subsets = TASKS[:3], SUBSETS[:1]
    else:
        tasks, subsets = TASKS, SUBSETS

    n_true = n_false = n_none = 0
    per_class: dict[str, list[int]] = {}
    rows = 0
    with open(args.out, "w") as fh:
        for task in tasks:
            for subset in subsets:
                row = run_condition(task, subset, seed=0, scenic_only=False, max_iter=2000)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows += 1
                con = row.get("g4_consistency") or {}
                for key, passed in con.items():
                    if not key.startswith("pose_tolerance:"):
                        continue
                    cls = key.split(":", 1)[1].rsplit("_", 1)[0]
                    if passed is True:
                        n_true += 1
                        per_class.setdefault(cls, [0, 0])[0] += 1
                    elif passed is False:
                        n_false += 1
                        per_class.setdefault(cls, [0, 0])[1] += 1
                    else:
                        n_none += 1
                g5 = row.get("g5")
                g6 = row.get("g6")
                print(
                    f"[{rows}] {task.split('/')[-1][:42]:42} {','.join(subset):20} "
                    f"g5={g5} g6={g6} pose_tol so far: {n_true}T/{n_false}F/{n_none}N",
                    flush=True,
                )

    total = n_true + n_false
    print("\n===== POSE_TOLERANCE SUMMARY =====")
    print(f"movables with pose_tolerance data: {total} (+{n_none} None/skip)")
    print(f"  TRUE : {n_true}")
    print(f"  FALSE: {n_false}")
    if total:
        print(f"  pass rate: {n_true}/{total} = {100*n_true/total:.0f}%")
    print("per-class (true/false):")
    for cls, (t, f) in sorted(per_class.items()):
        print(f"  {cls:28} {t}T / {f}F")
    print(f"\nJSONL: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
