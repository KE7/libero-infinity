"""Non-scenic-only smoke for the consolidated placement-clearance fix.

Builds real LIBERO/MuJoCo envs across the axis subsets that exercise the three
fixes (robot in the require graph, distractor↔object AABB, per-(variant,surface)
z) and reports, per movable:

  * ``pose_tolerance`` pass rate (Scenic-injected vs post-settle pose @ 5 mm), and
  * xy displacement |settled_xy − injected_xy| (the RCA Finding-B metric).

Usage:
    PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/smoke_clearance_fix.py \
        --out /tmp/smoke_clearance.jsonl [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics

TASKS = [
    "libero_goal/put_the_bowl_on_the_stove.bddl",
    "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl",
    "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
    "libero_goal/put_the_wine_bottle_on_the_rack.bddl",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
]

# Subsets chosen to exercise the robot-clearance and distractor↔object fixes
# (the dominant RCA Finding-B failure modes), plus object+robot for Fix 3.
SUBSETS = [
    ("robot",),
    ("distractor",),
    ("robot", "distractor"),
    ("object", "robot"),
    ("position", "robot"),
    ("position", "object", "robot", "camera", "lighting", "texture", "distractor", "background"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/smoke_clearance.jsonl")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import (
        is_scene_fixture,
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.consistency import (
        _env_get_object,
        assert_pose_tolerance,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    n_pass = n_fail = 0
    xy_robot_axis: list[float] = []
    xy_all: list[float] = []
    rows = 0
    n_conditions = 0
    n_build_fail = 0

    with open(args.out, "w") as fh:
        for task_rel in TASKS:
            bddl = str(resolve_task_path(task_rel))
            for subset in SUBSETS:
                for seed in range(args.seeds):
                    n_conditions += 1
                    try:
                        cfg = TaskConfig.from_bddl(bddl)
                        random.seed(seed)
                        scn = compile_task_to_scenario(cfg, ",".join(subset))
                        scene, _ = scn.generate(maxIterations=4000)
                        env = make_env(scene, bddl_path=bddl)
                        env.reset()
                    except Exception as exc:  # noqa: BLE001 — recorded, not masked
                        n_build_fail += 1
                        fh.write(
                            json.dumps(
                                {
                                    "task": task_rel,
                                    "subset": list(subset),
                                    "seed": seed,
                                    "build_error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            + "\n"
                        )
                        continue
                    robot_axis = "robot" in subset
                    for o in _iter_scene_objects(scene):
                        if is_scene_fixture(o):
                            continue
                        nm = resolve_object_name(o) or "?"
                        try:
                            st = _env_get_object(env, nm)
                        except Exception:  # noqa: BLE001
                            continue
                        res = assert_pose_tolerance(o, st)
                        p = res.payload
                        sp = p.get("scenic_position")
                        ep = p.get("env_position")
                        if sp is None or ep is None:
                            continue
                        xy = math.hypot(sp[0] - ep[0], sp[1] - ep[1])
                        xy_all.append(xy)
                        if robot_axis:
                            xy_robot_axis.append(xy)
                        if res.passed:
                            n_pass += 1
                        else:
                            n_fail += 1
                        fh.write(
                            json.dumps(
                                {
                                    "task": task_rel,
                                    "subset": list(subset),
                                    "seed": seed,
                                    "name": nm,
                                    "passed": bool(res.passed),
                                    "pos_err_mm": round(1000 * (p.get("position_error") or 0.0), 2),
                                    "xy_mm": round(1000 * xy, 2),
                                }
                            )
                            + "\n"
                        )
                        rows += 1
                    env.close()

    total = n_pass + n_fail
    pass_rate = (100.0 * n_pass / total) if total else 0.0

    def _stats(xs: list[float]) -> str:
        if not xs:
            return "n=0"
        xs_mm = [1000 * x for x in xs]
        return (
            f"n={len(xs_mm)} mean={statistics.mean(xs_mm):.1f}mm "
            f"median={statistics.median(xs_mm):.1f}mm max={max(xs_mm):.1f}mm "
            f"p90={sorted(xs_mm)[int(0.9 * (len(xs_mm) - 1))]:.1f}mm"
        )

    print(f"\n=== smoke_clearance_fix: {n_conditions} conditions, {n_build_fail} build-fail ===")
    print(f"movable pose_tolerance: {n_pass}/{total} = {pass_rate:.1f}% pass")
    print(f"xy displacement (robot-axis subsets): {_stats(xy_robot_axis)}")
    print(f"xy displacement (all subsets):        {_stats(xy_all)}")
    print(f"rows written: {rows} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
