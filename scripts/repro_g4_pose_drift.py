"""RCA repro: measure g4 task-object pose_tolerance drift per-axis.

For a handful of failing conditions, build the real LIBERO env, reset+settle,
and report for EACH movable object:
  * scenic-injected xyz, env settled xyz
  * dx, dy, dz (mm), |xy| (mm), 3D pos_err (mm), rot_err (deg)
  * passed @ 5mm/1deg
"""

from __future__ import annotations

import argparse
import json
import random

TASKS = [
    "libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl",
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl",
]
SUBSETS = [
    ("position",),
    ("object",),
    ("position", "object", "camera", "lighting", "texture"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/repro_g4_pose_drift.jsonl")
    ap.add_argument("--seeds", type=int, default=2)
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

    fh = open(args.out, "w")
    for task_rel in TASKS:
        bddl = str(resolve_task_path(task_rel))
        for subset in SUBSETS:
            for seed in range(args.seeds):
                try:
                    cfg = TaskConfig.from_bddl(bddl)
                    random.seed(seed)
                    scn = compile_task_to_scenario(cfg, ",".join(subset))
                    scene, _ = scn.generate(maxIterations=4000)
                    env = make_env(scene, bddl_path=bddl)
                    env.reset()
                except Exception as exc:  # noqa: BLE001
                    print(f"BUILD-FAIL {task_rel} {subset} s{seed}: {type(exc).__name__}: {exc}")
                    continue
                eval_scene = getattr(env, "realized_scene", None) or scene
                print(f"\n=== {task_rel} | {','.join(subset)} | seed {seed} ===")
                for o in _iter_scene_objects(eval_scene):
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
                        print(f"  {nm:24s} MISSING sp={sp} ep={ep}")
                        continue
                    dx = 1000 * (sp[0] - ep[0])
                    dy = 1000 * (sp[1] - ep[1])
                    dz = 1000 * (sp[2] - ep[2])
                    rot = p.get("rotation_error_deg")
                    rot_s = f"{rot:6.2f}" if rot is not None else "  none"
                    flag = "PASS" if res.passed else "FAIL"
                    print(
                        f"  {nm:24s} {flag}  dx={dx:8.2f} dy={dy:8.2f} dz={dz:8.2f} "
                        f"|3d|={1000*(p.get('position_error') or 0):8.2f}mm rot={rot_s}deg  "
                        f"sp=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) ep=({ep[0]:.3f},{ep[1]:.3f},{ep[2]:.3f})"
                    )
                    fh.write(
                        json.dumps(
                            {
                                "task": task_rel,
                                "subset": list(subset),
                                "seed": seed,
                                "name": nm,
                                "passed": bool(res.passed),
                                "dx_mm": dx,
                                "dy_mm": dy,
                                "dz_mm": dz,
                                "pos_err_mm": 1000 * (p.get("position_error") or 0.0),
                                "rot_err_deg": rot,
                                "scenic_pos": sp,
                                "env_pos": ep,
                            }
                        )
                        + "\n"
                    )
                env.close()
    fh.close()
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
