"""Measure the end-of-settle velocity / dz / xy distributions for g4 living_room
metastable objects, split by whether the STRICT 5mm/1deg gate passes.

Purpose: pick physically-defensible thresholds for the alt-rest acceptance path
(consistency.py) that cleanly separate converged stable rests (the residual tail
we want to admit) from moving/tipped/fallen states (which must stay rejected).

Prints, per (class, strict_pass?): count, and min/median/max of
lin_speed, ang_speed, |dz|, xy_drift, rot_err.
"""

import argparse
import collections
import random
import statistics

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--subset", default="position")
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import (
        is_scene_fixture,
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.consistency import (
        _coerce_quat,
        _env_get_object,
        _quat_angle_deg,
    )
    from libero_infinity.validation.invariants.domain import (
        _iter_scene_objects,
        _obj_position,
    )
    from libero_infinity.validation.sweep import resolve_task_path

    tasks = [t for t in args.tasks.split(",") if t.strip()] or [
        "libero_90/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.bddl",
    ]

    POS_TOL, ROT_TOL = 0.005, 1.0
    # rows: (cls, strict_pass) -> dict of lists
    agg = collections.defaultdict(lambda: {"lin": [], "ang": [], "dz": [], "xy": [], "rot": []})

    for task in tasks:
        path = resolve_task_path(task)
        for seed in range(args.seeds):
            random.seed(seed)
            np.random.seed(seed)
            try:
                cfg = TaskConfig.from_bddl(str(path))
                scn = compile_task_to_scenario(cfg, args.subset)
                scene, _ = scn.generate(maxIterations=8000)
                env = make_env(scene, bddl_path=str(path))
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP {task} seed{seed}: {type(exc).__name__}: {exc}")
                continue
            try:
                env.reset()
                scene = env.realized_scene
                for o in _iter_scene_objects(scene):
                    if is_scene_fixture(o):
                        continue
                    nm = resolve_object_name(o)
                    try:
                        st = _env_get_object(env, nm)
                    except Exception:  # noqa: BLE001
                        continue
                    cls = getattr(o, "asset_class", "?")
                    s_pos = _obj_position(o)
                    e_pos = st.get("position")
                    if s_pos is None or e_pos is None:
                        continue
                    dz = abs(float(s_pos[2]) - float(e_pos[2]))
                    xy = float(np.hypot(s_pos[0] - e_pos[0], s_pos[1] - e_pos[1]))
                    pos_err = float(np.linalg.norm([s_pos[i] - e_pos[i] for i in range(3)]))
                    s_ori = _coerce_quat(getattr(o, "orientation", None))
                    e_ori = _coerce_quat(st.get("orientation"))
                    rot = _quat_angle_deg(s_ori, e_ori) if (s_ori and e_ori) else 0.0
                    strict = pos_err <= POS_TOL and rot <= ROT_TOL
                    lin = st.get("settle_conv_lin")
                    ang = st.get("settle_conv_ang")
                    key = (cls, strict)
                    if lin is not None:
                        agg[key]["lin"].append(lin)
                        agg[key]["ang"].append(ang)
                    agg[key]["dz"].append(dz)
                    agg[key]["xy"].append(xy)
                    agg[key]["rot"].append(rot)
            finally:
                env.close()

    def stats(xs):
        if not xs:
            return "  --"
        return f"n={len(xs):3d} min={min(xs):.5f} med={statistics.median(xs):.5f} max={max(xs):.5f}"

    print("\n=== per (class, strict_pass): metrics ===")
    for (cls, strict), d in sorted(agg.items()):
        tag = "PASS" if strict else "FAIL"
        print(f"\n{cls}  strict={tag}")
        print(f"  conv_lin(m): {stats(d['lin'])}")
        print(f"  conv_ang(°): {stats(d['ang'])}")
        print(f"  |dz| (m)  : {stats(d['dz'])}")
        print(f"  xy   (m)  : {stats(d['xy'])}")
        print(f"  rot (deg) : {stats(d['rot'])}")


if __name__ == "__main__":
    main()
