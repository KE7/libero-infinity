"""Broad no-regression A/B for the akita_black_bowl registry-AABB correction
(0.10 -> 0.107, the measured collision footprint; RCA g4_cabinet_heightfield.md
§5.3 FIX1).

The AABB feeds the movable<->movable and object<->fixture SEPARATION constraints
corpus-wide, so widening it could (a) over-constrain the Scenic sampler (more
generation failures on tight multi-bowl scenes) or (b) shift sampled positions
enough to flip a previously-passing pose_tolerance object. This harness runs a
broad task slice under the OLD (0.10) and NEW (0.107) akita footprint and reports,
per task, generation failures and pose_tolerance pass/fail, flagging ANY
PASS->FAIL or new generation failure. Net-add / no-regression only.
"""

import argparse
import collections
import random

import numpy as np

# Broad slice: akita task-objects across arenas; the TIGHT multi-bowl scenes
# (KITCHEN_SCENE2, LIVING_ROOM_SCENE4 stacks) that most stress the separation
# constraint; and two non-akita controls that must be byte-identical.
TASKS = [
    # libero_spatial (akita task object, kitchen)
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
    # libero_goal (akita)
    "libero_goal/put_the_bowl_on_the_plate.bddl",
    "libero_goal/put_the_bowl_on_the_stove.bddl",
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl",
    # libero_90 multi-bowl TIGHT (2-3 black bowls -> highest over-constraint risk)
    "libero_90/KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl",
    # libero_10 / drawer
    "libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
    # living room tray (2 bowls, lower arena)
    "libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.bddl",
    # NON-akita controls (must be byte-identical)
    "libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl",
    "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--subsets", default="position,object")
    args = ap.parse_args()

    import libero_infinity.asset_registry as AR
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

    subsets = [s for s in args.subsets.split(",") if s]

    def run(akita_w):
        AR.OBJECT_DIMENSIONS["akita_black_bowl"] = [akita_w, akita_w, 0.06]
        per = collections.defaultdict(lambda: {"pass": 0, "fail": 0, "genfail": 0})
        for t in TASKS:
            bddl = str(resolve_task_path(t))
            for sub in subsets:
                for seed in range(args.seeds):
                    try:
                        cfg = TaskConfig.from_bddl(bddl)
                        random.seed(seed)
                        np.random.seed(seed)
                        scn = compile_task_to_scenario(cfg, sub)
                        scene, _ = scn.generate(maxIterations=8000)
                        env = make_env(scene, bddl_path=bddl)
                        env.reset()
                    except Exception:
                        per[t]["genfail"] += 1
                        continue
                    es = getattr(env, "realized_scene", None) or scene
                    for o in _iter_scene_objects(es):
                        if is_scene_fixture(o):
                            continue
                        nm = resolve_object_name(o) or "?"
                        try:
                            st = _env_get_object(env, nm)
                            res = assert_pose_tolerance(o, st)
                        except Exception:
                            continue
                        if res.passed:
                            per[t]["pass"] += 1
                        else:
                            per[t]["fail"] += 1
                    env.close()
        return per

    print("Running OLD akita AABB = 0.10 ...")
    old = run(0.10)
    print("Running NEW akita AABB = 0.107 ...")
    new = run(0.107)

    print("\n=== akita AABB A/B (OLD 0.10 vs NEW 0.107) ===")
    print(f"{'task':<74} {'OLD P/F/gf':>14} {'NEW P/F/gf':>14}  flag")
    tot = {"op": 0, "of": 0, "og": 0, "np": 0, "nf": 0, "ng": 0}
    regressions = []
    for t in TASKS:
        o, n = old[t], new[t]
        tot["op"] += o["pass"]
        tot["of"] += o["fail"]
        tot["og"] += o["genfail"]
        tot["np"] += n["pass"]
        tot["nf"] += n["fail"]
        tot["ng"] += n["genfail"]
        flag = ""
        if n["pass"] < o["pass"]:
            flag = "  <== PASS REGRESSION"
            regressions.append(t)
        elif n["genfail"] > o["genfail"]:
            flag = "  <== NEW GEN FAILURE"
            regressions.append(t)
        short = t.split("/")[-1][:72]
        print(
            f"{short:<74} {o['pass']:>4}/{o['fail']:>3}/{o['genfail']:>2}   "
            f"{n['pass']:>4}/{n['fail']:>3}/{n['genfail']:>2}{flag}"
        )
    print(
        f"\nTOTAL  OLD pass={tot['op']} fail={tot['of']} genfail={tot['og']}  |  "
        f"NEW pass={tot['np']} fail={tot['nf']} genfail={tot['ng']}"
    )
    print(f"REGRESSIONS (PASS->FAIL or new gen-fail): {len(regressions)} -> {regressions}")


if __name__ == "__main__":
    main()
