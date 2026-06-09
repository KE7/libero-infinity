"""Decompose pose_tolerance error into xy vs z for task objects, comparing a
NON-robot subset (position) against position,robot, across arenas. Tells whether
the residual is a universal z-frame bug or a robot-specific xy shove."""
from __future__ import annotations
import random, sys
import numpy as np
from libero_infinity.compiler import compile_task_to_scenario
from libero_infinity.gym_env import make_env
from libero_infinity.task_config import TaskConfig
from libero_infinity.validation.invariants._scene_view import is_scene_fixture, resolve_object_name
from libero_infinity.validation.invariants.consistency import _env_get_object, assert_pose_tolerance
from libero_infinity.validation.invariants.domain import _iter_scene_objects
from libero_infinity.validation.sweep import resolve_task_path

TASKS = [
    "libero_90/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
    "libero_goal/put_the_wine_bottle_on_the_rack.bddl",
]
SUBSETS = ["position", "position,robot"]
SEEDS = [0,1,2,3]

def main():
    for task in TASKS:
        bddl = str(resolve_task_path(task))
        print(f"\n==== {task.split('/')[-1]}")
        for subset in SUBSETS:
            xy_max=z_max=tot_max=0.0; npass=nfail=0; worst=""
            for seed in SEEDS:
                try:
                    cfg=TaskConfig.from_bddl(bddl); random.seed(seed)
                    scn=compile_task_to_scenario(cfg,subset); scene,_=scn.generate(maxIterations=6000)
                    env=make_env(scene,bddl_path=bddl); env.reset()
                except Exception as e:
                    print(f"   [{subset} seed{seed}] build fail {type(e).__name__}"); continue
                for o in _iter_scene_objects(scene):
                    if is_scene_fixture(o): continue
                    nm=resolve_object_name(o)
                    if not nm or str(nm).startswith("distractor_"): continue
                    try: res=assert_pose_tolerance(o,_env_get_object(env,nm))
                    except Exception: continue
                    sp=res.payload.get("scenic_position"); ep=res.payload.get("env_position")
                    if sp is None or ep is None: continue
                    sp=np.array(sp,float); ep=np.array(ep,float)
                    xy=float(np.hypot(*(sp[:2]-ep[:2])))*1000; z=abs(float(sp[2]-ep[2]))*1000
                    tot=float(np.linalg.norm(sp-ep))*1000
                    if res.passed: npass+=1
                    else: nfail+=1
                    if tot>tot_max: tot_max=tot; worst=f"{nm}(xy={xy:.0f},z={z:.0f})"
                    xy_max=max(xy_max,xy); z_max=max(z_max,z)
                env.close()
            print(f"   [{subset:>15}] pass={npass} fail={nfail}  MAX xy={xy_max:.0f}mm z={z_max:.0f}mm tot={tot_max:.0f}mm worst={worst}")

if __name__=="__main__":
    sys.exit(main())
