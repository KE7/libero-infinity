"""Quantify the settle-retry scene-desync lever across the smoke matrix.

For each (task, subset, seed): build via the smoke's path (external preset
scene), reset the env, then evaluate pose_tolerance TWO ways:
  * PRESET  : compare env state vs the externally-held scene (current smoke/sweep)
  * ACTUAL  : compare env state vs env._sim.scene (the scene the env realized)
Reports per-object pass rates both ways + how many conditions retried.
"""
from __future__ import annotations
import random, sys
import numpy as np

TASKS = [
    "libero_goal/put_the_bowl_on_the_stove.bddl",
    "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl",
    "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
    "libero_goal/put_the_wine_bottle_on_the_rack.bddl",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
]
SUBSETS = [("position","robot"), ("position","object","robot","camera","lighting","texture","distractor","background")]
SEEDS = list(range(8))

def main():
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import is_scene_fixture, resolve_object_name
    from libero_infinity.validation.invariants.consistency import _env_get_object, assert_pose_tolerance
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    def eval_scene(sc, env):
        p=f=0
        for o in _iter_scene_objects(sc):
            if is_scene_fixture(o): continue
            nm=resolve_object_name(o)
            if not nm or str(nm).startswith("distractor_"): continue
            try: r=assert_pose_tolerance(o,_env_get_object(env,nm))
            except Exception: continue
            if r.passed: p+=1
            else: f+=1
        return p,f

    pp=pf=ap=af=0; retried=0; total=0
    for task in TASKS:
        bddl=str(resolve_task_path(task))
        for subset in SUBSETS:
            for seed in SEEDS:
                total+=1
                try:
                    cfg=TaskConfig.from_bddl(bddl); random.seed(seed)
                    scn=compile_task_to_scenario(cfg,",".join(subset)); scene,_=scn.generate(maxIterations=5000)
                    env=make_env(scene,bddl_path=bddl); env.reset()
                except Exception:
                    continue
                actual=env._sim.scene
                if actual is not scene: retried+=1
                a,b=eval_scene(scene,env); pp+=a; pf+=b
                c,d=eval_scene(actual,env); ap+=c; af+=d
                env.close()
    print(f"conditions={total} retried={retried} ({100*retried/max(total,1):.1f}%)")
    print(f"PRESET (current): pass={pp} fail={pf}  rate={100*pp/max(pp+pf,1):.1f}%")
    print(f"ACTUAL (fixed)  : pass={ap} fail={af}  rate={100*ap/max(ap+af,1):.1f}%")

if __name__=="__main__":
    sys.exit(main())
