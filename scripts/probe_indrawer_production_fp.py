"""Production fixed point for the in-drawer akita bowl.

pose_tolerance uses the FULL production reset: inject at the heightfield z H
(while the drawer is CLOSED), open the drawer, then settle 50 steps. We need H
with |H - f(H)| < gate, where f is that production settle. Both renderer and
simulator read the SAME heightfield value, so scenic_z == injected_z == H and
the check is |H - settled|. This probe patches the in-memory heightfield
clearance to each candidate H, runs the real env.reset(), and reads the settled
z of akita_black_bowl_1 across seeds — i.e. it evaluates f(H) on the exact path
pose_tolerance sees.
"""

import argparse
import collections
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--subset", default="position")
    # candidate REST z values H (absolute); clearance = H - 0.82
    ap.add_argument("--hlist", default="1.1264,1.120,1.115,1.111,1.108,1.105,1.100")
    args = ap.parse_args()

    import libero_infinity.asset_metadata as AM
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import resolve_object_name
    from libero_infinity.validation.invariants.consistency import (
        _env_get_object,
        assert_pose_tolerance,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(
        resolve_task_path(
            "libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
        )
    )
    ARENA = 0.82
    hlist = [float(z) for z in args.hlist.split(",")]

    def set_clearance(clear):
        AM.FIXTURE_HEIGHTFIELDS["wooden_cabinet"]["support_rests"]["inside|open"][
            "akita_black_bowl"
        ] = clear

    agg = collections.defaultdict(list)
    scenic_seen = collections.defaultdict(list)
    for H in hlist:
        set_clearance(round(H - ARENA, 4))
        for seed in range(args.seeds):
            cfg = TaskConfig.from_bddl(bddl)
            random.seed(seed)
            np.random.seed(seed)
            scn = compile_task_to_scenario(cfg, args.subset)
            scene, _ = scn.generate(maxIterations=20000)
            env = make_env(scene, bddl_path=bddl)
            env.reset()
            es = getattr(env, "realized_scene", None) or scene
            for o in _iter_scene_objects(es):
                nm = resolve_object_name(o) or ""
                if "akita_black_bowl_1" not in nm:
                    continue
                try:
                    st = _env_get_object(env, nm)
                    res = assert_pose_tolerance(o, st)
                    p = res.payload
                    sp, ep = p.get("scenic_position"), p.get("env_position")
                    if sp and ep:
                        agg[H].append(float(ep[2]))
                        scenic_seen[H].append(float(sp[2]))
                except Exception:
                    pass
            env.close()

    print(f"subset={args.subset} seeds={args.seeds}  (H = injected/emitted REST z)")
    print(
        f"{'H_inject':>10} {'scenic_z':>10} {'settled_mean':>13} {'spread_mm':>10} {'|dz|_mm':>8}"
    )
    for H in hlist:
        s = agg[H]
        if not s:
            print(f"{H:10.4f}  no data")
            continue
        m = sum(s) / len(s)
        spread = (max(s) - min(s)) * 1000
        sc = sum(scenic_seen[H]) / len(scenic_seen[H]) if scenic_seen[H] else float("nan")
        print(f"{H:10.4f} {sc:10.4f} {m:13.4f} {spread:10.1f} {abs(sc-m)*1000:8.1f}")


if __name__ == "__main__":
    main()
