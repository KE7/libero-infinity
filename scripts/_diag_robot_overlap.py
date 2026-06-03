"""Confirm whether the distractor_0 TABLE launch is a robot-arm overlap during
settle: for the cabinet robot,distractor s0 scene, report distractor_0's nearest
robot geom at reset (t=0, pre-settle) and after settle."""
from __future__ import annotations
import random
import numpy as np


def main():
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path("libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl"))
    for rep in range(4):
        cfg = TaskConfig.from_bddl(bddl)
        random.seed(0)
        scn = compile_task_to_scenario(cfg, ",".join(("robot", "distractor")))
        scene, _ = scn.generate(maxIterations=4000)
        env = make_env(scene, bddl_path=bddl)
        env.reset()
        sim = env._sim.libero_env.env.sim
        active = getattr(env._sim, "_active_distractor_names", set())
        m, d = sim.model, sim.data
        for nm in sorted(active):
            o = next((x for x in scene.objects if getattr(x, "libero_name", "") == nm), None)
            if o is None or (getattr(o, "support_parent_name", "") or ""):
                continue  # only TABLE distractors
            bid = None
            for c in (nm, nm + "_main"):
                try:
                    bid = m.body_name2id(c); break
                except Exception:
                    pass
            if bid is None:
                continue
            dbody = np.array(d.body_xpos[bid])
            # nearest robot/gripper geom to the distractor body
            best = (1e9, "", None)
            for gid in range(m.ngeom):
                gn = m.geom_id2name(gid) or ""
                gl = gn.lower()
                if not any(t in gl for t in ("robot", "gripper", "link", "finger", "hand")):
                    continue
                gp = np.array(d.geom_xpos[gid])
                dist = float(np.linalg.norm(gp - dbody))
                if dist < best[0]:
                    best = (dist, gn, gp)
            cls = getattr(o, "asset_class", "")
            print(f"rep{rep} {nm} {cls:14} body=({dbody[0]:+.3f},{dbody[1]:+.3f},{dbody[2]:.3f}) "
                  f"nearest_robot_geom={best[1]} dist={best[0]*1000:.0f}mm "
                  f"geom_z={best[2][2]:.3f}" if best[2] is not None else f"rep{rep} {nm}: no robot geom", flush=True)
        env.close()


if __name__ == "__main__":
    main()
