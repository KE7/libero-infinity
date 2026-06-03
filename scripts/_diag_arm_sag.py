"""Measure the robot arm's transient downward sag during the reset settle:
gripper/lower-link z at t=0 (post-reset home) vs its MINIMUM over the settle
steps. This is the unmodeled motion that lets a sagging arm strike a table
distractor cleared only at the home pose."""
from __future__ import annotations
import random
import numpy as np


def main():
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path("libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl"))
    for seed in range(3):
        cfg = TaskConfig.from_bddl(bddl)
        random.seed(seed)
        scn = compile_task_to_scenario(cfg, ",".join(("robot", "distractor")))
        scene, _ = scn.generate(maxIterations=4000)
        env = make_env(scene, bddl_path=bddl)
        env.reset()
        sim = env._sim.libero_env.env.sim
        m, d = sim.model, sim.data
        # lower-link / gripper geoms
        gids = [g for g in range(m.ngeom)
                if any(t in (m.geom_id2name(g) or "").lower()
                       for t in ("finger", "gripper0_hand", "link5", "link6", "link7"))]
        if not gids:
            env.close(); continue
        z0 = {g: float(d.geom_xpos[g][2]) for g in gids}
        zmin = dict(z0)
        qpos0 = d.qpos.copy(); qvel0 = d.qvel.copy()
        for _ in range(80):
            sim.step()
            for g in gids:
                zmin[g] = min(zmin[g], float(d.geom_xpos[g][2]))
        d.qpos[:] = qpos0; d.qvel[:] = qvel0; sim.forward()
        worst = min(gids, key=lambda g: zmin[g] - z0[g])
        print(f"seed{seed}: max sag={1000*(z0[worst]-zmin[worst]):.0f}mm "
              f"({m.geom_id2name(worst)}) z0={z0[worst]:.3f} zmin={zmin[worst]:.3f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
