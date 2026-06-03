"""t=0 (post-reset, pre-settle) contact + penetration audit for the launching
TABLE distractor in the cabinet robot,distractor scene. Identifies WHAT the
distractor is touching/penetrating that drives the chaotic launch."""
from __future__ import annotations
import random
import numpy as np


def main():
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path("libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl"))
    for rep in range(5):
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
                continue
            bid = None
            for c in (nm, nm + "_main"):
                try:
                    bid = m.body_name2id(c); break
                except Exception:
                    pass
            if bid is None:
                continue
            dgeoms = frozenset(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)
            bp = np.array(d.body_xpos[bid])
            cons = []
            for c in range(int(d.ncon)):
                con = d.contact[c]
                g1, g2 = int(con.geom1), int(con.geom2)
                if (g1 in dgeoms) == (g2 in dgeoms):
                    continue
                other = g2 if g1 in dgeoms else g1
                on = m.geom_id2name(other) or f"geom{other}"
                cons.append(f"{on}(pen={con.dist*1000:+.1f}mm)")
            cls = getattr(o, "asset_class", "")
            print(f"rep{rep} {nm} {cls:14} t0_pos=({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:.3f}) "
                  f"ncon={len(cons)} contacts=[{', '.join(cons[:6])}]", flush=True)
        env.close()


if __name__ == "__main__":
    main()
