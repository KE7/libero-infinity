"""Trace the REAL env.reset settle for moka_pot (kitchen, position,robot) by
wrapping mujoco.mj_step. Records, per step, moka xy displacement and any
robot/gripper<->moka contact, to pin why the in-frame (base offset 0) clearance
graph still lets moka get shoved ~210mm."""
from __future__ import annotations
import random, sys
import numpy as np
import mujoco

TASK = "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl"
SUBSET = "position,robot"

_state = {"rows": [], "prev": None, "nstep": 0}

def main():
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    orig_step = mujoco.mj_step

    def traced_step(m, d, nstep=1):
        orig_step(m, d, nstep)
        _state["nstep"] += 1
        # resolve moka body + robot geoms lazily
        try:
            tgt_geoms = []
            moka_qadr = None
            robot_geoms = set()
            for b in range(m.nbody):
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                if bn and "moka_pot_1" in bn and "main" in bn:
                    ja = m.body_jntadr[b]
                    if ja >= 0:
                        moka_qadr = m.jnt_qposadr[ja]
                    for g in range(m.ngeom):
                        if m.geom_bodyid[g] == b:
                            tgt_geoms.append(g)
            if moka_qadr is None:
                return
            for g in range(m.ngeom):
                bb = m.geom_bodyid[g]
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bb)
                if bn and (bn.startswith("robot0_") or bn.startswith("gripper0_")):
                    robot_geoms.add(g)
            pos = np.array(d.qpos[moka_qadr:moka_qadr+3])
            if _state["prev"] is None:
                _state["prev"] = pos.copy()
            dxy = float(np.hypot(*(pos[:2]-_state["prev"][:2])))*1000
            # robot<->moka contacts
            hits = []
            tg = set(tgt_geoms)
            for i in range(int(d.ncon)):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in tg and g2 in robot_geoms) or (g2 in tg and g1 in robot_geoms):
                    rg = g1 if g1 in robot_geoms else g2
                    hits.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[rg]), c.dist))
            _state["rows"].append((_state["nstep"], pos.copy(), dxy, hits))
        except Exception as e:
            pass

    mujoco.mj_step = traced_step

    for seed in [0,1,2,3,4,5,6,7]:
        _state["rows"].clear(); _state["prev"]=None; _state["nstep"]=0
        bddl=str(resolve_task_path(TASK)); cfg=TaskConfig.from_bddl(bddl); random.seed(seed)
        scn=compile_task_to_scenario(cfg,SUBSET)
        try: scene,_=scn.generate(maxIterations=6000)
        except Exception as e:
            print(f"seed{seed}: gen fail {type(e).__name__}"); continue
        env=make_env(scene,bddl_path=bddl); env.reset()
        rows=_state["rows"]
        if not rows: print(f"seed{seed}: no moka rows"); env.close(); continue
        start=rows[0][1]; end=rows[-1][1]
        net=float(np.hypot(*(end[:2]-start[:2])))*1000
        # earliest robot contact + biggest single-step jump
        firstcontact=next(((n,h) for (n,p,dxy,h) in rows if h), None)
        bigjump=max(rows, key=lambda r:r[2])
        print(f"seed{seed}: moka NET xy={net:.0f}mm start_z={start[2]:.3f} end_z={end[2]:.3f} nsteps={len(rows)}")
        if firstcontact:
            print(f"   first robot<->moka contact @step{firstcontact[0]}: {[(n,round(d*1000,1)) for n,d in firstcontact[1]]}")
        else:
            print(f"   NO robot<->moka contact during settle (shove is NOT direct robot contact)")
        env.close()

    mujoco.mj_step = orig_step

if __name__=="__main__":
    sys.exit(main())
