"""Confirm the mechanism: for the failing (task,seed) samples, compare the
renderer's LINEARIZED link world-origin prediction (c0 + J@dq, what the require
graph evaluates) against the TRUE FK link world origin (mjdata.body_xpos), and
evaluate the SAT clearance clause under both for the object that gets shoved.

If the linear model says CLEAR while the true geometry OVERLAPS, the residual is
the bug: the require graph admits samples the simulator must shove.
"""
from __future__ import annotations
import random, sys
import numpy as np
from libero_infinity.compiler import compile_task_to_scenario
from libero_infinity.gym_env import make_env
from libero_infinity.task_config import TaskConfig
from libero_infinity.robot_metadata import get_robot_footprint
from libero_infinity.asset_registry import get_dimensions
from libero_infinity.validation.invariants._scene_view import is_scene_fixture, resolve_object_name
from libero_infinity.validation.invariants.consistency import _env_get_object, assert_pose_tolerance
from libero_infinity.validation.invariants.domain import _iter_scene_objects
from libero_infinity.validation.sweep import resolve_task_path

COND = [
    ("libero_90/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl", "position,robot", 2),
    ("libero_90/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl", "position,robot", 0),
]

def main():
    fp = get_robot_footprint("Panda")
    canon = np.array(fp.canonical_qpos)
    for task, subset, seed in COND:
        bddl = str(resolve_task_path(task))
        print(f"\n==== {task.split('/')[-1]} seed={seed}")
        cfg = TaskConfig.from_bddl(bddl)
        random.seed(seed)
        scn = compile_task_to_scenario(cfg, subset)
        scene, _ = scn.generate(maxIterations=6000)
        env = make_env(scene, bddl_path=bddl); env.reset()
        sim = env._sim.libero_env.env.sim
        applied = np.array(env._sim._applied_robot_init_qpos)
        dq = applied - canon
        print(f"  ||dq||={np.linalg.norm(dq):.4f} rad  max|dq|={np.max(np.abs(dq)):.4f}")
        # injected object poses
        objs = {}
        for o in _iter_scene_objects(scene):
            if is_scene_fixture(o): continue
            nm = resolve_object_name(o)
            if not nm or str(nm).startswith("distractor_"): continue
            try:
                res = assert_pose_tolerance(o, _env_get_object(env, nm))
            except Exception: continue
            sp = res.payload.get("scenic_position")
            if sp is None: continue
            cls = getattr(o, "asset_class", None) or nm
            objs[str(nm)] = (np.array([float(sp[0]),float(sp[1]),float(sp[2])]), get_dimensions(cls))
        # true link origins at applied qpos
        robot = env._sim.libero_env.env.robots[0]
        jidx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        import mujoco
        mjm, mjd = sim.model._model, sim.data._data
        mjd.qpos[jidx] = applied; mjd.qvel[:]=0; mujoco.mj_forward(mjm,mjd)
        for lk in fp.active_links():
            bid = sim.model.body_name2id(lk.name) if lk.name in [sim.model.body_id2name(b) for b in range(sim.model.nbody)] else None
            true_o = None
            for b in range(sim.model.nbody):
                if sim.model.body_id2name(b)==lk.name:
                    true_o = np.array(sim.data.body_xpos[b]); break
            # linearized prediction
            lin = np.array([
                lk.x0 + sum(lk.jx[k]*dq[k] for k in range(min(len(dq),len(lk.jx)))),
                lk.y0 + sum(lk.jy[k]*dq[k] for k in range(min(len(dq),len(lk.jy)))),
                lk.z0 + sum(lk.jz[k]*dq[k] for k in range(min(len(dq),len(lk.jz)))),
            ])
            if true_o is None: continue
            resid = np.linalg.norm(lin - true_o)
            # find worst object (min SAT margin) under true vs lin
            worst=None
            for nm,(op,od) in objs.items():
                thx,thy,thz = od[0]/2,od[1]/2,od[2]/2
                dxh,dyh,dzh = lk.hx+thx, lk.hy+thy, lk.hz+thz
                def sat(p):  # positive = separated (clear); negative = overlap on all axes
                    return max(abs(p[0]-op[0])-dxh, abs(p[1]-op[1])-dyh, abs(p[2]-op[2])-dzh)
                m_lin, m_true = sat(lin), sat(true_o)
                if worst is None or m_true < worst[3]:
                    worst=(nm,m_lin,resid,m_true)
            # PROPOSED FIX preview: shift link x/y by base offset (+0.15,0) for
            # living_room arena; recompute SAT with corrected x/y, z unchanged.
            BASE_OFF = np.array([0.15, 0.0, 0.0])  # living_room base - ref base (x,y only)
            lin_fix = lin + np.array([BASE_OFF[0], BASE_OFF[1], 0.0])
            if worst and (worst[3] < 0 or abs(worst[1]-worst[3])>0.03):
                nm,m_lin,resid,m_true = worst
                op,od = objs[nm]; thx,thy,thz = od[0]/2,od[1]/2,od[2]/2
                dxh,dyh,dzh = lk.hx+thx, lk.hy+thy, lk.hz+thz
                m_fix = max(abs(lin_fix[0]-op[0])-dxh, abs(lin_fix[1]-op[1])-dyh, abs(lin_fix[2]-op[2])-dzh)
                tag = "  <<< LIN=CLEAR but TRUE=OVERLAP" if (m_lin>=0 and m_true<0) else ""
                fixtag = " FIX->OVERLAP(caught)" if m_fix<0 else " FIX->still-clear(MISS)"
                print(f"  {lk.name:>22} lin_resid={resid*1000:6.1f}mm  obj={nm:>16} SAT_lin={m_lin*1000:+7.1f}mm SAT_fix={m_fix*1000:+7.1f}mm SAT_true={m_true*1000:+7.1f}mm{tag}{fixtag if m_true<0 else ''}")
        env.close()

if __name__=="__main__":
    sys.exit(main())
