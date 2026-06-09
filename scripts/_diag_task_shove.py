"""RCA: pin the EXACT mechanism by which TASK objects are displaced during the
reset settle in heavy ``*,robot`` subsets (cream_cheese/moka_pot/wine_bottle).

Approach (decisive, subprocess-isolated): build the env (env.reset runs the real
settle), then RECONSTRUCT the pre-settle geometric state exactly:
  * arm joints  -> the applied perturbed init qpos (env._sim._applied_robot_init_qpos)
  * each task object free joint -> its Scenic-INJECTED pose (from assert_pose_tolerance)
  * mjdata.ctrl left as-is (unchanged by mj_step; same vector the settle used)
then step ``mujoco.mj_step`` 50x (the same count as simulator.setup) while tracing:
  * arm joint qpos drift per step (does the arm travel away from the perturbed pose?)
  * each task object xy displacement per step
  * the FIRST step where an object's |dxy| jumps, and the contacting geom pair
    (robot/gripper geom <-> that object geom) at that step.

This separates the four hypotheses:
  (a) perturbed arm SWEEPS the object (robot geom contacts object, arm drifting)
  (b) fixture-penetration (object geom contacts a FIXTURE geom, no robot involved)
  (c) object<->object
  (d) clearance graph not guarding task objects vs the perturbed qpos (initial overlap)
"""

from __future__ import annotations

import random
import sys

import numpy as np

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

CONDITIONS = [
    ("libero_90/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl", "position,robot"),
    ("libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl", "position,robot"),
    ("libero_goal/put_the_wine_bottle_on_the_rack.bddl", "position,robot"),
]
SEEDS = [0, 1, 2, 3]
N_SETTLE = 50


def geom_name(sim, gid):
    try:
        return sim.model.geom_id2name(gid)
    except Exception:
        return None


def body_of_geom(sim, gid):
    return int(sim.model.geom_bodyid[gid])


def main():
    import mujoco

    for task, subset in CONDITIONS:
        bddl = str(resolve_task_path(task))
        for seed in SEEDS:
            print(f"\n==== {task.split('/')[-1]}  subset={subset}  seed={seed}")
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(seed)
                scn = compile_task_to_scenario(cfg, subset)
                scene, _ = scn.generate(maxIterations=6000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception as exc:
                print(f"  build/reset fail {type(exc).__name__}: {exc}")
                continue

            sim = env._sim.libero_env.env.sim
            mjmodel = sim.model._model
            mjdata = sim.data._data
            applied = getattr(env._sim, "_applied_robot_init_qpos", None)
            if applied is None:
                print("  (no robot perturbation applied — skipping)")
                env.close()
                continue

            # robot arm joint qpos addresses
            robot = env._sim.libero_env.env.robots[0]
            jidx = np.asarray(getattr(robot, "_ref_joint_pos_indexes", ()), dtype=int)

            # robot/gripper geom ids
            robot_geoms = set()
            for g in range(sim.model.ngeom):
                bid = body_of_geom(sim, g)
                bn = sim.model.body_id2name(bid)
                if bn and (bn.startswith("robot0_") or bn.startswith("gripper0_")):
                    robot_geoms.add(g)

            # task objects: scenic-injected pose + settled, body id + free-joint qadr
            objs = {}
            for o in _iter_scene_objects(scene):
                if is_scene_fixture(o):
                    continue
                nm = resolve_object_name(o)
                if not nm or str(nm).startswith("distractor_"):
                    continue
                try:
                    st = _env_get_object(env, nm)
                    res = assert_pose_tolerance(o, st)
                except Exception:
                    continue
                p = res.payload
                sp = p.get("scenic_position")
                ep = p.get("env_position")
                if sp is None or ep is None:
                    continue
                inj = np.array([float(sp[0]), float(sp[1]), float(sp[2])])
                settled = np.array([float(ep[0]), float(ep[1]), float(ep[2])])
                # body + free joint
                bid = None
                for b in range(sim.model.nbody):
                    bn = sim.model.body_id2name(b)
                    if bn and nm in bn:
                        bid = b
                        break
                if bid is None:
                    continue
                jadr_b = int(sim.model.body_jntadr[bid])
                if jadr_b < 0:
                    continue
                qadr = int(sim.model.jnt_qposadr[jadr_b])
                # current full free-joint qpos (pos+quat) to reuse the quat
                quat = np.array(mjdata.qpos[qadr + 3:qadr + 7])
                objs[str(nm)] = dict(bid=bid, qadr=qadr, inj=inj, settled=settled, quat=quat,
                                     geoms={g for g in range(sim.model.ngeom) if body_of_geom(sim, g) == bid})

            if not objs:
                print("  (no task objects resolved)")
                env.close()
                continue

            # ---- reconstruct PRE-settle state: arm perturbed, objects injected ----
            mjdata.qpos[jidx] = applied
            for nm, d in objs.items():
                qadr = d["qadr"]
                mjdata.qpos[qadr:qadr + 3] = d["inj"]
                mjdata.qpos[qadr + 3:qadr + 7] = d["quat"]
            mjdata.qvel[:] = 0.0
            mujoco.mj_forward(mjmodel, mjdata)

            arm0 = np.array(mjdata.qpos[jidx])
            pos0 = {nm: np.array(mjdata.qpos[d["qadr"]:d["qadr"] + 3]) for nm, d in objs.items()}

            # initial robot<->object overlap check (hypothesis d)
            for i in range(int(mjdata.ncon)):
                c = mjdata.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                for nm, d in objs.items():
                    if (g1 in robot_geoms and g2 in d["geoms"]) or (g2 in robot_geoms and g1 in d["geoms"]):
                        rg = g1 if g1 in robot_geoms else g2
                        print(f"  [t=0 INITIAL-OVERLAP robot<->{nm}] {geom_name(sim,rg)} dist={c.dist*1000:+.1f}mm  (hyp-d)")

            first_move = {nm: None for nm in objs}
            contact_log = {nm: [] for nm in objs}
            for t in range(1, N_SETTLE + 1):
                mujoco.mj_step(mjmodel, mjdata)
                # contacts this step
                for i in range(int(mjdata.ncon)):
                    c = mjdata.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    b1, b2 = body_of_geom(sim, g1), body_of_geom(sim, g2)
                    n1, n2 = sim.model.body_id2name(b1), sim.model.body_id2name(b2)
                    for nm, d in objs.items():
                        if g1 in d["geoms"] or g2 in d["geoms"]:
                            other = g2 if g1 in d["geoms"] else g1
                            ob = body_of_geom(sim, other)
                            obn = sim.model.body_id2name(ob)
                            kind = "ROBOT" if other in robot_geoms else (
                                "OBJ" if other in {gg for dd in objs.values() for gg in dd["geoms"]} else "FIXTURE/WORLD")
                            contact_log[nm].append((t, kind, obn, geom_name(sim, other), c.dist))
                # displacement
                for nm, d in objs.items():
                    cur = np.array(mjdata.qpos[d["qadr"]:d["qadr"] + 3])
                    dxy = float(np.hypot(cur[0] - pos0[nm][0], cur[1] - pos0[nm][1])) * 1000
                    if first_move[nm] is None and dxy > 5.0:
                        first_move[nm] = (t, dxy)

            armf = np.array(mjdata.qpos[jidx])
            arm_drift = float(np.max(np.abs(armf - arm0)))
            print(f"  arm joint drift over {N_SETTLE} settle steps: max|dq|={arm_drift:.4f} rad")
            for nm, d in objs.items():
                cur = np.array(mjdata.qpos[d["qadr"]:d["qadr"] + 3])
                dxy = float(np.hypot(cur[0] - pos0[nm][0], cur[1] - pos0[nm][1])) * 1000
                dz = (cur[2] - pos0[nm][2]) * 1000
                fm = first_move[nm]
                fm_s = f"first>5mm @step{fm[0]}" if fm else "never>5mm"
                print(f"  {nm}: settle_xy={dxy:.1f}mm z={dz:+.1f}mm  {fm_s}")
                # summarize who it contacted, by kind, earliest step
                seen = {}
                for (t, kind, obn, gn, dist) in contact_log[nm]:
                    key = (kind, obn)
                    if key not in seen:
                        seen[key] = (t, gn, dist)
                for (kind, obn), (t, gn, dist) in sorted(seen.items(), key=lambda kv: kv[1][0]):
                    print(f"       contact[{kind}] {obn}/{gn} first@step{t} dist={dist*1000:+.1f}mm")
            env.close()


if __name__ == "__main__":
    sys.exit(main())
