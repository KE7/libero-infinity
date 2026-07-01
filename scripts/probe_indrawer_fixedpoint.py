"""Find the inject-in-place fixed point z* for the in-drawer akita bowl.

The settle-from-above rest (1.1264) is NOT the fixed point of the production
50-step inject-in-place settle (which lands ~1.111). pose_tolerance injects at
the EMITTED z, so we need z* with |z* - settle50(inject z*)| < gate. Sweep z0
across a band, settle in place, and report settled z per seed to locate z*.
"""

import argparse
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--zlist", default="1.126,1.118,1.111,1.108,1.105")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    import mujoco  # noqa: F401

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(
        resolve_task_path(
            "libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl"
        )
    )
    zlist = [float(z) for z in args.zlist.split(",")]
    joint = "akita_black_bowl_1_joint0"

    import collections

    agg = collections.defaultdict(list)
    for seed in range(args.seeds):
        cfg = TaskConfig.from_bddl(bddl)
        random.seed(seed)
        np.random.seed(seed)
        scn = compile_task_to_scenario(cfg, "position")
        scene, _ = scn.generate(maxIterations=20000)
        env = make_env(scene, bddl_path=bddl)
        env.reset()
        sim = env._sim.libero_env.env.sim
        mjmodel = sim.model._model
        mjdata = sim.data._data
        # locate the in-drawer bowl free joint (akita_black_bowl_1)
        jn = None
        for cand in (joint, "akita_black_bowl_1_joint", "akita_black_bowl_1"):
            try:
                sim.data.get_joint_qpos(cand)
                jn = cand
                break
            except Exception:
                continue
        if jn is None:
            # brute-force: any joint name containing akita_black_bowl_1
            for j in range(mjmodel.njnt):
                nm = sim.model.joint_id2name(j) or ""
                if "akita_black_bowl_1" in nm:
                    jn = nm
                    break
        base = sim.data.get_joint_qpos(jn).copy()
        x0, y0 = float(base[0]), float(base[1])
        for z0 in zlist:
            qpos = sim.data.get_joint_qpos(jn).copy()
            qpos[0], qpos[1], qpos[2] = x0, y0, z0
            sim.data.set_joint_qpos(jn, qpos)
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)
            for _ in range(args.steps):
                mujoco.mj_step(mjmodel, mjdata)
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)
            s = sim.data.get_joint_qpos(jn)
            agg[z0].append(float(s[2]))
        env.close()

    print(f"joint={jn}  seeds={args.seeds}")
    print(f"{'inject_z':>10} {'settled_mean':>13} {'spread_mm':>10} {'dz_mm':>8}")
    for z0 in zlist:
        s = agg[z0]
        m = sum(s) / len(s)
        spread = (max(s) - min(s)) * 1000
        print(f"{z0:10.4f} {m:13.4f} {spread:10.1f} {(z0-m)*1000:8.1f}")


if __name__ == "__main__":
    main()
