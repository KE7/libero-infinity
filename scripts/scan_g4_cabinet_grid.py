"""Dense settle-from-above heightfield scan of the wooden_cabinet top_side support.

Fix ONE realized scene per drawer state, then teleport the top_side akita bowl to
each point of a dense world-xy grid, settle-from-above (long) to the stable rest,
and record: rest z, and the 50-step-in-place stability (dz, xy drift) that
pose_tolerance would see if the renderer emitted that rest z at that xy.

This maps h(local_x, local_y, drawer_state) and, crucially, whether each cell is a
STABLE rest (50-step settle from the measured rest stays <5mm in z AND xy) — the
condition an xy-dependent emission needs to make pose_tolerance pass.
"""

import argparse
import json

import numpy as np


def _settle(mujoco, mjmodel, mjdata, sim, jn, x, y, z, n):
    q = sim.data.get_joint_qpos(jn).copy()
    q[0], q[1], q[2] = x, y, z
    sim.data.set_joint_qpos(jn, q)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    for _ in range(n):
        mujoco.mj_step(mjmodel, mjdata)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    s = sim.data.get_joint_qpos(jn)
    return float(s[2]), float(s[0]), float(s[1])


def _settle_inplace(mujoco, mjmodel, mjdata, sim, jn, z, n):
    q = sim.data.get_joint_qpos(jn).copy()
    q[2] = z
    sim.data.set_joint_qpos(jn, q)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    for _ in range(n):
        mujoco.mj_step(mjmodel, mjdata)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    s = sim.data.get_joint_qpos(jn)
    return float(s[2]), float(s[0]), float(s[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nx", type=int, default=9)
    ap.add_argument("--ny", type=int, default=11)
    ap.add_argument("--lx-lo", type=float, default=-0.13)
    ap.add_argument("--lx-hi", type=float, default=0.07)
    ap.add_argument("--ly-lo", type=float, default=0.16)
    ap.add_argument("--ly-hi", type=float, default=0.36)
    ap.add_argument("--z0", type=float, default=1.30)
    ap.add_argument("--long", type=int, default=400)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    import random

    import mujoco

    from libero_infinity.compiler import build_semantic_scene_graph, compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import resolve_object_name
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path(args.task))
    cfg = TaskConfig.from_bddl(bddl)
    random.seed(args.seed)
    np.random.seed(args.seed)
    build_semantic_scene_graph(cfg)
    scn = compile_task_to_scenario(cfg, "position")
    scene, _ = scn.generate(maxIterations=20000)
    env = make_env(scene, bddl_path=bddl)
    env.reset()
    es = getattr(env, "realized_scene", None) or scene
    sim = env._sim.libero_env.env.sim
    mjmodel = sim.model._model
    mjdata = sim.data._data

    # drawer state
    dq = None
    try:
        dq = float(np.ravel(sim.data.get_joint_qpos("wooden_cabinet_1_top_level"))[0])
    except Exception:
        pass
    drawer_state = "open" if (dq is not None and dq < -0.05) else "closed"

    cab_bid = sim.model.body_name2id("wooden_cabinet_1_main")
    cab = np.array(sim.data.body_xpos[cab_bid][:3], dtype=float)

    # pick the top_side bowl = akita on wooden_cabinet with the HIGHEST emitted z
    best = None
    for o in _iter_scene_objects(es):
        if (getattr(o, "support_surface_class", "") or "") != "wooden_cabinet":
            continue
        nm = resolve_object_name(o) or ""
        pos = getattr(o, "position", None)
        sz = float(pos[2]) if pos is not None else 0.0
        if nm and (best is None or sz > best[1]):
            best = (nm, sz)
    if best is None:
        print("no top_side bowl found")
        env.close()
        return
    jn = f"{best[0]}_joint0"
    print(
        f"# task={args.task.split('/')[-1]} drawer={drawer_state} (qpos={dq}) "
        f"bowl={best[0]} emit_z={best[1]:.4f} cab=({cab[0]:.4f},{cab[1]:.4f})"
    )
    print("# local_x local_y | rest_z clear_vs_table | stab_dz_mm stab_xy_mm  STABLE?")

    rows = []
    xs = np.linspace(args.lx_lo, args.lx_hi, args.nx)
    ys = np.linspace(args.ly_lo, args.ly_hi, args.ny)
    for ly in ys:
        line = []
        for lx in xs:
            wx, wy = cab[0] + lx, cab[1] + ly
            rz, rx, ry = _settle(mujoco, mjmodel, mjdata, sim, jn, wx, wy, args.z0, args.long)
            # stability: emit rz at the ORIGINAL sampled (wx,wy), 50-step settle
            sz, sx, sy = _settle_inplace(mujoco, mjmodel, mjdata, sim, jn, rz, 50)
            # reset xy for the in-place test: teleport to wx,wy first then drop rz
            # (do a proper in-place: set xy=wx,wy z=rz)
            q = sim.data.get_joint_qpos(jn).copy()
            q[0], q[1], q[2] = wx, wy, rz
            sim.data.set_joint_qpos(jn, q)
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)
            for _ in range(50):
                mujoco.mj_step(mjmodel, mjdata)
            mjdata.qvel[:] = 0
            mujoco.mj_forward(mjmodel, mjdata)
            s = sim.data.get_joint_qpos(jn)
            sdz = (float(s[2]) - rz) * 1000
            sxy = float(np.hypot(float(s[0]) - wx, float(s[1]) - wy)) * 1000
            stable = abs(sdz) < 5 and sxy < 5
            rows.append(
                {
                    "local_x": round(lx, 4),
                    "local_y": round(ly, 4),
                    "rest_z": round(rz, 4),
                    "clear_vs_table": round(rz - 0.82, 4),
                    "stab_dz_mm": round(sdz, 1),
                    "stab_xy_mm": round(sxy, 1),
                    "stable": stable,
                }
            )
            line.append(f"{rz:.3f}{'*' if stable else ' '}")
        print(f"ly={ly:+.3f} " + " ".join(line))
    env.close()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {"drawer_state": drawer_state, "cab": cab.tolist(), "rows": rows}, f, indent=2
            )
        print(f"# wrote {len(rows)} cells to {args.json_out}")


if __name__ == "__main__":
    main()
