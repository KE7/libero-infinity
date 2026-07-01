"""Probe the realized OPEN-drawer top_side envelope: for many seeds, capture the
top_side bowl's sampled (local_x, local_y) and its true 50-step-STABLE rest, then
report whether a single scalar, or an xy-split, or neither, closes it."""

import argparse

import numpy as np


def _settle(mujoco, mm, md, sim, jn, x, y, z, n):
    q = sim.data.get_joint_qpos(jn).copy()
    q[0], q[1], q[2] = x, y, z
    sim.data.set_joint_qpos(jn, q)
    md.qvel[:] = 0
    mujoco.mj_forward(mm, md)
    for _ in range(n):
        mujoco.mj_step(mm, md)
    md.qvel[:] = 0
    mujoco.mj_forward(mm, md)
    s = sim.data.get_joint_qpos(jn)
    return float(s[2]), float(s[0]), float(s[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        default="libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
    )
    ap.add_argument("--seeds", type=int, default=16)
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
    recs = []
    for seed in range(args.seeds):
        cfg = TaskConfig.from_bddl(bddl)
        random.seed(seed)
        np.random.seed(seed)
        build_semantic_scene_graph(cfg)
        scn = compile_task_to_scenario(cfg, "position")
        scene, _ = scn.generate(maxIterations=20000)
        env = make_env(scene, bddl_path=bddl)
        env.reset()
        es = getattr(env, "realized_scene", None) or scene
        sim = env._sim.libero_env.env.sim
        mm, md = sim.model._model, sim.data._data
        cab = np.array(sim.data.body_xpos[sim.model.body_name2id("wooden_cabinet_1_main")][:3])
        best = None
        for o in _iter_scene_objects(es):
            if (getattr(o, "support_surface_class", "") or "") != "wooden_cabinet":
                continue
            nm = resolve_object_name(o) or ""
            pos = getattr(o, "position", None)
            sz = float(pos[2]) if pos is not None else 0.0
            if pos is not None and (best is None or sz > best[3]):
                best = (nm, float(pos[0]), float(pos[1]), sz)
        if best is None:
            env.close()
            continue
        nm, sx, sy, sz = best
        jn = f"{nm}_joint0"
        # true stable rest: long settle from above at sampled xy
        rz, rx, ry = _settle(mujoco, mm, md, sim, jn, sx, sy, 1.30, 400)
        # 50-step stability from that rest at sampled xy
        sz2, sx2, sy2 = _settle(mujoco, mm, md, sim, jn, sx, sy, rz, 50)
        stab_dz = (sz2 - rz) * 1000
        stab_xy = float(np.hypot(sx2 - sx, sy2 - sy)) * 1000
        stable = abs(stab_dz) < 5 and stab_xy < 5
        recs.append((seed, sx - cab[0], sy - cab[1], rz, stable, stab_dz, stab_xy))
        print(
            f"seed={seed:2d} lx={sx-cab[0]:+.3f} ly={sy-cab[1]:+.3f} rest={rz:.4f} "
            f"stable={stable} (dz={stab_dz:+.1f}mm xy={stab_xy:.1f}mm)"
        )
        env.close()

    rests = [r[3] for r in recs]
    lo = [r for r in recs if r[3] < 1.0]
    hi = [r for r in recs if r[3] >= 1.0]
    print(f"\nn={len(recs)}  table-mode(<1.0)={len(lo)}  drawer-mode(>=1.0)={len(hi)}")
    print(f"stable count = {sum(1 for r in recs if r[4])}/{len(recs)}")
    if rests:
        print(f"rest range [{min(rests):.4f},{max(rests):.4f}]")


if __name__ == "__main__":
    main()
