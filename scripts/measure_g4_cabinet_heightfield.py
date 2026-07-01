"""SETTLE-FROM-ABOVE support-surface (heightfield) measurement for the g4 §6
cabinet residual: ``akita_black_bowl`` on ``wooden_cabinet_1_top_side``.

Why a new harness (RCA g4_fixed_point_settle.md §2)
--------------------------------------------------
The scalar iterated fixed point (``measure_g4_fixture_fixedpoint.py``) TUNNELS on
the cabinet: re-injecting at the settled pose compounds an initial penetration of
the cabinet's THIN top collision panel, so ``f(z)=settle50(z)`` converges to a
spurious ``z*≈0.898`` that is 228 mm BELOW the true rest on the solid top (1.126).
A single *continuous* settle FROM CLEARLY ABOVE does not penetrate the panel, so
the ONLY trustworthy rest measurement is SETTLE-FROM-ABOVE: inject the bowl once
at ``z0`` well above the cabinet top and settle without re-injection.

The cabinet top is ALSO not a single scalar: the realized ``top_side`` region sits
off the collision-less cabinet body, so the settled rest is xy/drawer-state
dependent (tri-modal across the 4 tasks that share the ``akita|wooden_cabinet``
key). This harness measures ``h(local_x, local_y, drawer_state) -> rest z`` by
scanning the realized placement envelope under position perturbation, in BOTH
drawer states, and reports per-state determinism (cross-seed spread) so the
operator can decide whether a per-state scalar or a full xy heightfield is needed.

Nothing is written. The operator reviews the numbers and populates the additive
``data/fixture_heightfields.json`` support-surface table.
"""

import argparse
import collections
import json
import random

import numpy as np

# The cabinet tasks. Task ``next_to_the_cookie_box`` has NO cabinet-top placement
# (its akita bowls are on the table / stove) and is intentionally omitted; it is a
# control that must stay byte-identical (no heightfield touches it).
_CABINET_TASKS = [
    "libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
]


def _settle_from_above(mujoco, mjmodel, mjdata, sim, joint_name, x, y, z0, n_steps):
    """Inject the free-joint object ONCE at (x, y, z0) and settle ``n_steps`` with
    the production qvel-zero sequence, WITHOUT re-injection. Returns (z, x, y).

    This is the tunnel-free rest: a single continuous fall onto the solid support,
    matching the physics the validation settle applies from the emitted spawn z."""
    qpos = sim.data.get_joint_qpos(joint_name).copy()
    qpos[0], qpos[1], qpos[2] = x, y, z0
    sim.data.set_joint_qpos(joint_name, qpos)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    for _ in range(n_steps):
        mujoco.mj_step(mjmodel, mjdata)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    s = sim.data.get_joint_qpos(joint_name)
    return float(s[2]), float(s[0]), float(s[1])


def _settle_in_place(mujoco, mjmodel, mjdata, sim, joint_name, z_inject, n_steps):
    """Production-style settle: inject at the CURRENT xy but z=z_inject, settle
    ``n_steps``. Returns (z, x, y). Used to reproduce what pose_tolerance sees when
    the renderer emits ``z_inject`` at the sampled xy."""
    qpos = sim.data.get_joint_qpos(joint_name).copy()
    qpos[2] = z_inject
    sim.data.set_joint_qpos(joint_name, qpos)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    for _ in range(n_steps):
        mujoco.mj_step(mjmodel, mjdata)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    s = sim.data.get_joint_qpos(joint_name)
    return float(s[2]), float(s[0]), float(s[1])


def _cabinet_drawer_qpos(sim):
    """Return {joint_name: qpos} for every wooden_cabinet drawer-slider joint."""
    out = {}
    model = sim.model
    njnt = model._model.njnt
    for j in range(njnt):
        try:
            nm = model.joint_id2name(j)
        except Exception:
            nm = None
        if not nm or "cabinet" not in nm.lower():
            continue
        try:
            out[nm] = float(np.ravel(sim.data.get_joint_qpos(nm))[0])
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--subset", default="position")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--steps", type=int, default=50, help="validation settle steps")
    ap.add_argument("--long-steps", type=int, default=400, help="stability-confirm settle steps")
    ap.add_argument("--z0", type=float, default=1.30, help="settle-from-above inject height (m)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    import mujoco

    from libero_infinity.asset_metadata import (
        TABLE_SURFACE_Z,
    )
    from libero_infinity.compiler import build_semantic_scene_graph, compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.renderer.scenic_renderer import _arena_surface_z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import resolve_object_name
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    tasks = list(_CABINET_TASKS)
    if args.tasks:
        tasks = [t for t in args.tasks.split(",") if t.strip()]

    all_records = []
    # aggregate by (class, fixture, drawer_state, relation)
    agg = collections.defaultdict(list)

    for task in tasks:
        bddl = str(resolve_task_path(task))
        for seed in range(args.seeds):
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(seed)
                np.random.seed(seed)
                graph = build_semantic_scene_graph(cfg)
                arena_z = _arena_surface_z(graph)
                scn = compile_task_to_scenario(cfg, args.subset)
                scene, _ = scn.generate(maxIterations=20000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception as exc:  # noqa: BLE001
                print(f"# build failed {task} [seed {seed}]: {exc}")
                continue

            es = getattr(env, "realized_scene", None) or scene
            sim = env._sim.libero_env.env.sim
            mjmodel = sim.model._model
            mjdata = sim.data._data

            drawer_q = _cabinet_drawer_qpos(sim)
            # wooden_cabinet drawer OPEN band is negative (see scene_semantics
            # _ARTICULATION_RANGES: Open (-0.16,-0.14), Close (0,0.005)).
            drawer_open_val = min(drawer_q.values()) if drawer_q else 0.0
            drawer_state = "open" if drawer_open_val < -0.05 else "closed"

            # cabinet body xy
            cab_xy = None
            try:
                cab_bid = sim.model.body_name2id("wooden_cabinet_1_main")
                cab_xy = np.array(sim.data.body_xpos[cab_bid][:3], dtype=float)
            except Exception:
                try:
                    cab_bid = sim.model.body_name2id("wooden_cabinet_1")
                    cab_xy = np.array(sim.data.body_xpos[cab_bid][:3], dtype=float)
                except Exception:
                    cab_xy = np.array([0.0, 0.0, 0.0])

            for o in _iter_scene_objects(es):
                if getattr(o, "graspable", True) is False:
                    continue
                sc = getattr(o, "support_surface_class", "") or ""
                if sc != "wooden_cabinet":
                    continue
                nm = resolve_object_name(o) or ""
                cc = getattr(o, "asset_class", "") or ""
                if not nm or not cc:
                    continue
                joint_name = f"{nm}_joint0"
                try:
                    qpos0 = sim.data.get_joint_qpos(joint_name).copy()
                except Exception:
                    continue
                # scenic-emitted pose (what pose_tolerance compares against)
                scenic_pos = getattr(o, "position", None)
                sx = float(scenic_pos[0]) if scenic_pos is not None else float(qpos0[0])
                sy = float(scenic_pos[1]) if scenic_pos is not None else float(qpos0[1])
                sz = float(scenic_pos[2]) if scenic_pos is not None else float(qpos0[2])
                # today's production settle from the emitted z (what fails today)
                prod_z, prod_x, prod_y = _settle_in_place(
                    mujoco, mjmodel, mjdata, sim, joint_name, sz, args.steps
                )
                prod_dz_mm = (prod_z - sz) * 1000.0
                prod_xy_mm = float(np.hypot(prod_x - sx, prod_y - sy)) * 1000.0
                # settle-from-above true rest at the SAMPLED xy (50 and long)
                fa_z, fa_x, fa_y = _settle_from_above(
                    mujoco, mjmodel, mjdata, sim, joint_name, sx, sy, args.z0, args.steps
                )
                fa_z_long, fa_x_long, fa_y_long = _settle_from_above(
                    mujoco, mjmodel, mjdata, sim, joint_name, sx, sy, args.z0, args.long_steps
                )
                # STABILITY: emit the true (long) rest at the SAMPLED xy, then run
                # the 50-step validation settle — this is exactly what
                # pose_tolerance sees if the renderer emits ``fa_z_long``.
                stab_z, stab_x, stab_y = _settle_from_above(
                    mujoco, mjmodel, mjdata, sim, joint_name, sx, sy, fa_z_long, args.steps
                )
                stab_dz_mm = (stab_z - fa_z_long) * 1000.0
                stab_xy_mm = float(np.hypot(stab_x - sx, stab_y - sy)) * 1000.0

                # relation: in-drawer bowls are emitted LOW (~0.82, contained);
                # top_side bowls are emitted high (absolute on-surface z ~1.23).
                relation = "inside" if sz < 1.0 else "on_surface"
                rec = {
                    "task": task.split("/")[-1],
                    "seed": seed,
                    "name": nm,
                    "class": cc,
                    "fixture": sc,
                    "relation": relation,
                    "drawer_state": drawer_state,
                    "drawer_qpos": round(drawer_open_val, 4),
                    "arena_z": round(arena_z, 5),
                    "scenic_z": round(sz, 5),
                    "cab_x": round(float(cab_xy[0]), 5),
                    "cab_y": round(float(cab_xy[1]), 5),
                    "local_x": round(sx - float(cab_xy[0]), 5),
                    "local_y": round(sy - float(cab_xy[1]), 5),
                    "sample_x": round(sx, 5),
                    "sample_y": round(sy, 5),
                    "prod_settle_z": round(prod_z, 5),
                    "prod_dz_mm": round(prod_dz_mm, 2),
                    "prod_xy_mm": round(prod_xy_mm, 2),
                    "fromabove_z50": round(fa_z, 5),
                    "fromabove_z_long": round(fa_z_long, 5),
                    "fromabove_xy_drift_mm": round(
                        float(np.hypot(fa_x - sx, fa_y - sy)) * 1000.0, 2
                    ),
                    "fa_50_vs_long_mm": round((fa_z - fa_z_long) * 1000.0, 2),
                    "stab_dz_mm": round(stab_dz_mm, 2),
                    "stab_xy_mm": round(stab_xy_mm, 2),
                    "clear_fromabove_vs_arena": round(fa_z - arena_z, 5),
                    "clear_fromabove_vs_tablez": round(fa_z - TABLE_SURFACE_Z, 5),
                }
                all_records.append(rec)
                agg[(cc, sc, relation, drawer_state)].append(rec)
                print(
                    f"[{task.split('/')[-1][:34]:34s}] s{seed} {nm:18s} "
                    f"{relation:10s} drawer={drawer_state:6s} scz={sz:.4f} "
                    f"long={fa_z_long:.4f} clr={fa_z_long-arena_z:.4f} "
                    f"STABLE(dz={stab_dz_mm:+.1f},xy={stab_xy_mm:.1f})"
                )
            env.close()

    print("\n" + "=" * 78)
    print("SETTLE-FROM-ABOVE SUPPORT SURFACE SUMMARY  (class|fixture|drawer_state)")
    print("=" * 78)
    for key, recs in sorted(agg.items()):
        cc, fx, rel, ds = key
        zl = [r["fromabove_z_long"] for r in recs]
        clears = [r["clear_fromabove_vs_arena"] for r in recs]
        spread_long = (max(zl) - min(zl)) * 1000
        n_stable = sum(1 for r in recs if abs(r["stab_dz_mm"]) < 5 and r["stab_xy_mm"] < 5)
        det = (
            "DETERMINISTIC+STABLE"
            if spread_long <= 5 and n_stable == len(recs)
            else f"XY/METASTABLE(spread={spread_long:.0f}mm,stable={n_stable}/{len(recs)})"
        )
        cl = sum(clears) / len(clears)
        print(
            f"  {cc}|{fx}|{rel}|{ds}: n={len(recs)} "
            f"long=[{min(zl):.4f},{max(zl):.4f}] mean={sum(zl)/len(zl):.4f} "
            f"clear_vs_arena~{cl:.4f} stable={n_stable}/{len(recs)} -> {det}"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\nwrote {len(all_records)} records to {args.json_out}")


if __name__ == "__main__":
    main()
