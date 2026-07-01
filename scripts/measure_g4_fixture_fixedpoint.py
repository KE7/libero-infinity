"""Iterated FIXED-POINT (converged-rest) measurement for task-object-on-fixture-top.

Motivation (RCA g4_remaining_arenas.md §3b)
-------------------------------------------
A task object resting on a fixture EXTERIOR top (``white_bowl`` on ``microwave``;
``akita_black_bowl`` on ``wooden_cabinet`` / ``flat_stove``) does NOT reach a
stable rest inside the 50-step validation settle. The analytic on-fixture spawn z
(rule 2 of :func:`asset_metadata.spawn_clearance`) uses the fixture's raw
``top_z`` = geom-AABB top (the HIGHEST collision geom), which for these irregular
fixtures sits well ABOVE the real rest face (the ``flat_stove`` cook_region grate
is visual-only — same mode as #32/#34; the microwave / cabinet tops have a raised
lip / rounded shell). The object is therefore injected too high, falls tens of mm
each 50-step settle, and no SINGLE analytic clearance is a fixed point of the map

    f(z) = body_origin_z after (set object z, zero qvel, 50 * mj_step, zero qvel).

The predecessor observed f(z) ≈ z − 0.049 for white_bowl|microwave over the first
two iterations (no fixed point at 50 steps) and correctly FLAGGED it rather than
band-aiding a single clearance.

What this script does
---------------------
Measures the CONVERGED rest of f by ITERATING it: z_{n+1} = f(z_n), starting from
the production analytic spawn z, until |z_{n+1} − z_n| < ``--tol`` (default 1 mm)
— the object has landed on its true rest face and the 50-step map is now a fixed
point — or ``--max-iter`` is exceeded (non-convergent). A converged z* is a fixed
point of the SAME 50-step settle the g4 validation uses, so emitting scenic_z = z*
makes pose_tolerance pass WITHOUT widening the 5 mm gate.

Roll-off / no-stable-rest detection (the genuine FLAG case): if during the descent
the object's xy drifts past ``--xy-flag`` (it slid off the fixture) or its z falls
to/below the arena table surface (it fell off entirely), the case is reported
NON_CONVERGENT and NOT turned into a clearance — per the no-force guard.

Determinism: the whole descent is repeated for several seeds; the converged z* (and
its clearance-above-TABLE_Z) is reported with the cross-seed spread. A spread > 5 mm
means the rest itself is metastable → also FLAG, do not record.

Nothing is written. The operator reviews the numbers and, for each CONVERGED
deterministic case, adds an ADDITIVE ``"<class>|<fixture>"`` row to
``data/spawn_clearances_variants.json`` (clearance = z* − TABLE_Z), which both the
renderer and the simulator resolve via rule 1 (superseding the analytic rule 2).
"""

import argparse
import collections
import json
import random

import numpy as np

# Representative BDDLs whose :init places a MOVABLE task object On a fixture
# EXTERIOR top/region. The harness auto-discovers WHICH object (and its fixture
# class) from the realized scene's ``support_surface_class`` — so a task where the
# object merely has the fixture as a :goal (starts table-resting) contributes no
# fixture-supported object and is silently ignored. This makes the scan robust to
# init-vs-goal confusion in a raw BDDL grep.
_DEFAULT_TASKS = [
    # --- the three RCA §3b named cases (white_bowl|microwave, akita|wooden_cabinet,
    #     akita|flat_stove); the harness reads the fixture from the realized scene ---
    "libero_90/KITCHEN_SCENE7_put_the_white_bowl_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl",
    # --- other candidate fixture-exterior placements found by a raw corpus scan;
    #     most are :goal (object starts table-resting) so contribute no target ---
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
    "libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl",
    "libero_90/KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet.bddl",
    "libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.bddl",
    "libero_90/KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl",
    "libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl",
]


def _settle_map(mujoco, mjmodel, mjdata, sim, joint_name, z, n_steps):
    """f(z): set the object's free-joint z to ``z``, zero all velocities, run
    ``n_steps`` mj_step, re-zero velocities, and return (settled_z, x, y).

    Mirrors the production 50-step settle in ``simulator.setup`` exactly (same
    qvel-zero → mj_forward → N*mj_step → qvel-zero → mj_forward sequence), so the
    fixed point measured here is the fixed point of the validation settle."""
    qpos = sim.data.get_joint_qpos(joint_name).copy()
    qpos[2] = z
    sim.data.set_joint_qpos(joint_name, qpos)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    for _ in range(n_steps):
        mujoco.mj_step(mjmodel, mjdata)
    mjdata.qvel[:] = 0
    mujoco.mj_forward(mjmodel, mjdata)
    settled = sim.data.get_joint_qpos(joint_name)
    return float(settled[2]), float(settled[0]), float(settled[1])


def _iterate_fixed_point(
    mujoco,
    mjmodel,
    mjdata,
    sim,
    joint_name,
    z0,
    x0,
    y0,
    *,
    n_steps,
    tol,
    max_iter,
    arena_z,
    xy_flag,
):
    """Iterate f from z0 until |Δz| < tol (CONVERGED) or max_iter (NON_CONVERGENT)
    or roll-off (xy drift > xy_flag, or z <= arena_z). Returns a dict."""
    traj = []
    z = z0
    x, y = x0, y0
    status = "NON_CONVERGENT"
    for i in range(max_iter):
        z_next, x_next, y_next = _settle_map(mujoco, mjmodel, mjdata, sim, joint_name, z, n_steps)
        dz = z_next - z
        xy_drift = float(np.hypot(x_next - x0, y_next - y0))
        traj.append(
            {
                "i": i,
                "z": round(z_next, 5),
                "dz_mm": round(dz * 1000, 2),
                "xy_drift_mm": round(xy_drift * 1000, 2),
            }
        )
        # roll-off: slid off the fixture footprint, or fell to the table surface
        if xy_drift > xy_flag:
            status = "ROLLED_OFF_XY"
            z, x, y = z_next, x_next, y_next
            break
        if z_next <= arena_z + 1e-4:
            status = "FELL_TO_TABLE"
            z, x, y = z_next, x_next, y_next
            break
        z, x, y = z_next, x_next, y_next
        if abs(dz) < tol:
            status = "CONVERGED"
            break
    return {
        "status": status,
        "z_star": round(z, 5),
        "x": round(x, 5),
        "y": round(y, 5),
        "iters": len(traj),
        "traj": traj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--subset", default="position")
    ap.add_argument("--tasks", default="", help="comma list; default = built-in candidate set")
    ap.add_argument("--tasks-file", default="")
    ap.add_argument(
        "--steps", type=int, default=50, help="settle steps per iteration (match validation)"
    )
    ap.add_argument("--tol", type=float, default=0.001, help="convergence |Δz| threshold (m)")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--xy-flag", type=float, default=0.03, help="xy drift (m) that marks roll-off")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    import mujoco

    from libero_infinity.asset_metadata import (
        TABLE_SURFACE_Z,
        _is_fixture_surface,
        spawn_clearance,
        surface_spawn_z,
    )
    from libero_infinity.compiler import build_semantic_scene_graph, compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.renderer.scenic_renderer import _arena_surface_z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import resolve_object_name
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    tasks = list(_DEFAULT_TASKS)
    if args.tasks_file:
        with open(args.tasks_file) as f:
            tasks = [ln.strip() for ln in f if ln.strip()]
    if args.tasks:
        tasks = [t for t in args.tasks.split(",") if t.strip()]

    # per (class, fixture) → list of converged records
    agg = collections.defaultdict(list)
    all_records = []

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
            # Auto-discover every movable object whose INIT support is a fixture.
            targets = []  # (name, class, fixture_class)
            for o in _iter_scene_objects(es):
                if getattr(o, "graspable", True) is False:
                    continue
                sc = getattr(o, "support_surface_class", "") or ""
                if sc and _is_fixture_surface(sc):
                    nm = resolve_object_name(o) or ""
                    cc = getattr(o, "asset_class", "") or ""
                    if nm and cc:
                        targets.append((nm, cc, sc))
            if not targets:
                env.close()
                continue

            sim = env._sim.libero_env.env.sim
            mjmodel = sim.model._model
            mjdata = sim.data._data

            for target_name, target_class, fixture_class in targets:
                joint_name = f"{target_name}_joint0"
                analytic_clear = spawn_clearance(target_class, fixture_class)
                analytic_z = surface_spawn_z(arena_z, target_class, fixture_class)
                try:
                    qpos0 = sim.data.get_joint_qpos(joint_name)
                    settled0_z = float(qpos0[2])
                    x0, y0 = float(qpos0[0]), float(qpos0[1])
                except Exception as exc:  # noqa: BLE001
                    print(f"# no joint {joint_name}: {exc}")
                    continue

                rec = _iterate_fixed_point(
                    mujoco,
                    mjmodel,
                    mjdata,
                    sim,
                    joint_name,
                    analytic_z,
                    x0,
                    y0,
                    n_steps=args.steps,
                    tol=args.tol,
                    max_iter=args.max_iter,
                    arena_z=arena_z,
                    xy_flag=args.xy_flag,
                )
                z_star = rec["z_star"]
                rec.update(
                    {
                        "task": task,
                        "class": target_class,
                        "fixture": fixture_class,
                        "name": target_name,
                        "seed": seed,
                        "arena_z": round(arena_z, 5),
                        "analytic_z": round(analytic_z, 5),
                        "analytic_clear": round(analytic_clear, 5),
                        "settled0_z": round(settled0_z, 5),
                        "clear_star": round(z_star - arena_z, 5),
                        "clear_star_vs_tablez": round(z_star - TABLE_SURFACE_Z, 5),
                        # Does today's emission already match the converged rest? If so
                        # the pair PASSES and must NOT get a variant row (no regression).
                        "analytic_gap_mm": round((analytic_z - z_star) * 1000, 2),
                    }
                )
                all_records.append(rec)
                if rec["status"] == "CONVERGED":
                    agg[(target_class, fixture_class)].append(rec)
                print(
                    f"[{target_class}|{fixture_class}] seed={seed} {rec['status']:14s} "
                    f"analytic_z={analytic_z:.4f} settled0={settled0_z:.4f} "
                    f"z*={z_star:.4f} clear*={rec['clear_star']:.4f} "
                    f"gap={rec['analytic_gap_mm']:+.1f}mm iters={rec['iters']}"
                )
            env.close()

    print("\n" + "=" * 72)
    print("CONVERGED FIXED-POINT SUMMARY (per class|fixture)")
    print("=" * 72)
    print("A pair needs an ADDITIVE variant row ONLY if |analytic_gap| > 5mm")
    print("(today's emission misses the converged rest) AND spread <= 5mm.\n")
    for (cc, fx), recs in sorted(agg.items()):
        zs = [r["z_star"] for r in recs]
        clears = [r["clear_star_vs_tablez"] for r in recs]
        gaps = [r["analytic_gap_mm"] for r in recs]
        spread = (max(zs) - min(zs)) * 1000
        mode_clear = collections.Counter(round(c, 4) for c in clears).most_common(1)[0][0]
        det = "DETERMINISTIC" if spread <= 5 else "METASTABLE(>5mm)->FLAG"
        gap = sum(gaps) / len(gaps)
        action = (
            "ADD ROW"
            if (spread <= 5 and abs(gap) > 5)
            else ("already-passes" if abs(gap) <= 5 else "FLAG(metastable)")
        )
        print(
            f"  {cc}|{fx}: n={len(recs)} z*={sum(zs)/len(zs):.4f} "
            f"clear(z*-TABLE_Z)={mode_clear:.4f} spread={spread:.2f}mm gap={gap:+.1f}mm "
            f"{det}  -> {action}"
        )
    # non-converged cases
    nonconv = [r for r in all_records if r["status"] != "CONVERGED"]
    if nonconv:
        print("\nNON-CONVERGENT / FLAGGED cases (no row; documented as residual):")
        seen = set()
        for r in nonconv:
            key = (r["class"], r["fixture"], r["status"])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  {r['class']}|{r['fixture']}: {r['status']} "
                f"(e.g. seed {r['seed']}, iters={r['iters']})"
            )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\nwrote {len(all_records)} records to {args.json_out}")


if __name__ == "__main__":
    main()
