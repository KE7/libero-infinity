"""Measure the TRUE canonical-frame xy footprint (AABB) of movable objects from
their loaded MuJoCo geometry, to validate / correct the hand-set registry
``data/asset_variants.json:dimensions`` used by the movable<->movable and
object<->fixture separation constraints (RCA g4_cabinet_heightfield.md §5.3).

For each requested class we build a scene that contains it, reset the env, find
the object's free-joint body, and compute the union AABB of all its collision
(contype != 0) geoms in the BODY-LOCAL frame (so a settle tilt does not inflate
the measurement). We report full extents (width, length, height) in metres, both
over collision geoms only and over ALL geoms (visual+collision), alongside the
current registry dims. Nothing is written; the operator reviews and edits the
registry.
"""

import argparse
import random

import numpy as np

# Tasks that are known to instantiate each class, so we can load real geometry.
_CLASS_TASKS = {
    "akita_black_bowl": "libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
    "white_bowl": "libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl",
    "cookies": "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
    "plate": "libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
    "ketchup": "libero_object/pick_up_the_ketchup_and_place_it_in_the_basket.bddl",
    "glazed_rim_porcelain_ramekin": "libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl",
    "basket": "libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl",
    "alphabet_soup": "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
    "milk": "libero_object/pick_up_the_milk_and_place_it_in_the_basket.bddl",
}


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _body_geom_aabb(sim, mjmodel, bid, collision_only):
    """Union AABB (half-extents) of body ``bid``'s geoms in the BODY-LOCAL frame."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = 0
    for gid in range(mjmodel.ngeom):
        if mjmodel.geom_bodyid[gid] != bid:
            continue
        if collision_only and mjmodel.geom_contype[gid] == 0 and mjmodel.geom_conaffinity[gid] == 0:
            continue
        # geom-local tight AABB (center + halfextents) in the geom frame.
        c = mjmodel.geom_aabb[gid][:3]
        h = mjmodel.geom_aabb[gid][3:]
        gpos = mjmodel.geom_pos[gid]
        gmat = _quat_to_mat(mjmodel.geom_quat[gid])
        # 8 corners of the geom AABB -> body frame
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = c + np.array([sx * h[0], sy * h[1], sz * h[2]])
                    p = gpos + gmat @ corner
                    lo = np.minimum(lo, p)
                    hi = np.maximum(hi, p)
        found += 1
    if found == 0:
        return None
    return hi - lo  # full extents (w, l, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="akita_black_bowl")
    args = ap.parse_args()

    from libero_infinity.asset_registry import get_dimensions
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"{'class':<32} {'registry(w,l,h)':<24} {'coll_geom(w,l,h)':<26} {'all_geom(w,l,h)':<26}")
    print("-" * 110)
    for cls in classes:
        task = _CLASS_TASKS.get(cls)
        if not task:
            print(f"{cls:<32} NO TASK MAPPED")
            continue
        bddl = str(resolve_task_path(task))
        try:
            cfg = TaskConfig.from_bddl(bddl)
            random.seed(0)
            np.random.seed(0)
            scn = compile_task_to_scenario(cfg, "position")
            scene, _ = scn.generate(maxIterations=20000)
            env = make_env(scene, bddl_path=bddl)
            env.reset()
        except Exception as exc:  # noqa: BLE001
            print(f"{cls:<32} build failed: {exc}")
            continue

        sim = env._sim.libero_env.env.sim
        mjmodel = sim.model._model

        # find bodies whose name contains the class
        target_bids = []
        for bid in range(mjmodel.nbody):
            nm = sim.model.body_id2name(bid) or ""
            if cls in nm:
                target_bids.append((bid, nm))
        if not target_bids:
            print(f"{cls:<32} no body matched")
            env.close()
            continue

        # pick the body that actually carries geoms
        best = None
        for bid, nm in target_bids:
            coll = _body_geom_aabb(sim, mjmodel, bid, collision_only=True)
            allg = _body_geom_aabb(sim, mjmodel, bid, collision_only=False)
            if allg is not None:
                best = (nm, coll, allg)
                break
        reg = get_dimensions(cls)

        def fmt(v):
            return "None" if v is None else f"({v[0]:.4f},{v[1]:.4f},{v[2]:.4f})"

        if best is None:
            print(f"{cls:<32} {fmt(reg):<24} no geoms found on any matching body")
        else:
            nm, coll, allg = best
            print(f"{cls:<32} {fmt(reg):<24} {fmt(coll):<26} {fmt(allg):<26}  [{nm}]")
        env.close()


if __name__ == "__main__":
    main()
