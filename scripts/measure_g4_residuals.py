"""Measure the G4 residual z-references the arena-table sweep does not cover:

  1. ``cookies`` table-rest clearance (reference-canonical correction check):
     cookies_1 is the table-resting cookie box in the libero_spatial scenes; its
     canonical ``spawn_clearances.json`` value is suspected ~20 mm too high.
  2. ``white_bowl|microwave`` task-object-on-fixture clearance: white_bowl_1 rests
     ON microwave_1_top_side in KITCHEN_SCENE7; the analytic top_z fallback
     over-estimates by ~49 mm (microwave AABB top is the highest geom, not the
     real rest surface — same failure mode as flat_stove).
  3. STACK_DELTA(child_class, parent_class): for the true object-on-object stacks
     (akita_black_bowl on cookies / ramekin / another bowl) the settled child
     body-origin sits ``child_settled_z − parent_settled_z`` above the parent
     origin. The renderer currently emits a 0.0 relative-z offset (inherits the
     parent origin) — this measures the offset it SHOULD emit so scenic_z ==
     settled_z. Reported with the cross-seed spread (determinism gate).

All quantities are measured through the production env path (the SAME path
pose_tolerance scores), over several deterministic seeds, with the dominant
settle mode + spread reported. Nothing is written; the operator reviews the
numbers and applies the additive data edits.
"""

import argparse
import collections
import random

import numpy as np

# (task, parent_instance_substr -> parent_class) stack specs: which child rests
# on which movable parent. Child class is read from the realized scene.
_STACK_TASKS = [
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.bddl",
    "libero_90/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.bddl",
    "libero_90/LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.bddl",
]
_COOKIES_TASKS = [
    "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
]
_MICROWAVE_TASKS = [
    "libero_90/KITCHEN_SCENE7_put_the_white_bowl_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate.bddl",
]


def _origin_z(env, name):
    st = env.get_object_state(name)
    return None if st is None else float(st["position"][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--subsets", default="position,object")
    args = ap.parse_args()

    from libero_infinity.asset_metadata import TABLE_SURFACE_Z
    from libero_infinity.compiler import build_semantic_scene_graph, compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.planner.composition import plan_perturbations
    from libero_infinity.renderer.scenic_renderer import _collect_support_relations
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import (
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    subsets = [s for s in args.subsets.split(",") if s]

    # ---- cookies table-rest clearance ----
    cookies_samples = []
    for t in _COOKIES_TASKS:
        bddl = str(resolve_task_path(t))
        for seed in range(args.seeds):
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(seed)
                scn = compile_task_to_scenario(cfg, "position")
                scene, _ = scn.generate(maxIterations=20000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception:
                continue
            es = getattr(env, "realized_scene", None) or scene
            for o in _iter_scene_objects(es):
                if getattr(o, "asset_class", "") == "cookies":
                    z = _origin_z(env, resolve_object_name(o) or "")
                    if z is not None:
                        cookies_samples.append(round(z - TABLE_SURFACE_Z, 5))
            env.close()

    # ---- white_bowl|microwave fixture clearance ----
    mw_samples = []
    for t in _MICROWAVE_TASKS:
        bddl = str(resolve_task_path(t))
        for seed in range(args.seeds):
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(seed)
                scn = compile_task_to_scenario(cfg, "position")
                scene, _ = scn.generate(maxIterations=20000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception:
                continue
            es = getattr(env, "realized_scene", None) or scene
            for o in _iter_scene_objects(es):
                if getattr(o, "asset_class", "") == "white_bowl":
                    z = _origin_z(env, resolve_object_name(o) or "")
                    if z is not None:
                        mw_samples.append(round(z - TABLE_SURFACE_Z, 5))
            env.close()

    # ---- stack deltas ----
    # key: (child_class, parent_class) -> list[child_origin - parent_origin]
    stack_samples = collections.defaultdict(list)
    for t in _STACK_TASKS:
        bddl = str(resolve_task_path(t))
        for sub in subsets:
            for seed in range(args.seeds):
                try:
                    cfg = TaskConfig.from_bddl(bddl)
                    random.seed(seed)
                    np.random.seed(seed)
                    scn = compile_task_to_scenario(cfg, sub)
                    scene, _ = scn.generate(maxIterations=20000)
                    graph = build_semantic_scene_graph(cfg)
                    plan = plan_perturbations(graph, sub)
                    rels = _collect_support_relations(plan, graph)
                    env = make_env(scene, bddl_path=bddl)
                    env.reset()
                except Exception as exc:
                    print(f"# build failed {t} [{sub} {seed}]: {exc}")
                    continue
                es = getattr(env, "realized_scene", None) or scene
                cls_by_name = {}
                for o in _iter_scene_objects(es):
                    nm = resolve_object_name(o) or ""
                    cls_by_name[nm] = getattr(o, "asset_class", "")
                for rel in rels:
                    # genuine movable stack: not a fixture support, not contained
                    if rel.support_is_fixture or rel.kind == "inside":
                        continue
                    cz = _origin_z(env, rel.child_name)
                    pz = _origin_z(env, rel.support_name)
                    if cz is None or pz is None:
                        continue
                    cc = cls_by_name.get(rel.child_name, "?")
                    pc = cls_by_name.get(rel.support_name, "?")
                    stack_samples[(cc, pc)].append(round(cz - pz, 5))
                env.close()

    def _report(name, samples):
        if not samples:
            print(f"\n{name}: NO SAMPLES")
            return
        mode = collections.Counter(round(s, 3) for s in samples).most_common(1)[0][0]
        spread = (max(samples) - min(samples)) * 1000
        print(
            f"\n{name}: n={len(samples)} mode={mode:.4f} "
            f"min={min(samples):.4f} max={max(samples):.4f} spread={spread:.1f}mm"
            f"{'  <-- NON-DETERMINISTIC' if spread > 5 else ''}"
        )

    print("=" * 60)
    _report("cookies|table (clearance above TABLE_SURFACE_Z)", cookies_samples)
    _report("white_bowl|microwave (clearance above TABLE_SURFACE_Z)", mw_samples)
    print("\n=== STACK_DELTA(child|parent) = child_origin - parent_origin ===")
    for (cc, pc), s in sorted(stack_samples.items()):
        _report(f"  STACK {cc}|{pc}", s)


if __name__ == "__main__":
    main()
