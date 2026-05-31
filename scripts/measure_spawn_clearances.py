"""Generate ``data/spawn_clearances.json`` — the per-asset-class settled spawn
clearance registry used by ``asset_metadata.surface_spawn_z``.

The *spawn clearance* of an asset class is the height of its settled MuJoCo
body-origin above the Scenic table-surface constant ``TABLE_Z`` when the object
rests on the kitchen workspace table:

    clearance(class) = body_xpos_z(settled) - TABLE_Z

This is measured by instantiating real LIBERO environments (the authoritative
MuJoCo assets), resetting, and reading ``body_xpos`` for every table-resting
movable. It is the ground-truth resolved spawn z that the renderer must emit so
that the Scenic-sampled pose compares 1-to-1 against the post-reset MuJoCo pose
(G4 family-C ``pose_tolerance``; validation plan §4).

Reproducible: deterministic seed, fixed task list, median aggregation. Re-run
after asset upgrades:

    PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/measure_spawn_clearances.py
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics

# Kitchen tasks whose workspace table sits at the canonical TABLE_Z frame.
# Chosen for broad movable-class coverage (bowls, plates, bottles, boxes,
# cans, pots). Non-kitchen suites (e.g. living-room tables at a different
# height) are intentionally excluded — their surface is not TABLE_Z.
MEASURE_TASKS = [
    "libero_goal/put_the_bowl_on_the_stove.bddl",
    "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl",
    "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
    "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
    "libero_goal/put_the_cream_cheese_in_the_bowl.bddl",
    "libero_goal/put_the_wine_bottle_on_the_rack.bddl",
    "libero_goal/put_the_bowl_on_the_plate.bddl",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
    "libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove.bddl",
    "libero_90/KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl",
    "libero_90/KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet.bddl",
    "libero_90/KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.bddl",
    "libero_90/KITCHEN_SCENE7_put_the_white_bowl_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.bddl",
    "libero_90/KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it.bddl",
]

SUBSETS = ["position", "position,object"]

# Per-(variant, surface) measurement draws the object axis many times so every
# OOD variant in each object's pool gets instantiated and settled. The key is
# the SAME (asset_class, support_surface_class) the renderer resolves and emits
# on each object, so the measured table and the renderer's lookup are guaranteed
# to agree (no independent surface re-derivation).
_VARIANT_SEEDS = 12


def measure_variants() -> dict:
    """Measure settled clearance keyed by (variant_class, support_surface_class).

    Resolves Finding A: an object-axis OOD variant generally seats at a
    different clearance than its canonical class, and the seating height depends
    on the support surface. For every measure task we draw ``position,object``
    scenes across multiple seeds so the variant pool is covered, reset the real
    MuJoCo env, and record ``settled_z - TABLE_Z`` bucketed by the object's
    instantiated ``asset_class`` and its renderer-emitted
    ``support_surface_class`` (empty → the default workspace table, keyed
    ``"table"``). Output feeds ``asset_metadata.VARIANT_CLEARANCES``.
    """
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import (
        is_scene_fixture,
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    avail = set(discover_all_tasks())
    samples: dict[str, list[float]] = {}

    for task_rel in MEASURE_TASKS:
        if task_rel not in avail:
            continue
        bddl = str(resolve_task_path(task_rel))
        for seed in range(_VARIANT_SEEDS):
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(seed)
                scenario = compile_task_to_scenario(cfg, "position,object")
                scene, _ = scenario.generate(maxIterations=2000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception as exc:  # noqa: BLE001 — measurement noise, recorded
                print(f"# build failed {task_rel} [seed {seed}]: {exc}")
                continue
            for o in _iter_scene_objects(scene):
                if is_scene_fixture(o):
                    continue
                if not getattr(o, "graspable", True):
                    continue
                # Contained/movable-supported children derive z from a support
                # relation, not the table surface model — exclude them, exactly
                # as the canonical-class measurement does.
                sp = getattr(o, "support_parent_name", "")
                if sp and "table" not in sp.lower():
                    continue
                nm = resolve_object_name(o) or "?"
                cls = getattr(o, "asset_class", None)
                if not cls:
                    continue
                surface = getattr(o, "support_surface_class", "") or "table"
                st = env.get_object_state(nm)
                if st is None:
                    continue
                clearance = st["position"][2] - TABLE_Z
                if not (0.0 <= clearance <= 0.18):
                    continue
                key = f"{cls}|{surface}"
                samples.setdefault(key, []).append(round(float(clearance), 5))
            env.close()

    registry = {k: round(statistics.median(v), 5) for k, v in sorted(samples.items())}
    return {
        "_meta": {
            "description": "Per-(variant_class|surface_class) settled spawn "
            "clearance above TABLE_Z (metres). clearance = settled body_xpos_z - "
            "TABLE_Z. Generated by scripts/measure_spawn_clearances.py "
            "(measure_variants) from real LIBERO MuJoCo assets. Key is "
            "'<asset_class>|<support_surface_class>' matching the renderer's "
            "emitted support_surface_class.",
            "table_z": None,  # filled below
            "n_samples": {k: len(v) for k, v in sorted(samples.items())},
        },
        "clearances": registry,
    }


def measure() -> dict:
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import (
        is_scene_fixture,
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    avail = set(discover_all_tasks())
    samples: dict[str, list[float]] = {}

    for task_rel in MEASURE_TASKS:
        if task_rel not in avail:
            print(f"# SKIP (not found): {task_rel}")
            continue
        bddl = str(resolve_task_path(task_rel))
        for subset in SUBSETS:
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(0)
                scenario = compile_task_to_scenario(cfg, subset)
                scene, _ = scenario.generate(maxIterations=2000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception as exc:  # noqa: BLE001 — measurement noise, recorded
                print(f"# build failed {task_rel} [{subset}]: {exc}")
                continue
            for o in _iter_scene_objects(scene):
                if is_scene_fixture(o):
                    continue
                # Skip contained / supported children — their z derives from a
                # support relation, not the table surface. The per-scene
                # workspace fixture is named ``*_table`` (main_table,
                # kitchen_table, …) and IS the TABLE_Z surface, so objects
                # resting on it are exactly what we measure. Any other support
                # (a stacked object, a container, a drawer) is excluded.
                sp = getattr(o, "support_parent_name", "")
                if sp and "table" not in sp.lower():
                    continue
                if not getattr(o, "graspable", True):
                    continue
                nm = resolve_object_name(o) or "?"
                cls = getattr(o, "asset_class", None)
                if not cls:
                    continue
                st = env.get_object_state(nm)
                if st is None:
                    continue
                ez = st["position"][2]
                clearance = ez - TABLE_Z
                # Table-resting band only: reject objects that settled onto an
                # elevated fixture (cabinet top, stove) or fell off the table.
                if not (0.0 <= clearance <= 0.18):
                    continue
                samples.setdefault(str(cls), []).append(round(float(clearance), 5))
            env.close()

    registry = {cls: round(statistics.median(vals), 5) for cls, vals in sorted(samples.items())}
    return {
        "_meta": {
            "description": "Per-asset-class settled spawn clearance above TABLE_Z "
            "(metres). clearance = settled body_xpos_z - TABLE_Z. Generated by "
            "scripts/measure_spawn_clearances.py from real LIBERO MuJoCo assets.",
            "table_z": None,  # filled below
            "n_samples": {cls: len(vals) for cls, vals in sorted(samples.items())},
        },
        "clearances": registry,
    }


if __name__ == "__main__":
    import sys

    from libero_infinity.simulator import TABLE_Z

    out = measure()
    out["_meta"]["table_z"] = TABLE_Z
    dest = pathlib.Path("src/libero_infinity/data/spawn_clearances.json")
    dest.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    print(f"\nWrote {dest} with {len(out['clearances'])} classes:")
    for cls, v in out["clearances"].items():
        print(f"  {cls:28} {v:.4f}  (n={out['_meta']['n_samples'][cls]})")

    if "--no-variants" not in sys.argv:
        vout = measure_variants()
        vout["_meta"]["table_z"] = TABLE_Z
        vdest = pathlib.Path("src/libero_infinity/data/spawn_clearances_variants.json")
        vdest.write_text(json.dumps(vout, indent=2, sort_keys=False) + "\n")
        print(f"\nWrote {vdest} with {len(vout['clearances'])} (variant|surface) keys:")
        for key, v in vout["clearances"].items():
            print(f"  {key:44} {v:.4f}  (n={vout['_meta']['n_samples'][key]})")
