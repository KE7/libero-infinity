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

# Workspace-table fixture classes (the arena tables objects rest on directly).
# Must stay in sync with ``WorldModel.root_workspace_fixtures`` in
# ``ir/nodes.py``. A movable keyed under ``<class>|<surface>`` for one of these
# surfaces is only a valid table-resting sample if it physically settles in
# contact with the table — not on an elevated fixture that happened to sit
# under its sampled (x, y).
_WORKSPACE_TABLE_CLASSES = frozenset(
    {"table", "kitchen_table", "living_room_table", "study_table", "floor"}
)

# Settling budget for the physical-support check. A movable resting at its
# Scenic spawn z sits ~1–2 mm above the table collision surface after the
# setup() settle; a few dozen extra steps close that micro-gap so a genuine
# table contact registers, while an object perched on a fixture never makes a
# table contact no matter how long it settles.
_SUPPORT_CHECK_STEPS = 80

# Max allowed deviation (m) between a measured (canonical_class|workspace-table)
# clearance and that class's canonical per-class clearance. This is a COARSE
# backstop, not the primary defense — the per-sample physical-support contact
# guard (``_settled_on_table_surface``) is what precisely removes the
# over-measurement (a bowl settling on a wine_rack, mis-bucketed as
# table-resting, which shifted akita_black_bowl|table by ~31 mm). The tolerance
# must clear legit small-sample seating variance between the pooled-canonical
# median and a per-bucket median (multimodal assets such as an upright-vs-tipped
# milk carton span ~25 mm) while still catching gross fixture-stacking
# contamination if the contact guard ever regresses.
_TABLE_ROW_TOL = 0.03


def _settled_on_table_surface(env, name: str, *, max_steps: int = _SUPPORT_CHECK_STEPS) -> bool:
    """Return True iff movable ``name`` physically rests on the workspace table.

    Root-cause guard for the per-(variant, surface) over-measurement: the
    sample loop buckets a movable by the renderer's *intended*
    ``support_surface_class``, but position sampling can land the object's
    (x, y) over an elevated fixture (e.g. a ``wine_rack``) so it settles ON the
    fixture, ~tens of mm above the table. Its ``body_xpos[2] - TABLE_Z`` then
    over-states the true table-resting clearance and, in a small per-bucket
    median, dominates (akita_black_bowl|table: 0.100 → 0.132).

    We resolve the *physical* support by stepping the live MuJoCo sim forward
    (settling the micro-gap above the table) and inspecting ``data.contact``:
    a genuine table-rester makes a contact with a table geom; a fixture-perched
    object never does. The sim state (qpos/qvel) is snapshotted and restored so
    the check is side-effect-free and the recorded z stays the reset pose.
    """
    sim = env._sim.libero_env.env.sim  # noqa: SLF001 — measurement-only sim handle
    body_ids = getattr(env._sim, "_body_ids", None) or {}
    bid = body_ids.get(name)
    if bid is None:
        return False
    model, data = sim.model, sim.data
    obj_geoms = frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid)
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    try:
        for _ in range(max_steps):
            sim.step()
            for c in range(data.ncon):
                con = data.contact[c]
                if con.geom1 in obj_geoms or con.geom2 in obj_geoms:
                    other = con.geom2 if con.geom1 in obj_geoms else con.geom1
                    obody = (model.body_id2name(model.geom_bodyid[other]) or "").lower()
                    ogeom = (model.geom_id2name(other) or "").lower()
                    if "table" in obody or "table" in ogeom:
                        return True
        return False
    finally:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        sim.forward()


def _assert_table_rows_match_canonical(
    variant_clearances: dict[str, float],
    canonical_clearances: dict[str, float],
    *,
    tol: float = _TABLE_ROW_TOL,
) -> None:
    """Fail loudly if any *canonical-class* table row drifts from its canonical z.

    For a workspace-table surface the settled clearance of a *known* base class
    is its canonical per-class clearance (the arena tables are all at the same
    height; FV Task 5 found z is surface-invariant across them). Any
    ``<canonical_class>|<table>`` row that deviates by more than ``tol`` means a
    non-table-resting sample leaked into the bucket (the exact over-measurement
    the physical-support guard removes), so raise rather than ship an inflated z.

    Rows whose asset is NOT a measured canonical class are object-axis OOD
    variants with no per-class ground truth: a tall variant
    (``macaroni_and_cheese``, an elongated box) legitimately seats higher than
    the ~0.10 median, so comparing it against a median prior would false-positive.
    Those rows are already protected by the per-sample physical-support contact
    guard; we only pin the rows for which a canonical value exists.
    """
    offenders = []
    for key, measured in variant_clearances.items():
        asset, _, surface = key.partition("|")
        if surface not in _WORKSPACE_TABLE_CLASSES:
            continue
        canonical = canonical_clearances.get(asset)
        if canonical is None:
            # OOD variant with no canonical ground truth — guarded by the
            # physical-support contact check, not pinned to the median prior.
            continue
        if abs(measured - canonical) > tol:
            offenders.append(
                f"{key}: measured={measured:.5f} canonical={canonical:.5f} "
                f"|Δ|={abs(measured - canonical) * 1000:.1f}mm > {tol * 1000:.0f}mm"
            )
    if offenders:
        raise AssertionError(
            "Per-(variant, surface) clearance sanity check FAILED — workspace-table "
            "rows for canonical base classes must equal the canonical per-class "
            f"clearance within {tol * 1000:.0f} mm (a larger gap means a "
            "non-table-resting sample polluted the bucket):\n  " + "\n  ".join(offenders)
        )

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
                # Physical-support guard: only bucket the object under a
                # workspace-table key when it actually settles in contact with
                # the table. Objects whose (x, y) landed over a fixture settle
                # ON the fixture (tens of mm higher) and would otherwise inflate
                # the median (Finding: akita_black_bowl|table 0.100 → 0.132).
                if surface in _WORKSPACE_TABLE_CLASSES and not _settled_on_table_surface(env, nm):
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
                # Physical-support guard (see measure_variants): the band alone
                # admits objects perched on a fixture whose top happens to fall
                # in the band; require an actual table contact so the pooled
                # median is not biased by fixture-resting outliers.
                if not _settled_on_table_surface(env, nm):
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

        # Fail loudly before writing if any canonical-class workspace-table row
        # drifted from the canonical per-class clearance — that can only happen
        # if a non-table-resting sample leaked past the physical-support guard.
        _assert_table_rows_match_canonical(vout["clearances"], out["clearances"])

        vdest = pathlib.Path("src/libero_infinity/data/spawn_clearances_variants.json")
        vdest.write_text(json.dumps(vout, indent=2, sort_keys=False) + "\n")
        print(f"\nWrote {vdest} with {len(vout['clearances'])} (variant|surface) keys:")
        for key, v in vout["clearances"].items():
            print(f"  {key:44} {v:.4f}  (n={vout['_meta']['n_samples'][key]})")
