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


# Distractor-on-fixture measurement (Fix 2, option i)
# -----------------------------------------------------
# Per-(distractor_class, fixture_class) settled spawn clearance for distractors
# that rest ON a scene fixture (stove burner, cabinet top, wine-rack shelf).
# The clearance is MEASURED from a real settle (ground truth) and cross-checked,
# loudly, against the fixture's actual top SURFACE the distractor contacts:
#
#   resting_center_z (settled body_xpos[2])  ==  contact_surface_z + body_origin_offset
#
# Both terms are measured from geometry. ``contact_surface_z`` is the world z of
# the distractor↔fixture contact point (NOT the fixture's max geom z — a wine
# rack's frame extends well above the shelf objects rest on, so max-geom-z would
# be the wrong "top"). ``body_origin_offset`` is ``settled body z - distractor
# bottom z`` (the body origin's height above its own lowest point at rest, which
# is the frame-correct "half height" that does NOT assume the body origin is the
# geometric centre — directly guarding the frame-confusion failure class). The
# identity therefore reduces to the physical truth ``distractor_bottom_z ==
# contact_surface_z`` (object bottom sits on the surface), asserted within
# ``_FIXTURE_SETTLE_TOL`` per sample, or we fail loudly.

# Tolerance (m) for the frame/stability assertion: the settled distractor's
# bottom face must sit on the fixture geom it contacts. This is the frame-
# correct restatement of the EA's "resting_center_z = fixture_top + half_height"
# — with half_height = (settled center − settled bottom) measured AT REST, so it
# makes no assumption that the body origin is the geometric centre (the exact
# 40/43-asset frame-confusion class). A frame-conversion error manifests as a
# ≥cm-scale gap; the bound is set generously enough to also tolerate irregular-
# footprint distractors (e.g. a bowl_drainer / desk_caddy whose AABB extends
# below its contact feet by a few cm). The precise injected==settled guarantee
# is independently verified by the smoke's 5 mm pose_tolerance.
_FIXTURE_SETTLE_TOL = 0.05

# Distractor-on-fixture clearance must stay in a plausible physical band; a
# sample outside it means the object bounced off the fixture rather than resting.
_FIXTURE_CLEARANCE_MAX = 0.60

_DISTRACTOR_FIXTURE_SEEDS = 16


def _geom_world_aabb(sim, geom_id: int) -> tuple[float, float, float, float, float, float]:
    """World-frame AABB (xmin,xmax,ymin,ymax,zmin,zmax) of a single geom."""
    import numpy as np

    model = sim.model
    raw = model._model  # noqa: SLF001 — mujoco MjModel for geom_aabb
    aabb = raw.geom_aabb[geom_id]
    c_local = np.asarray(aabb[:3], dtype=float)
    half = np.asarray(aabb[3:], dtype=float)
    xpos = np.asarray(sim.data.geom_xpos[geom_id], dtype=float)
    rot = np.asarray(sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    center = xpos + rot @ c_local
    ext = np.array([sum(abs(rot[ax, k]) * half[k] for k in range(3)) for ax in range(3)])
    lo = center - ext
    hi = center + ext
    return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]


def _body_world_aabb(sim, body_id: int):
    """Union world AABB over all geoms of ``body_id``; None if it has none."""
    model = sim.model
    boxes = [
        _geom_world_aabb(sim, g) for g in range(model.ngeom) if model.geom_bodyid[g] == body_id
    ]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
        min(b[4] for b in boxes),
        max(b[5] for b in boxes),
    )


def _fixture_body_ids(sim, fixture_instance: str) -> list[int]:
    """Body ids of every body belonging to ``fixture_instance`` (subtree by name)."""
    model = sim.model
    ids: list[int] = []
    for b in range(model.nbody):
        nm = model.body_id2name(b) or ""
        if nm == fixture_instance or nm.startswith(fixture_instance + "_"):
            ids.append(b)
    return ids


def _fixture_world_aabb(sim, fixture_instance: str):
    """Union world AABB over all geoms of every body of ``fixture_instance``."""
    boxes = [
        _body_world_aabb(sim, b) for b in _fixture_body_ids(sim, fixture_instance)
    ]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
        min(b[4] for b in boxes),
        max(b[5] for b in boxes),
    )


def _distractor_fixture_contact_tops(
    sim, distractor_bid: int, fixture_instance: str
) -> list[float]:
    """World-AABB top z of every fixture geom the distractor is in contact with.

    Resolves the rest surface for multi-level fixtures (a wine rack's shelf vs
    its post tops) by looking at the SPECIFIC geom the distractor touches, then
    taking that geom's world-AABB top — the geometry the object sits on. We use
    the geom AABB (which is reliable) rather than ``contact.pos`` (degenerate in
    this robosuite/mujoco binding — it returns a broadcast scalar). Returns an
    empty list when the distractor touches no fixture geom (floated / landed
    elsewhere — not a valid on-fixture sample).
    """
    model = sim.model
    data = sim.data
    obj_geoms = frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] == distractor_bid)
    fixture_bodies = frozenset(_fixture_body_ids(sim, fixture_instance))
    tops: list[float] = []
    for c in range(int(data.ncon)):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        a_obj, b_obj = g1 in obj_geoms, g2 in obj_geoms
        if a_obj == b_obj:
            continue  # neither or both are the distractor → not an obj↔other pair
        other = g2 if a_obj else g1
        if model.geom_bodyid[other] not in fixture_bodies:
            continue
        tops.append(_geom_world_aabb(sim, other)[5])
    return tops


def measure_distractor_fixtures(table_clearances: dict[str, float]) -> tuple[dict, dict]:
    """Measure per-(distractor_class, fixture_class) on-fixture spawn clearance.

    Returns ``(variant_rows, fixture_geometry)``:
      * ``variant_rows`` — ``{"<class>|<fixture_class>": clearance}`` to merge
        into ``spawn_clearances_variants.json`` (clearance = settled body z -
        TABLE_Z).
      * ``fixture_geometry`` — ``{fixture_class: {"footprint": [w,l], "top_z":
        rest_top_above_table, "height": h}}`` for ``data/fixture_geometry.json``.
        ``top_z`` is the REST surface (contact) height above the table — the
        surface a distractor actually settles onto — and ``footprint``/``height``
        are the measured geom-AABB extents (used for clearance exclusion).

    Raises loudly if any settled distractor's bottom face does not coincide with
    its fixture contact surface within ``_FIXTURE_SETTLE_TOL`` (frame error).
    """

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    avail = set(discover_all_tasks())
    samples: dict[str, list[float]] = {}
    fixture_footprints: dict[str, list[tuple[float, float, float]]] = {}
    fixture_rest_tops: dict[str, list[float]] = {}
    frame_offenders: list[str] = []
    n_contact_miss = 0

    for task_rel in MEASURE_TASKS:
        if task_rel not in avail:
            continue
        bddl = str(resolve_task_path(task_rel))
        for seed in range(_DISTRACTOR_FIXTURE_SEEDS):
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(seed)
                scenario = compile_task_to_scenario(cfg, "distractor")
                scene, _ = scenario.generate(maxIterations=8000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception as exc:  # noqa: BLE001 — measurement noise, recorded
                print(f"# build failed {task_rel} [seed {seed}]: {exc}")
                continue
            sim = env._sim.libero_env.env.sim  # noqa: SLF001
            active = getattr(env._sim, "_active_distractor_names", set())  # noqa: SLF001
            for o in scene.objects:
                nm = getattr(o, "libero_name", "")
                if not nm.startswith("distractor_") or nm not in active:
                    continue
                surface_class = getattr(o, "support_surface_class", "") or ""
                fixture_inst = getattr(o, "support_parent_name", "") or ""
                if not surface_class or not fixture_inst:
                    continue  # table-assigned distractor — measured elsewhere
                cls = getattr(o, "asset_class", "") or ""
                if not cls:
                    continue
                bid = None
                for cand in (nm, nm + "_main"):
                    try:
                        bid = sim.model.body_name2id(cand)
                        break
                    except Exception:
                        continue
                if bid is None:
                    continue
                body_z = float(sim.data.body_xpos[bid][2])
                box = _body_world_aabb(sim, bid)
                tops = _distractor_fixture_contact_tops(sim, bid, fixture_inst)
                if box is None or not tops:
                    n_contact_miss += 1
                    continue
                bottom_z = box[4]
                clearance = body_z - TABLE_Z
                if not (0.0 <= clearance <= _FIXTURE_CLEARANCE_MAX):
                    continue
                # Rest surface: the top of the fixture geom the distractor sits on
                # — the contacted geom whose top is nearest the distractor's bottom
                # face (a side contact at another height is not the support).
                nearest_top = min(tops, key=lambda t: abs(t - bottom_z))
                rest_top_above_table = nearest_top - TABLE_Z
                # Loud frame/stability cross-check (orientation-invariant): the
                # distractor's settled bottom face must sit on the contacted
                # fixture-geom top. settled_center = nearest_top + (center −
                # bottom), i.e. the EA's fixture_top + half_height with the
                # half-height MEASURED at rest (no body-origin-is-centre
                # assumption). A frame error shows as a ≥cm gap here.
                if abs(bottom_z - nearest_top) > _FIXTURE_SETTLE_TOL:
                    body_half_above_bottom = body_z - bottom_z
                    frame_offenders.append(
                        f"{cls}|{surface_class} ({nm}@{task_rel} seed {seed}): "
                        f"bottom_z={bottom_z:.4f} nearest_contact_geom_top={nearest_top:.4f} "
                        f"(settled_center={body_z:.4f}, half_above_bottom="
                        f"{body_half_above_bottom:.4f}) "
                        f"|Δ|={abs(bottom_z - nearest_top) * 1000:.1f}mm > "
                        f"{_FIXTURE_SETTLE_TOL * 1000:.0f}mm"
                    )
                samples.setdefault(f"{cls}|{surface_class}", []).append(round(clearance, 5))
                fixture_rest_tops.setdefault(surface_class, []).append(
                    round(rest_top_above_table, 5)
                )
                faabb = _fixture_world_aabb(sim, fixture_inst)
                if faabb is not None:
                    fixture_footprints.setdefault(surface_class, []).append(
                        (
                            round(faabb[1] - faabb[0], 5),
                            round(faabb[3] - faabb[2], 5),
                            round(faabb[5] - faabb[4], 5),
                        )
                    )
            env.close()

    if frame_offenders:
        raise AssertionError(
            "Distractor-on-fixture frame/stability check FAILED — a settled "
            "distractor's bottom face must coincide with its fixture contact "
            f"surface within {_FIXTURE_SETTLE_TOL * 1000:.0f} mm (a larger gap "
            "means a frame-conversion error or an unstable settle):\n  "
            + "\n  ".join(frame_offenders[:40])
        )

    variant_rows = {k: round(statistics.median(v), 5) for k, v in sorted(samples.items())}
    fixture_geometry: dict[str, dict] = {}
    for fclass in sorted(set(fixture_footprints) | set(fixture_rest_tops)):
        fps = fixture_footprints.get(fclass, [])
        tops = fixture_rest_tops.get(fclass, [])
        entry: dict = {}
        if fps:
            entry["footprint"] = [
                round(statistics.median(p[0] for p in fps), 5),
                round(statistics.median(p[1] for p in fps), 5),
            ]
            entry["height"] = round(statistics.median(p[2] for p in fps), 5)
        if tops:
            entry["top_z"] = round(statistics.median(tops), 5)
        if entry:
            fixture_geometry[fclass] = entry

    print(
        f"\n# distractor-on-fixture: {len(variant_rows)} (class|fixture) rows, "
        f"{len(fixture_geometry)} fixture classes, {n_contact_miss} contact-miss samples skipped"
    )
    for k, v in variant_rows.items():
        cls, _, fclass = k.partition("|")
        on_table = table_clearances.get(cls)
        top = fixture_geometry.get(fclass, {}).get("top_z")
        chk = ""
        if on_table is not None and top is not None:
            analytic = top + on_table
            chk = f"  analytic(top+table)={analytic:.4f} Δ={abs(v - analytic) * 1000:.1f}mm"
        print(f"  {k:44} {v:.4f} (n={len(samples[k])}){chk}")
    return variant_rows, fixture_geometry


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


def _run_distractor_fixtures_only() -> None:
    """Measure ONLY the per-(distractor, fixture) clearances + fixture geometry
    and merge them into the existing data files, leaving the already-validated
    table-resting and object-axis variant rows untouched.

    Used to extend the landed (table) measurement with the Fix 2 on-fixture
    rows without re-running (and risking perturbing) the validated table/object
    measurement.
    """
    from libero_infinity.simulator import TABLE_Z

    table_path = pathlib.Path("src/libero_infinity/data/spawn_clearances.json")
    table_clearances = json.loads(table_path.read_text()).get("clearances", {})

    dist_rows, fixture_geometry = measure_distractor_fixtures(table_clearances)

    vdest = pathlib.Path("src/libero_infinity/data/spawn_clearances_variants.json")
    vdata = json.loads(vdest.read_text())
    vdata["clearances"].update(dist_rows)
    vdata["clearances"] = {k: vdata["clearances"][k] for k in sorted(vdata["clearances"])}
    vdata["_meta"]["n_distractor_fixture_rows"] = len(dist_rows)
    vdest.write_text(json.dumps(vdata, indent=2, sort_keys=False) + "\n")
    print(f"\nMerged {len(dist_rows)} (class|fixture) rows into {vdest}")

    fg_out = {
        "_meta": {
            "description": "Measured fixture geometry. footprint=[w,l] and height "
            "are the geom-AABB world extents; top_z is the REST surface height "
            "above TABLE_Z (the distractor↔fixture contact surface, NOT max geom "
            "z). Generated by scripts/measure_spawn_clearances.py "
            "(measure_distractor_fixtures).",
            "table_z": TABLE_Z,
        },
        "fixtures": fixture_geometry,
    }
    fgdest = pathlib.Path("src/libero_infinity/data/fixture_geometry.json")
    fgdest.write_text(json.dumps(fg_out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {fgdest} with {len(fixture_geometry)} fixture classes:")
    for fclass, g in sorted(fixture_geometry.items()):
        print(f"  {fclass:24} {g}")


if __name__ == "__main__":
    import sys

    from libero_infinity.simulator import TABLE_Z

    if "--distractor-fixtures-only" in sys.argv:
        _run_distractor_fixtures_only()
        raise SystemExit(0)

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

        # Fix 2: per-(distractor_class, fixture_class) on-fixture clearances +
        # measured fixture geometry. Merged into the same variant table so
        # ``asset_metadata.surface_spawn_z`` resolves an on-fixture distractor's
        # seating z exactly (renderer/simulator lockstep). Fails loudly if any
        # settled distractor's bottom face does not sit on its fixture contact
        # surface (frame error).
        if "--no-distractor-fixtures" not in sys.argv:
            dist_rows, fixture_geometry = measure_distractor_fixtures(out["clearances"])
            vout["clearances"].update(dist_rows)
            vout["clearances"] = {k: vout["clearances"][k] for k in sorted(vout["clearances"])}
            vout["_meta"]["n_distractor_fixture_rows"] = len(dist_rows)

            fg_out = {
                "_meta": {
                    "description": "Measured fixture geometry. footprint=[w,l] and "
                    "height are the geom-AABB world extents; top_z is the REST "
                    "surface height above TABLE_Z (the distractor↔fixture contact "
                    "surface, NOT max geom z). Generated by "
                    "scripts/measure_spawn_clearances.py (measure_distractor_fixtures).",
                    "table_z": TABLE_Z,
                },
                "fixtures": fixture_geometry,
            }
            fgdest = pathlib.Path("src/libero_infinity/data/fixture_geometry.json")
            fgdest.write_text(json.dumps(fg_out, indent=2, sort_keys=True) + "\n")
            print(f"\nWrote {fgdest} with {len(fixture_geometry)} fixture classes:")
            for fclass, g in sorted(fixture_geometry.items()):
                print(f"  {fclass:24} {g}")

        vdest = pathlib.Path("src/libero_infinity/data/spawn_clearances_variants.json")
        vdest.write_text(json.dumps(vout, indent=2, sort_keys=False) + "\n")
        print(f"\nWrote {vdest} with {len(vout['clearances'])} (variant|surface) keys:")
        for key, v in vout["clearances"].items():
            print(f"  {key:44} {v:.4f}  (n={vout['_meta']['n_samples'].get(key, '?')})")
