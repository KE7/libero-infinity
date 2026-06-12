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
    # white_cabinet only ever appears as the GOAL fixture in the cabinet tasks
    # above (goal fixtures are excluded from distractor assignment), so it
    # produced no on-fixture rows. This SCENE5 task has white_cabinet as a
    # NON-goal fixture (goal is the plate), so distractors get assigned to it
    # and we measure real white_cabinet (class|fixture) rows + footprint/top_z.
    "libero_90/KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.bddl",
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

# Per-sample stability gate: RETIRED (validation_run2 RCA
# `distractor_z_convergence.md`).
#
# The old gate asserted the settled distractor's AABB *bottom* sits on the
# contacted fixture geom top within 50 mm. That silently assumes the body's AABB
# bottom *is* its contact surface — FALSE for irregular open-bottom distractors:
# a desk_caddy's open multi-compartment AABB hangs ~56 mm below its actual
# contact feet, so the gate *excluded its stable settle*, and the sparse
# surviving (atypical) sample under-stated the resting clearance by ~40 mm → the
# renderer injected it that far too low → it penetrated the cabinet top and was
# ejected (the "xy shove" Finding-B refuted).
#
# A live-stepping "is-it-quiescent?" replacement was tried and SEGFAULTS: the
# irregular distractor↔cabinet contact set overflows MuJoCo's contact arena
# (ncon = 5000) when stepped. So the gate is removed and its job is done two
# stepping-free ways instead, exactly as the gate-free audit validated:
#   * a per-sample admission of contact-existence + a physical clearance band
#     (below), which already rejects floated / bounced samples; and
#   * dominant-MODE aggregation over the pair's samples (``_dominant_mode``),
#     which is robust to a lone atypical settle dragging the row — the row is the
#     clearance the object rests at MOST often, i.e. the stable attractor the
#     renderer's injection converges to.
# The precise injected==settled guarantee remains independently verified by the
# smoke's 5 mm pose_tolerance over the merged data (a residual frame error would
# surface there). No tolerance was widened; the wrong criterion was removed.

# A measured fixture row is rewritten only when it diverges from the stored row by
# more than this (== the smoke's pose_tolerance). Rows within tolerance stay
# byte-identical, so validated box / table / fixture rows are preserved by
# construction and only genuinely-wrong rows are corrected.
_POSE_TOLERANCE = 0.005

# Distractor-on-fixture clearance must stay in a plausible physical band; a
# sample outside it means the object bounced off the fixture rather than resting.
_FIXTURE_CLEARANCE_MAX = 0.60

_DISTRACTOR_FIXTURE_SEEDS = 16


def _dominant_mode(values: list[float], *, bandwidth: float = 2 * _POSE_TOLERANCE) -> float:
    """Median of the largest single-linkage cluster of settled clearances.

    Robust to a lone atypical settle dragging the per-pair row. Irregular
    distractors can produce an occasional off-mode sample; the dominant MODE is
    the clearance the object rests at MOST often — the attractor the renderer's
    injection converges to — so injected z == settled z for the common case. For
    the tight (unimodal) box-distractor distributions the single cluster spans the
    whole sample, so this reduces exactly to the median (no change / no regression).
    """
    vs = sorted(values)
    if len(vs) <= 2:
        return float(statistics.median(vs))
    clusters: list[list[float]] = [[vs[0]]]
    for x in vs[1:]:
        if x - clusters[-1][-1] <= bandwidth:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    # Largest cluster wins; ties → the tighter (smaller-spread) cluster, i.e. the
    # more sharply-defined stable mode.
    best = max(clusters, key=lambda c: (len(c), -(c[-1] - c[0])))
    return float(statistics.median(best))


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
    boxes = [_body_world_aabb(sim, b) for b in _fixture_body_ids(sim, fixture_instance)]
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

    Per-pair rows are the dominant settle MODE over contacted, in-band samples
    (``_dominant_mode``); the retired AABB-bottom stability gate is gone (it
    wrongly excluded irregular open-bottom distractors — see RCA
    `distractor_z_convergence.md`). The injected==settled invariant is verified
    independently by the smoke's 5 mm pose_tolerance.
    """

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    avail = set(discover_all_tasks())
    samples: dict[str, list[float]] = {}
    table_samples: dict[str, list[float]] = {}
    fixture_footprints: dict[str, list[tuple[float, float, float]]] = {}
    fixture_rest_tops: dict[str, list[float]] = {}
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
                if not surface_class or not fixture_inst:
                    # Table-assigned distractor. Capture its clean table-resting
                    # body-origin clearance so the renderer's table slot resolves
                    # a MEASURED z instead of the DEFAULT_CLEARANCE prior. The
                    # distractor-only pool classes (desk_caddy, bowl_drainer,
                    # cookies, popcorn, alphabet_soup) are never task objects, so
                    # measure_variants/measure() never sees them — the distractor
                    # placement path is the ONLY way to measure their table z.
                    # That gap is the dominant table-distractor z error (e.g.
                    # desk_caddy injected 0.10 vs settled 0.42 = -320 mm).
                    # Guard with the physical table-contact check so a distractor
                    # whose (x, y) landed over a fixture and perched on it (the
                    # known table-distractor churn) is EXCLUDED, not mis-bucketed
                    # as a table rest.
                    clr = body_z - TABLE_Z
                    if 0.0 <= clr <= _FIXTURE_CLEARANCE_MAX and _settled_on_table_surface(env, nm):
                        table_samples.setdefault(cls, []).append(round(clr, 5))
                    continue
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
                # Per-sample admission (validation_run2 RCA
                # `distractor_z_convergence.md`): the sample is a valid on-fixture
                # rest iff it is in fixture contact (``tops`` non-empty, checked
                # above) and its clearance is in the physical band (checked above).
                # The retired AABB-bottom-vs-contact-top stability gate wrongly
                # excluded the stable settle of irregular open-bottom distractors
                # (desk_caddy's AABB hangs 56 mm below its contact feet), and a
                # live-stepping quiescence replacement SEGFAULTS on the irregular
                # distractor↔cabinet contact set (ncon overflow), so per-sample
                # stability is enforced instead by the dominant-MODE aggregation
                # below (robust to a lone atypical settle) + the smoke's 5 mm
                # pose_tolerance over the merged data (the independent
                # injected==settled / frame-error check).
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

    variant_rows = {k: round(_dominant_mode(v), 5) for k, v in sorted(samples.items())}
    table_rows = {c: round(statistics.median(v), 5) for c, v in sorted(table_samples.items())}
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

    print(
        f"\n# distractor-on-TABLE: {len(table_rows)} class rows "
        f"(measured via the table-distractor slot — the only path that covers "
        f"distractor-only pool classes):"
    )
    for c, v in table_rows.items():
        prior = table_clearances.get(c)
        chk = (
            f"  canonical={prior:.4f} Δ={abs(v - prior) * 1000:.1f}mm"
            if prior is not None
            else "  (NEW — no canonical row; was DEFAULT prior)"
        )
        print(f"  {c:28} {v:.4f} (n={len(table_samples[c])}){chk}")
    return variant_rows, fixture_geometry, table_rows


def _isolated_object_world_aabb(asset_class: str):
    """Union world-frame AABB of one distractor class loaded IN ISOLATION.

    Builds a minimal MuJoCo model containing only an ``EmptyArena`` and a single
    instance of ``asset_class`` (resolved from LIBERO's ``OBJECTS_DICT``), runs
    ``mj_forward`` (NO dynamics, NO settle, NO contacts), and unions the
    world-frame AABB over every geom belonging to the object. Returns
    ``(wx, wy, hz)`` extents in metres, or ``None`` if the class will not load.

    A distractor footprint is STATIC asset geometry — the geom-AABB extents of the
    loaded mesh — so this needs no scene generation and CANNOT overflow MuJoCo's
    contact arena (the failure mode that killed the prior scene-generation
    approach; see RCA ``proxy_footprint_measure.md``).
    """
    import math as _math

    import mujoco
    import numpy as np
    from libero.libero.envs.objects import get_object_fn
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.world import MujocoWorldBase

    try:
        obj = get_object_fn(asset_class)(name=f"{asset_class}_probe")
    except Exception as exc:  # noqa: BLE001 — unloadable class, recorded by caller
        print(f"# isolate build failed {asset_class}: {exc}")
        return None

    world = MujocoWorldBase()
    world.merge(EmptyArena())
    world.merge_assets(obj)
    world.worldbody.append(obj.get_obj())
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mins = np.full(3, _math.inf)
    maxs = np.full(3, -_math.inf)
    found = 0
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        # Object geoms are name-prefixed with the instance name; arena geoms
        # (floor/walls) never contain the asset class token.
        if gname is None or asset_class not in gname:
            continue
        # geom_aabb is the LOCAL (center, half) box; rotate into world via
        # geom_xmat/geom_xpos — identical method to ``_geom_world_aabb``.
        aabb = model.geom_aabb[gid]
        c_local = np.asarray(aabb[:3], dtype=float)
        half = np.asarray(aabb[3:], dtype=float)
        xpos = np.asarray(data.geom_xpos[gid], dtype=float)
        rot = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        center = xpos + rot @ c_local
        ext = np.array([sum(abs(rot[ax, k]) * half[k] for k in range(3)) for ax in range(3)])
        mins = np.minimum(mins, center - ext)
        maxs = np.maximum(maxs, center + ext)
        found += 1

    if found == 0:
        return None
    wx, wy, hz = (maxs - mins).tolist()
    return round(wx, 5), round(wy, 5), round(hz, 5)


def measure_distractor_footprints() -> dict:
    """Measure per-class distractor footprints from the STATIC asset geometry.

    A distractor footprint is fixed by the asset XML — it is the geom-AABB extent
    of the loaded mesh, not a scene-dependent quantity. So each pool class is
    loaded IN ISOLATION (``_isolated_object_world_aabb``: EmptyArena + the single
    object, ``mj_forward``, read the geom world-AABB) and we record:

        wx = xmax-xmin,  wy = ymax-ymin,  hz = zmax-zmin   (canonical resting pose)

    and the circumscribed planar half-extent ``r = 0.5·sqrt(wx²+wy²)`` — the
    yaw-ROBUST half-extent threaded into every clearance constraint
    (``asset_metadata.distractor_planar_half``). The LIBERO assets are authored in
    their resting pose, and the pool-fit rejection (the structural fix) keeps
    oversized irregular classes off undersized fixtures — so they rest flat on the
    table in this canonical pose and the static footprint is exactly what the
    settled object presents. This replaces the prior scene-generation measurement,
    which overflowed MuJoCo's ncon contact arena settling irregular distractors
    onto fixtures (RCA ``proxy_footprint_measure.md``). Measurement is
    deterministic — one observation per class (``n = 1``).

    Writes ``data/distractor_geometry.json`` consumed by
    ``asset_metadata.distractor_footprint`` / ``distractor_planar_half``.
    """
    import math as _math

    from libero_infinity.asset_registry import DEFAULT_DISTRACTOR_POOL

    distractors: dict[str, dict] = {}
    for cls in sorted(DEFAULT_DISTRACTOR_POOL):
        extents = _isolated_object_world_aabb(cls)
        if extents is None:
            print(f"# WARNING: no geom AABB for distractor class {cls!r}")
            continue
        wx, wy, hz = extents
        # Reject degenerate / un-loaded AABBs.
        if not (0.0 < wx < 1.0 and 0.0 < wy < 1.0 and 0.0 < hz < 1.0):
            print(f"# WARNING: degenerate AABB for {cls!r}: {extents}")
            continue
        distractors[cls] = {
            "footprint": [wx, wy],
            "height": hz,
            "radius": round(0.5 * _math.hypot(wx, wy), 5),
            "n": 1,
        }

    missing = [c for c in DEFAULT_DISTRACTOR_POOL if c not in distractors]
    print(f"\n# distractor footprints: {len(distractors)} classes measured")
    for cls, g in distractors.items():
        print(
            f"  {cls:24} radius={g['radius']:.4f} h={g['height']:.4f} "
            f"footprint={g['footprint']} (n={g['n']})"
        )
    if missing:
        print(f"# WARNING: pool classes with no footprint row: {missing}")

    return {
        "_meta": {
            "description": "Measured per-class distractor footprints. footprint=[w,l] "
            "and height are the STATIC geom-AABB world extents (m) of the asset loaded "
            "in isolation (EmptyArena + object, mj_forward — no scene, no settle); "
            "radius is the yaw-robust circumscribed planar half-extent "
            "0.5*sqrt(w^2+l^2) threaded into renderer clearance + support-fit. "
            "Generated by scripts/measure_spawn_clearances.py "
            "(measure_distractor_footprints).",
            "n_classes": len(distractors),
            "missing_pool_classes": missing,
        },
        "distractors": distractors,
    }


def _run_distractor_footprints_only() -> None:
    """Measure ONLY per-class distractor footprints and write
    ``data/distractor_geometry.json``, leaving all clearance data untouched."""
    out = measure_distractor_footprints()
    dest = pathlib.Path("src/libero_infinity/data/distractor_geometry.json")
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {dest} with {len(out['distractors'])} distractor classes.")


# ---------------------------------------------------------------------------
# Unmeasured SUPPORT-fixture geometry — isolated static-AABB + in-scene rest top
# ---------------------------------------------------------------------------
#
# WS-1 (hardcoding audit): only 4 of the corpus's support fixtures carried
# MEASURED geometry (flat_stove, microwave, white_cabinet, wooden_cabinet); the
# rest (wine_rack, wooden_two_layer_shelf, desk_caddy) fell back to the
# hand-coded ``asset_metadata._FIXTURE_DIMS_FALLBACK`` (microwave shows ~131 %
# width error vs measured; wine_rack's fallback 0.18×0.12 vs measured 0.27×0.33
# is so small that NO distractor pool class "fits" it, so distractors are never
# seated on the rack at all). This path measures the missing fixtures WITHOUT
# the contact-arena overflow that killed prior agents:
#
#   * ``footprint`` [w, l], ``height``, ``offset`` [dx, dy] — STATIC asset
#     geometry, read by loading the fixture IN ISOLATION (EmptyArena + the single
#     welded fixture, ``mj_forward``, geom world-AABB; ``_isolated_fixture_world_aabb``).
#     Deterministic, runs in seconds, and STRUCTURALLY cannot overflow the contact
#     arena (no scene, no dynamics, no contacts). Cross-validated against the 4
#     already-measured fixtures: the isolated AABB reproduces their stored
#     footprint/height/offset for the NON-articulated fixtures (the missing 3 are
#     all non-articulated, so isolated == in-scene; articulated microwave/
#     white_cabinet differ only because their stored rows captured the OPEN
#     door/drawer envelope, which is irrelevant here).
#
#   * ``top_z`` — the REST surface a centrally-placed object/distractor settles
#     onto, expressed above the ``TABLE_Z`` constant. This is the one quantity
#     that needs the IN-SCENE vertical placement (the fixture's authored seat on
#     the table), so it is read from a real scene built with the ``position``
#     perturbation (NO distractors — so NO irregular-distractor settling, the
#     overflow trigger) and SUBPROCESS-ISOLATED per (fixture, task) so any crash
#     loses only that sample. The rest surface is the highest *substantial*
#     collision geom (``_FIXTURE_REST_MIN_GEOM_EXTENT``) — a thin post/divider
#     cannot support a resting object, it straddles to the broad surface below.
#     VALIDATED: this rule reproduces the stored flat_stove top_z (0.13502)
#     EXACTLY — the distractor-contact method's value was the central raised
#     grate, which is precisely the highest substantial collision geom.

# Minimum horizontal half? NO — minimum FULL horizontal extent (m) a collision
# geom must have in BOTH x and y to count as a rest surface. The smallest
# distractor/object footprints span ~50–80 mm, so a collision geom narrower than
# 40 mm in either axis cannot seat a resting object (it straddles/tips and the
# object comes to rest on the broad surface below). 40 mm is comfortably below
# the smallest real footprint yet above every thin divider/post/rim wall in the
# corpus fixtures. Validated: reproduces the stored flat_stove top_z exactly.
_FIXTURE_REST_MIN_GEOM_EXTENT = 0.04


def _isolated_fixture_world_aabb(fixture_class: str):
    """Static geom-AABB of one FIXTURE class loaded IN ISOLATION.

    Like :func:`_isolated_object_world_aabb`, but fixtures are immovable bodies
    with no inertial mass, so their free joint is stripped (the fixture is welded
    to the worldbody) before ``mj_forward`` — otherwise MuJoCo rejects the
    massless moving body. The fixture body is appended at the origin, so the
    geom-AABB *center* xy IS the fixture's body-origin → geom offset (the same
    quantity ``measure_fixture_offsets.py`` reads in-scene as
    ``aabb_center - init_xy``; cross-checked: flat_stove isolated center x=0.0945
    vs stored offset 0.0947).

    Returns ``{"footprint": [w, l], "height": h, "offset": [dx, dy]}`` (all
    rounded, m) over the fixture's **collision** geoms only — or ``None`` if the
    class will not load. No scene, no dynamics, no contacts → cannot overflow.

    Collision-only is deliberate (audit WS-1 g3): the footprint is consumed as a
    PLACEMENT/CLEARANCE keepout (renderer object↔fixture and robot↔fixture boxes,
    and the goal-region distractor keepout). Only collision geometry physically
    blocks placement — a visual-only overhang (a decorative shell, an open-frame
    side panel) does NOT, so unioning visual geoms over-inflates the keepout and
    spuriously fails Scenic generation in budget (g3). For the corpus's
    non-articulated support fixtures the visual mesh overhangs the collision body
    by up to ~0.19 m (desk_caddy width, wooden_two_layer_shelf length), which
    drove a −7.4 pp g3 regression; the collision AABB is the true seating/keepout
    extent. (The already-stored articulated fixtures — flat_stove, microwave,
    {white,wooden}_cabinet — keep their in-scene door/drawer-OPEN envelopes and
    are not re-measured through this path; the cross-check skips them.)
    """
    import math as _math

    import mujoco
    import numpy as np
    from libero.libero.envs.objects import get_object_fn
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.world import MujocoWorldBase

    try:
        obj = get_object_fn(fixture_class)(name=f"{fixture_class}_probe")
    except Exception as exc:  # noqa: BLE001 — unloadable class, recorded by caller
        print(f"# isolate build failed {fixture_class}: {exc}")
        return None

    world = MujocoWorldBase()
    world.merge(EmptyArena())
    world.merge_assets(obj)
    body = obj.get_obj()
    # Strip free/articulation joints → a welded static body (no mass required).
    for j in list(body.findall("joint")) + list(body.findall("freejoint")):
        body.remove(j)
    world.worldbody.append(body)
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mins = np.full(3, _math.inf)
    maxs = np.full(3, -_math.inf)
    found = 0
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if gname is None or fixture_class not in gname:
            continue
        # Skip visual-only geoms (contype == conaffinity == 0): they never make a
        # contact, so they cannot block a placed object — including them would
        # over-inflate the keepout footprint (audit WS-1 g3 regression).
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        aabb = model.geom_aabb[gid]
        c_local = np.asarray(aabb[:3], dtype=float)
        half = np.asarray(aabb[3:], dtype=float)
        xpos = np.asarray(data.geom_xpos[gid], dtype=float)
        rot = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        center = xpos + rot @ c_local
        ext = np.array([sum(abs(rot[ax, k]) * half[k] for k in range(3)) for ax in range(3)])
        mins = np.minimum(mins, center - ext)
        maxs = np.maximum(maxs, center + ext)
        found += 1

    if found == 0:
        return None
    wx, wy, hz = (maxs - mins).tolist()
    cx = (mins[0] + maxs[0]) / 2.0
    cy = (mins[1] + maxs[1]) / 2.0
    return {
        "footprint": [round(wx, 5), round(wy, 5)],
        "height": round(hz, 5),
        "offset": [round(float(cx), 5), round(float(cy), 5)],
    }


def _fixture_rest_top_inscene(task_rel: str, fixture_class: str, *, seed: int = 0):
    """In-scene REST-surface top_z above ``TABLE_Z`` for ``fixture_class``.

    Builds the task with the ``position`` perturbation (NO distractors → NO
    irregular-distractor settling, the contact-arena overflow trigger), resets,
    and returns the highest *substantial* collision-geom top of the fixture
    minus ``TABLE_Z`` (the surface a centrally-placed object rests on). Returns
    ``None`` when the fixture instance / a substantial surface is absent. Meant
    to be run in a SUBPROCESS (``--fixture-topz-worker``) so a crash is isolated.
    """
    import numpy as np

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.ir.graph_builder import build_semantic_scene_graph
    from libero_infinity.ir.nodes import FixtureNode
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path(task_rel))
    cfg = TaskConfig.from_bddl(bddl)
    graph = build_semantic_scene_graph(cfg)
    insts = [
        getattr(n, "instance_name", None)
        for n in graph.nodes.values()
        if isinstance(n, FixtureNode) and getattr(n, "object_class", None) == fixture_class
    ]
    insts = [i for i in insts if i]
    if not insts:
        return None
    random.seed(seed)
    scene, _ = compile_task_to_scenario(cfg, "position").generate(maxIterations=8000)
    env = make_env(scene, bddl_path=bddl)
    env.reset()
    sim = env._sim.libero_env.env.sim  # noqa: SLF001
    model = sim.model
    raw = model._model  # noqa: SLF001 — MjModel for geom_aabb
    best = None
    for inst in insts:
        for gid in range(model.ngeom):
            try:
                gname = model.geom_id2name(gid)
            except Exception:
                gname = None
            if gname is None:
                continue
            bid = model.geom_bodyid[gid]
            bname = model.body_id2name(bid)
            if bname is None or not (bname == inst or bname.startswith(inst + "_")):
                continue
            if int(model.geom_contype[gid]) == 0:
                continue  # visual-only geom — not a contact/rest surface
            aabb = raw.geom_aabb[gid]
            c_local = np.asarray(aabb[:3], dtype=float)
            half = np.asarray(aabb[3:], dtype=float)
            xpos = np.asarray(sim.data.geom_xpos[gid], dtype=float)
            rot = np.asarray(sim.data.geom_xmat[gid], dtype=float).reshape(3, 3)
            center = xpos + rot @ c_local
            ext = np.array([sum(abs(rot[ax, k]) * half[k] for k in range(3)) for ax in range(3)])
            wx, wy = float(2 * ext[0]), float(2 * ext[1])
            if min(wx, wy) < _FIXTURE_REST_MIN_GEOM_EXTENT:
                continue  # thin post/divider — cannot seat a resting object
            ztop = float(center[2] + ext[2])
            best = ztop if best is None else max(best, ztop)
    env.close()
    if best is None:
        return None
    return round(best - TABLE_Z, 5)


def _fixture_topz_subprocess(task_rel: str, fixture_class: str):
    """Run :func:`_fixture_rest_top_inscene` in an ISOLATED subprocess.

    The in-scene read does not live-step irregular distractors (it uses the
    distractor-free ``position`` perturbation), but per the WS-1 mandate any
    in-scene measurement is subprocess-isolated so a MuJoCo segfault on one
    (fixture, task) cannot abort the whole run. Returns the measured top_z float
    or ``None``.
    """
    import os
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, __file__, "--fixture-topz-worker", task_rel, fixture_class],
        capture_output=True,
        text=True,
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": "src"},
        timeout=600,
    )
    if proc.returncode != 0:
        print(f"# topz subprocess crashed {fixture_class} @ {task_rel} (rc={proc.returncode})")
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("TOPZ_RESULT "):
            val = line.split(" ", 1)[1].strip()
            return None if val == "None" else float(val)
    return None


def _corpus_support_fixtures() -> dict[str, list[str]]:
    """Discover the support fixtures (non-table FixtureNode classes) used across
    the whole task corpus, mapped to up to 3 representative task ids each.

    The authoritative "real on-fixture support surface" set the renderer can seat
    objects/distractors on — every non-table fixture class present as a graph
    ``FixtureNode``. The workspace tables/floor (arena surfaces, handled by #29)
    are excluded.
    """
    from libero_infinity.ir.graph_builder import build_semantic_scene_graph
    from libero_infinity.ir.nodes import ArticulationModel, FixtureNode
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    tables = set(ArticulationModel.canonical().root_workspace_fixtures)
    out: dict[str, list[str]] = {}
    for task_rel in discover_all_tasks():
        try:
            cfg = TaskConfig.from_bddl(str(resolve_task_path(task_rel)))
            graph = build_semantic_scene_graph(cfg)
        except Exception:  # noqa: BLE001 — unbuildable task, skipped
            continue
        classes = {
            getattr(n, "object_class", None)
            for n in graph.nodes.values()
            if isinstance(n, FixtureNode)
        }
        for cls in classes:
            if not cls or cls in tables:
                continue
            tasks = out.setdefault(cls, [])
            if len(tasks) < 3:
                tasks.append(task_rel)
    return out


def measure_support_fixtures(only_unmeasured: bool = True) -> dict[str, dict]:
    """Measure the corpus support fixtures' static geometry + rest top_z.

    For each non-table fixture class in the corpus (optionally only those not yet
    in the stored ``fixture_geometry.json``): footprint/height/offset from the
    ISOLATED static-AABB and top_z from the SUBPROCESS-ISOLATED in-scene rest
    surface, dominant-MODE aggregated over the representative tasks. Returns
    ``{fixture_class: {"footprint": [w,l], "offset": [dx,dy], "height": h,
    "top_z": z}}``.
    """
    from libero_infinity import asset_metadata

    corpus = _corpus_support_fixtures()
    measured_keys = set(asset_metadata.FIXTURE_GEOMETRY)
    targets = {
        c: ts for c, ts in sorted(corpus.items()) if not (only_unmeasured and c in measured_keys)
    }
    print(
        f"# corpus support fixtures: {sorted(corpus)}\n"
        f"# already measured: {sorted(measured_keys & set(corpus))}\n"
        f"# measuring: {sorted(targets)}"
    )
    result: dict[str, dict] = {}
    for fclass, tasks in targets.items():
        static = _isolated_fixture_world_aabb(fclass)
        if static is None:
            print(f"# WARNING: no isolated AABB for fixture {fclass!r} — skipped")
            continue
        tops: list[float] = []
        for task_rel in tasks:
            tz = _fixture_topz_subprocess(task_rel, fclass)
            if tz is not None:
                tops.append(tz)
                print(f"  {fclass:24} top_z={tz:+.5f} @ {task_rel.split('/')[-1][:46]}")
        if not tops:
            print(f"# WARNING: no in-scene top_z for fixture {fclass!r} — skipped")
            continue
        entry = {
            "footprint": static["footprint"],
            "offset": static["offset"],
            "height": static["height"],
            "top_z": round(_dominant_mode(tops), 5),
        }
        result[fclass] = entry
        print(f"# {fclass}: {entry}  (top_z n={len(tops)})")
    return result


def _crosscheck_isolated_vs_stored() -> None:
    """Sanity: confirm the isolated collision-AABB reproduces each NON-articulated
    fixture's stored footprint/height within tolerance (does NOT write — pure
    validation that the isolated frame matches the stored frame). The corpus's
    articulated originals (flat_stove, microwave, {white,wooden}_cabinet) store
    door/drawer-OPEN envelopes and are skipped; the non-articulated support
    fixtures (desk_caddy, wine_rack, wooden_two_layer_shelf) are validated here.
    """
    from libero_infinity import asset_metadata
    from libero_infinity.ir.nodes import ArticulationModel

    articulated = set(ArticulationModel.canonical().fixture_families)
    print("\n# cross-check isolated AABB vs stored (non-articulated only):")
    for fclass, stored in sorted(asset_metadata.FIXTURE_GEOMETRY.items()):
        iso = _isolated_fixture_world_aabb(fclass)
        if iso is None:
            continue
        note = " [ARTICULATED: stored row = open envelope, skip]" if fclass in articulated else ""
        sf = stored.get("footprint")
        if sf and not note:
            dw = abs(sf[0] - iso["footprint"][0]) * 1000
            dl = abs(sf[1] - iso["footprint"][1]) * 1000
            flag = "  <-- DIVERGES" if max(dw, dl) > 5 else ""
            print(
                f"  {fclass:24} stored_fp={sf} iso_fp={iso['footprint']} "
                f"Δ=({dw:.1f},{dl:.1f})mm{flag}"
            )
        else:
            print(f"  {fclass:24} stored_fp={sf} iso_fp={iso['footprint']}{note}")


# ---------------------------------------------------------------------------
# Distractor settle-z: the height a FLAT distractor actually rests at on a
# fixture top (WS-1 open-frame seating fix).
# ---------------------------------------------------------------------------
#
# ``top_z`` is the highest substantial COLLISION-geom edge of the fixture top.
# For a FLAT-topped fixture a flat object rests right on it, but for an OPEN-FRAME
# fixture (wine_rack, two-layer shelf, desk_caddy) a flat object sinks between the
# top rails until it catches lower, so it settles BELOW ``top_z`` (WS-1 RCA:
# wine_rack top_z≈0.396 collision edge vs ≈0.361 flat-settle rest → 35 mm drift →
# the on-rack distractor fails pose_tolerance). The renderer's rule-2 ANALYTIC
# on-fixture distractor z (``top_z + on_table``) is therefore too high for those
# fixtures.
#
# settle_top_z is the analytic-consistent rest surface measured directly from a
# real settle: drop the scene's flat distractors onto the fixture, read each
# settled body z, and subtract its own flat-table body-origin clearance and the
# table z. That delta is the contact-surface height the renderer must use so that
#   inject_z = TABLE_Z + settle_top_z + on_table(class) == settled body z.
# It is measured ONLY for the analytic-path fixtures (those with NO measured
# per-(class|fixture) variant row); the rule-1 fixtures (flat_stove, microwave,
# {white,wooden}_cabinet) already inject at their measured per-pair settle, never
# consult the analytic, and are left byte-identical. The merge only writes
# settle_top_z when it diverges from top_z by > pose_tolerance, so a fixture that
# happens to settle ≈ top_z stays byte-identical too (no spurious field).

_SETTLE_Z_SEEDS = 8

# Candidate corpus tasks per analytic-path open-frame fixture where the fixture
# is present as a NON-goal FixtureNode (goal fixtures are excluded from distractor
# assignment, so a distractor only seats on a non-goal one). Discovered with
# scripts/_diag_find_fixture_tasks.py.
_SETTLE_Z_TASKS: dict[str, list[str]] = {
    "wine_rack": [
        "libero_goal/put_the_bowl_on_the_stove.bddl",
        "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl",
        "libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
    ],
    "desk_caddy": [
        "libero_90/STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy.bddl",
        "libero_90/STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
        "libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
    ],
    "wooden_two_layer_shelf": [
        "libero_90/KITCHEN_SCENE9_turn_on_the_stove.bddl",
        "libero_90/KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet.bddl",
        "libero_90/KITCHEN_SCENE9_put_the_frying_pan_on_the_stove.bddl",
    ],
}


def _settle_z_inscene(task_rel: str, fixture_class: str, *, seeds: int = _SETTLE_Z_SEEDS):
    """Per-sample distractor settle deltas on ``fixture_class`` for ``task_rel``.

    Builds the ``distractor`` perturbation, resets (inject + settle), and for each
    active distractor whose ``support_surface_class`` is ``fixture_class`` returns
    ``settled_body_z - on_table_clearance(class) - TABLE_Z`` — the analytic-
    consistent contact-surface height above the table (distractor-independent for
    a given fixture). Meant to run in a SUBPROCESS so the irregular-distractor
    contact-arena (ncon) overflow stays isolated.
    """
    from libero_infinity.asset_metadata import spawn_clearance
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path(task_rel))
    deltas: list[float] = []
    for seed in range(seeds):
        try:
            cfg = TaskConfig.from_bddl(bddl)
            random.seed(seed)
            scene, _ = compile_task_to_scenario(cfg, "distractor").generate(maxIterations=8000)
            env = make_env(scene, bddl_path=bddl)
            env.reset()
        except Exception as exc:  # noqa: BLE001 — measurement noise, recorded
            print(f"# settle-z build failed {task_rel} [seed {seed}]: {exc}")
            continue
        sim = env._sim.libero_env.env.sim  # noqa: SLF001
        active = getattr(env._sim, "_active_distractor_names", set())  # noqa: SLF001
        for o in scene.objects:
            nm = getattr(o, "libero_name", "")
            if not nm.startswith("distractor_") or nm not in active:
                continue
            if (getattr(o, "support_surface_class", "") or "") != fixture_class:
                continue
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
            delta = body_z - spawn_clearance(cls, None) - TABLE_Z
            if 0.0 <= delta <= _FIXTURE_CLEARANCE_MAX:
                deltas.append(round(delta, 5))
        env.close()
    return deltas


def _settle_z_subprocess(task_rel: str, fixture_class: str) -> list[float]:
    """Run :func:`_settle_z_inscene` in an ISOLATED subprocess (ncon-overflow
    containment). Returns the list of per-sample settle deltas (possibly empty)."""
    import os
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, __file__, "--settle-z-worker", task_rel, fixture_class],
        capture_output=True,
        text=True,
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": "src"},
        timeout=900,
    )
    if proc.returncode != 0:
        print(f"# settle-z subprocess crashed {fixture_class} @ {task_rel} (rc={proc.returncode})")
        return []
    out: list[float] = []
    for line in proc.stdout.splitlines():
        if line.startswith("SETTLEZ "):
            try:
                out.append(float(line.split(" ", 2)[2]))
            except (ValueError, IndexError):
                continue
    return out


def measure_distractor_settle_z() -> dict[str, float]:
    """Settle-measured ``settle_top_z`` for every analytic-path open-frame fixture.

    For each fixture in ``fixture_geometry.json`` that has NO measured
    per-(class|fixture) variant row (so the renderer resolves its on-fixture
    distractor z analytically), drop the scene's distractors onto it across a few
    representative non-goal tasks (subprocess-isolated) and take the dominant
    settle MODE of the contact-surface deltas. Returns ``{fixture: settle_top_z}``;
    fixtures with too few samples are skipped (and warned).
    """
    from libero_infinity import asset_metadata

    measured_pairs = {k.split("|", 1)[1] for k in asset_metadata.VARIANT_CLEARANCES if "|" in k}
    analytic = sorted(f for f in asset_metadata.FIXTURE_GEOMETRY if f not in measured_pairs)
    print(f"# analytic-path fixtures (settle_top_z candidates): {analytic}")
    out: dict[str, float] = {}
    for fclass in analytic:
        tasks = _SETTLE_Z_TASKS.get(fclass, [])
        if not tasks:
            print(f"# WARNING: no candidate tasks for {fclass!r} — skipped (stays on top_z)")
            continue
        samples: list[float] = []
        for task_rel in tasks:
            ds = _settle_z_subprocess(task_rel, fclass)
            if ds:
                print(f"  {fclass:24} n={len(ds):3} mode={_dominant_mode(ds):.5f} @ "
                      f"{task_rel.split('/')[-1][:46]}")
                samples.extend(ds)
        if len(samples) < 3:
            print(f"# WARNING: only {len(samples)} settle samples for {fclass!r} — "
                  f"skipped (stays on top_z)")
            continue
        out[fclass] = round(_dominant_mode(samples), 5)
        top_z = asset_metadata.FIXTURE_GEOMETRY[fclass].get("top_z")
        drop = (top_z - out[fclass]) * 1000 if top_z is not None else float("nan")
        print(f"# {fclass}: settle_top_z={out[fclass]:.5f} (top_z={top_z}, "
              f"{drop:+.1f}mm below edge, n={len(samples)})")
    return out


def _run_distractor_settle_z() -> None:
    """Measure ``settle_top_z`` for the analytic-path open-frame fixtures and
    ADDITIVELY merge it into ``data/fixture_geometry.json``.

    Merge discipline (WS-1): a fixture gets a ``settle_top_z`` field ONLY when it
    diverges from the fixture's ``top_z`` by > ``_POSE_TOLERANCE`` (an open frame).
    A fixture that settles ≈ ``top_z`` is left byte-identical (no field). The
    rule-1 fixtures are never scanned, so flat_stove/{white,wooden}_cabinet/
    microwave rows stay byte-identical. No other field, fixture, or data file is
    touched. The file is re-dumped in its existing insertion order so untouched
    rows are byte-for-byte preserved.
    """
    settle = measure_distractor_settle_z()
    fg_path = pathlib.Path("src/libero_infinity/data/fixture_geometry.json")
    canon = json.loads(fg_path.read_text())
    fixtures = canon.get("fixtures", {})
    changed: list[str] = []
    for fclass, sz in settle.items():
        row = fixtures.get(fclass)
        if row is None:
            continue
        top_z = row.get("top_z")
        if top_z is not None and abs(top_z - sz) <= _POSE_TOLERANCE:
            print(f"# {fclass}: settle_top_z {sz:.5f} ≈ top_z {top_z:.5f} — left byte-identical")
            continue
        if row.get("settle_top_z") == sz:
            continue
        row["settle_top_z"] = sz
        changed.append(fclass)
    if not changed:
        print("\nNo fixture settle_top_z diverged — fixture_geometry.json untouched.")
        return
    meta = canon.get("_meta", {})
    if "settle_top_z" not in meta.get("description", ""):
        meta["description"] = (
            meta.get("description", "").rstrip()
            + " settle_top_z (open-frame fixtures only) is the settle-measured "
            "height a FLAT distractor actually rests at on the fixture top — below "
            "top_z for open frames (rails) — used for the rule-2 analytic on-fixture "
            "DISTRACTOR z; absent ⇒ falls back to top_z. Generated by "
            "scripts/measure_spawn_clearances.py --distractor-settle-z."
        )
        canon["_meta"] = meta
    fg_path.write_text(json.dumps(canon, indent=2) + "\n")
    print(f"\nWrote {fg_path}: added settle_top_z to {changed} "
          f"(all other rows byte-identical).")
    for fclass in changed:
        print(f"  {fclass:24} {fixtures[fclass]}")


def _run_support_fixtures_only() -> None:
    """Measure the UNMEASURED corpus support fixtures and ADDITIVELY merge them
    into ``data/fixture_geometry.json``.

    Corrective-merge discipline (WS-1 task D): the already-measured rows (the 4
    fixtures + #29's offset rows) are preserved BYTE-IDENTICAL — a fixture is
    written only if it is ABSENT, or (when re-measuring all) its value diverges
    from stored by > ``_POSE_TOLERANCE``. Touches NO other data file
    (distractor_geometry.json, spawn_clearances*.json, arena-table rows are all
    untouched).
    """
    _crosscheck_isolated_vs_stored()
    new_geom = measure_support_fixtures(only_unmeasured=True)
    fg_path = pathlib.Path("src/libero_infinity/data/fixture_geometry.json")
    canon = json.loads(fg_path.read_text())
    fixtures = canon.setdefault("fixtures", {})
    added: list[str] = []
    rewritten: list[str] = []
    for fclass, entry in new_geom.items():
        old = fixtures.get(fclass)
        if old is None:
            fixtures[fclass] = entry
            added.append(fclass)
            continue
        # Re-measure guard (only triggers if not only_unmeasured): rewrite a
        # stored row solely when it diverges > pose_tolerance on any scalar.
        diverged = (
            abs(old.get("top_z", 0.0) - entry["top_z"]) > _POSE_TOLERANCE
            or abs(old.get("height", 0.0) - entry["height"]) > _POSE_TOLERANCE
            or abs(old.get("footprint", [0, 0])[0] - entry["footprint"][0]) > _POSE_TOLERANCE
            or abs(old.get("footprint", [0, 0])[1] - entry["footprint"][1]) > _POSE_TOLERANCE
        )
        if diverged:
            fixtures[fclass] = entry
            rewritten.append(fclass)
    if not added and not rewritten:
        print("\nNo fixtures added or diverged — fixture_geometry.json untouched.")
        return
    canon["fixtures"] = fixtures
    fg_path.write_text(json.dumps(canon, indent=2) + "\n")
    print(
        f"\nWrote {fg_path}: added {added}, rewrote {rewritten} "
        f"(existing rows preserved byte-identical)."
    )
    for fclass in added + rewritten:
        print(f"  {fclass:24} {fixtures[fclass]}")


# ---------------------------------------------------------------------------
# Per-arena workspace-table clearances (non-reference arenas)
# ---------------------------------------------------------------------------
#
# The canonical ``spawn_clearances.json`` is measured on the KITCHEN / default
# tabletop arena (table top ≈ 0.90 m), and ``surface_spawn_z`` resolves the
# kitchen settled z exactly. But LIBERO's NON-reference arenas place objects at a
# materially different settled z that is NOT a rigid translation of the kitchen
# pose: the living-room table sits ~0.49 m lower AND LIBERO seats several tall
# objects at an elevated (metastable) reset pose (e.g. ketchup ~0.20 m above the
# living-room table top vs ~0.07 m above the kitchen table top). The
# ``arena_surface_z`` shift alone (reusing the kitchen clearance) therefore leaves
# a per-object residual of up to ~130 mm, which still fails pose_tolerance
# (RCA task_robot_shove.md §4). The fix is the SAME methodology that made kitchen
# exact: measure the per-(class, arena-table) settled clearance from the real
# LIBERO reset (the pose pose_tolerance compares against), keyed by the arena's
# workspace-table class and expressed relative to that arena's
# ``arena_surface_z`` so the renderer's emitted z == the simulator's realized z.
#
# Only NON-reference arenas need this: kitchen / default ``table`` are the
# reference (canonical clearance), and the study table is high enough that
# ``arena_surface_z`` + canonical already lands within tolerance. The reference
# arenas are intentionally NOT remeasured here so their validated rows stay
# byte-identical.
_ARENA_TABLE_MEASURE_TASKS: dict[str, list[str]] = {
    "living_room_table": [
        "libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate.bddl",
        "libero_90/LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
        "libero_90/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate.bddl",
    ],
}

# Settled-clearance plausibility band for an arena table (m, relative to the
# arena's ``arena_surface_z``). Wider than the kitchen band because LIBERO seats
# some tall objects elevated/metastable on the lower tables (ketchup ≈ 0.28).
_ARENA_CLEARANCE_BAND = (0.0, 0.45)
_ARENA_TABLE_SEEDS = (0, 1, 2)


def measure_arena_tables() -> dict[str, float]:
    """Measure per-(class, arena-table) settled clearance for NON-reference arenas.

    For each arena task, generate a ``position`` scene, reset the real LIBERO env
    (the same path the validation pipeline / pose_tolerance use), and record each
    table-resting movable's ``settled_z − arena_surface_z(arena_table_class)``,
    bucketed by ``"<asset_class>|<arena_table_class>"``. Aggregated by the
    dominant settle mode (robust to a stray fixture-perched sample), over a few
    deterministic seeds. The result is merged into
    ``spawn_clearances_variants.json`` so ``surface_spawn_z`` resolves the arena's
    settled z when the renderer threads the arena-table class as the surface.

    No table-contact guard is applied: an arena's own ``:init`` regions target
    the workspace table, so an object's LIBERO reset pose IS the ground-truth
    table-resting pose pose_tolerance scores — even when LIBERO seats it at an
    elevated metastable rest. Dominant-mode aggregation + multi-seed handle any
    xy sample that lands over a fixture.
    """
    from libero_infinity.asset_metadata import arena_surface_z
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import (
        is_scene_fixture,
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    avail = set(discover_all_tasks())
    samples: dict[str, list[float]] = {}
    lo, hi = _ARENA_CLEARANCE_BAND
    for arena_table, tasks in _ARENA_TABLE_MEASURE_TASKS.items():
        surf_z = arena_surface_z(arena_table)
        for task_rel in tasks:
            if task_rel not in avail:
                print(f"# SKIP (not found): {task_rel}")
                continue
            bddl = str(resolve_task_path(task_rel))
            for seed in _ARENA_TABLE_SEEDS:
                try:
                    cfg = TaskConfig.from_bddl(bddl)
                    random.seed(seed)
                    scenario = compile_task_to_scenario(cfg, "position")
                    scene, _ = scenario.generate(maxIterations=20000)
                    env = make_env(scene, bddl_path=bddl)
                    env.reset()
                except Exception as exc:  # noqa: BLE001 — measurement noise, recorded
                    print(f"# build failed {task_rel} [seed {seed}]: {exc}")
                    continue
                for o in _iter_scene_objects(scene):
                    if is_scene_fixture(o) or not getattr(o, "graspable", True):
                        continue
                    # Skip contained / fixture-supported children — their z
                    # derives from a support relation, not the arena table.
                    sp = getattr(o, "support_parent_name", "")
                    if sp and "table" not in sp.lower():
                        continue
                    nm = resolve_object_name(o) or "?"
                    cls = getattr(o, "asset_class", None)
                    if not cls:
                        continue
                    st = env.get_object_state(nm)
                    if st is None:
                        continue
                    clearance = float(st["position"][2]) - surf_z
                    if not (lo <= clearance <= hi):
                        continue
                    samples.setdefault(f"{cls}|{arena_table}", []).append(round(clearance, 5))
                env.close()

    rows = {k: round(_dominant_mode(v), 5) for k, v in sorted(samples.items())}
    print(f"\n# arena-table clearance rows ({len(rows)}):")
    for k, v in rows.items():
        print(f"  {k:44} {v:.4f}  (n={len(samples[k])})")
    return rows


def _merge_arena_table_rows(variants_path: pathlib.Path, rows: dict[str, float]) -> dict:
    """Corrective-merge arena-table clearance rows into the variants file.

    Rewrites a row only when it diverges from the stored value by more than the
    smoke pose_tolerance (so already-correct rows stay byte-identical), and adds
    new rows. Never touches non-arena-table keys.
    """
    data = json.loads(variants_path.read_text())
    clearances: dict[str, float] = data.get("clearances", {})
    changed: dict[str, tuple[float | None, float]] = {}
    for key, z in rows.items():
        old = clearances.get(key)
        if old is None or abs(float(old) - z) > _POSE_TOLERANCE:
            changed[key] = (None if old is None else float(old), z)
            clearances[key] = z
    data["clearances"] = {k: clearances[k] for k in sorted(clearances)}
    data.setdefault("_meta", {})["n_arena_table_rows"] = len(rows)
    variants_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    print(
        f"\nMerged {len(changed)} arena-table row(s) into {variants_path} "
        f"(>{_POSE_TOLERANCE * 1000:.0f}mm divergence; rest byte-identical):"
    )
    for k, (old, z) in sorted(changed.items()):
        delta = "NEW" if old is None else f"{(z - old) * 1000:+.1f}mm"
        print(f"  {k:44} {('—' if old is None else f'{old:.5f}'):>9} -> {z:.5f}  (Δ={delta})")
    return changed


def _run_arena_tables_only() -> None:
    """Measure ONLY per-(class, arena-table) clearances for non-reference arenas
    and corrective-merge them into ``spawn_clearances_variants.json``. Touches no
    other data file and no reference-arena rows."""
    rows = measure_arena_tables()
    vdest = pathlib.Path("src/libero_infinity/data/spawn_clearances_variants.json")
    _merge_arena_table_rows(vdest, rows)


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


def _merge_distractor_table_rows(table_path: pathlib.Path, table_rows: dict[str, float]) -> dict:
    """Add measured table-distractor clearances for distractor-only pool classes
    that are MISSING from the canonical per-class table.

    Only adds classes absent from the canonical table — never overwrites a
    validated task-object measurement (those rows have many samples from the
    object axis; the distractor path is a single-orientation observation). The
    renderer's table-distractor slot resolves ``surface_class=None`` →
    ``SPAWN_CLEARANCES`` (this canonical table), so adding the missing classes
    (desk_caddy, bowl_drainer, cookies, popcorn, alphabet_soup, …) makes the
    table-distractor injected z match the settled z instead of the DEFAULT prior.
    Returns the dict of rows actually added.
    """
    canon = json.loads(table_path.read_text())
    existing = canon.get("clearances", {})
    added = {c: z for c, z in sorted(table_rows.items()) if c not in existing}
    if not added:
        print(
            f"\nNo new distractor-only table rows to merge into {table_path} "
            f"(all {len(table_rows)} measured classes already have canonical rows)."
        )
        return {}
    existing.update(added)
    canon["clearances"] = {k: existing[k] for k in sorted(existing)}
    canon.setdefault("_meta", {})["n_distractor_table_rows"] = len(added)
    table_path.write_text(json.dumps(canon, indent=2, sort_keys=False) + "\n")
    print(f"\nMerged {len(added)} distractor-only table clearance row(s) into {table_path}:")
    for c, z in added.items():
        print(f"  {c:28} {z:.4f}")
    return added


def _merge_fixture_rows(
    existing: dict[str, float],
    dist_rows: dict[str, float],
    *,
    tol: float = _POSE_TOLERANCE,
) -> dict[str, float]:
    """Merge measured (class|fixture) rows into ``existing`` IN PLACE, rewriting a
    stored row only when the new value diverges from it by more than ``tol``.

    A row is "wrong" exactly when injected (== stored) z differs from the measured
    settled z by more than the smoke's pose_tolerance. Within-tolerance rows are
    left BYTE-IDENTICAL, so the validated box / fixture rows (and any pair whose
    re-measurement only jittered by physics noise) are preserved by construction;
    only genuinely-divergent rows (the irregular desk_caddy / bowl_drainer cabinet
    rows) are corrected. New pairs absent from ``existing`` are always added.
    Returns the dict of rows actually written (added or corrected).
    """
    changed: dict[str, float] = {}
    for k, v in sorted(dist_rows.items()):
        old = existing.get(k)
        if old is None or abs(v - old) > tol:
            existing[k] = v
            changed[k] = v
    return changed


def _run_distractor_fixtures_only() -> None:
    """Measure ONLY the per-(distractor, fixture) clearances + fixture geometry
    and merge them into the existing data files, leaving the already-validated
    table-resting and object-axis variant rows untouched.

    Used to extend the landed (table) measurement with the Fix 2 on-fixture
    rows without re-running (and risking perturbing) the validated table/object
    measurement.
    """
    table_path = pathlib.Path("src/libero_infinity/data/spawn_clearances.json")
    table_clearances = json.loads(table_path.read_text()).get("clearances", {})

    dist_rows, fixture_geometry, table_rows = measure_distractor_fixtures(table_clearances)
    _merge_distractor_table_rows(table_path, table_rows)

    vdest = pathlib.Path("src/libero_infinity/data/spawn_clearances_variants.json")
    vdata = json.loads(vdest.read_text())
    changed = _merge_fixture_rows(vdata["clearances"], dist_rows)
    vdata["clearances"] = {k: vdata["clearances"][k] for k in sorted(vdata["clearances"])}
    vdata["_meta"]["n_distractor_fixture_rows"] = len(dist_rows)
    vdest.write_text(json.dumps(vdata, indent=2, sort_keys=False) + "\n")
    print(
        f"\nMerged {len(dist_rows)} measured (class|fixture) rows into {vdest}; "
        f"{len(changed)} rewritten (>{_POSE_TOLERANCE * 1000:.0f}mm divergence), "
        f"rest preserved byte-identical:"
    )
    for k, v in changed.items():
        print(f"  REWROTE {k:40} -> {v:.5f}")

    # Preserve the validated fixture geometry (deterministic, already validated;
    # the clearance ROWS are what the z-data fix corrects, not the geometry). Only
    # ADD fixtures entirely missing from the stored file.
    fgdest = pathlib.Path("src/libero_infinity/data/fixture_geometry.json")
    fg_existing = json.loads(fgdest.read_text())
    added_fix = {f: g for f, g in fixture_geometry.items() if f not in fg_existing["fixtures"]}
    fg_existing["fixtures"].update(added_fix)
    fgdest.write_text(json.dumps(fg_existing, indent=2, sort_keys=True) + "\n")
    print(
        f"Fixture geometry: {len(added_fix)} new fixture(s) added "
        f"({sorted(added_fix)}), {len(fg_existing['fixtures']) - len(added_fix)} preserved."
    )


def measure_distractor_table(
    table_clearances: dict[str, float], *, seeds: int = _DISTRACTOR_FIXTURE_SEEDS
) -> dict[str, list[float]]:
    """Measure per-class TABLE-resting distractor settled clearance distributions.

    Generates the SAME ``"distractor"`` scenes as :func:`measure_distractor_fixtures`
    but collects ONLY table-resting distractor samples — it never touches the
    on-fixture contact/AABB machinery, so it cannot perturb the validated
    per-(class|fixture) rows and (with the 0660e57 pool-fit rejection routing the
    oversized irregular classes desk_caddy/bowl_drainer to the table) never settles
    an irregular distractor onto an undersized fixture (the contact-arena overflow).

    Admission is GATE-FREE per RCA ``distractor_z_convergence.md`` /
    ``distractor_table_z_recover.md``: a sample is admitted iff it is in the
    physical clearance band ``[0, _FIXTURE_CLEARANCE_MAX]`` AND makes a real
    workspace-table contact (:func:`_settled_on_table_surface`). No AABB-bottom
    gate, no live-stepping of irregular distractors. Returns the raw per-class
    sample lists; the caller aggregates by dominant settle MODE (:func:`_dominant_mode`).

    The renderer injects table distractors at IDENTITY orientation with
    ``preserve_default_z=False`` → ``surface_spawn_z(.., surface_class=None)`` →
    this canonical table. So the distractor-path settle measured here is exactly
    the height the renderer must inject to make injected z == settled z for a
    distractor — which can differ from a class's natural-orientation object-axis
    clearance (the stale ``measure()`` value for classes that are also task
    objects, e.g. butter). Task objects resting at table level keep LIBERO's
    default z (``preserve_default_z=True``) and never read this table, so
    correcting a row here cannot move the TASK pose-tolerance metric.
    """
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    avail = set(discover_all_tasks())
    table_samples: dict[str, list[float]] = {}

    tasks = [t for t in MEASURE_TASKS if t in avail]
    for ti, task_rel in enumerate(tasks):
        print(f"# [PROGRESS] task {ti + 1}/{len(tasks)} ({seeds} seeds): {task_rel}", flush=True)
        bddl = str(resolve_task_path(task_rel))
        for seed in range(seeds):
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
                cls = getattr(o, "asset_class", "") or ""
                if not cls or (surface_class and fixture_inst):
                    continue  # on-fixture distractor → out of scope (table-only)
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
                clr = body_z - TABLE_Z
                if 0.0 <= clr <= _FIXTURE_CLEARANCE_MAX and _settled_on_table_surface(env, nm):
                    table_samples.setdefault(cls, []).append(round(clr, 5))
            env.close()
    return table_samples


def _merge_distractor_table_corrective(
    table_path: pathlib.Path,
    table_samples: dict[str, list[float]],
    *,
    tol: float = _POSE_TOLERANCE,
) -> dict[str, tuple]:
    """Merge measured table-distractor clearances, CORRECTING divergent rows.

    Unlike :func:`_merge_distractor_table_rows` (add-missing-only — which is why
    the stale butter/popcorn/cookies rows were never re-measured), this rewrites a
    row whenever the dominant-MODE measured clearance diverges from the stored
    value by more than ``tol`` (the smoke's 5 mm pose_tolerance), and ADDS classes
    absent from the table. Within-tolerance rows are left BYTE-IDENTICAL, so every
    validated row is preserved by construction; only genuinely-wrong rows move.

    Returns ``{class: (old_or_None, new, n_samples)}`` for rows actually written.
    """
    canon = json.loads(table_path.read_text())
    existing = canon.get("clearances", {})
    rows = {c: round(_dominant_mode(v), 5) for c, v in sorted(table_samples.items())}
    changed: dict[str, tuple] = {}
    for c, z in rows.items():
        old = existing.get(c)
        if old is None or abs(z - float(old)) > tol:
            existing[c] = z
            changed[c] = (old, z, len(table_samples[c]))
    if not changed:
        print(
            f"\nNo distractor-table rows diverged > {tol * 1000:.0f}mm — "
            f"all {len(rows)} measured classes within tolerance, file untouched."
        )
        return {}
    canon["clearances"] = {k: existing[k] for k in sorted(existing)}
    meta = canon.setdefault("_meta", {})
    # NOTE: ``n_distractor_table_rows`` counts the distractor-ONLY rows present in
    # the table (added by ``_merge_distractor_table_rows``); this corrective merge
    # only rewrites existing object-axis rows (butter/cream_cheese) and adds none,
    # so that count is left untouched.
    # Transparent provenance: record the distractor-path correction for each row
    # rewritten (old → new, sample count), so the stale object-axis n_samples for
    # a corrected class (e.g. butter) is not silently misread as fresh.
    meta["distractor_table_corrections"] = {
        c: {"old": (None if old is None else round(float(old), 5)), "new": z, "n": n}
        for c, (old, z, n) in sorted(changed.items())
    }
    table_path.write_text(json.dumps(canon, indent=2, sort_keys=False) + "\n")
    print(
        f"\nCorrected {len(changed)} table-distractor clearance row(s) in {table_path} "
        f"(>{tol * 1000:.0f}mm divergence; rest byte-identical):"
    )
    for c, (old, z, n) in sorted(changed.items()):
        delta = "NEW" if old is None else f"{(z - float(old)) * 1000:+.1f}mm"
        print(
            f"  {c:24} {('—' if old is None else f'{float(old):.5f}'):>9} -> {z:.5f}  (n={n}, Δ={delta})"
        )
    return changed


def _run_distractor_table_only(seeds: int = _DISTRACTOR_FIXTURE_SEEDS) -> None:
    """Measure ONLY the per-class TABLE distractor clearances and corrective-merge
    them into ``spawn_clearances.json``. Touches NO other data file (footprints,
    fixture geometry, on-fixture variant rows are all untouched)."""
    table_path = pathlib.Path("src/libero_infinity/data/spawn_clearances.json")
    existing = json.loads(table_path.read_text()).get("clearances", {})
    samples = measure_distractor_table(existing, seeds=seeds)
    print(
        f"\n# distractor-table samples per class (n): "
        f"{ {c: len(v) for c, v in sorted(samples.items())} }"
    )
    for c, v in sorted(samples.items()):
        mode = _dominant_mode(v)
        old = existing.get(c)
        chk = (
            f"stored={float(old):.5f} Δ={abs(mode - float(old)) * 1000:.1f}mm"
            if old is not None
            else "(NEW)"
        )
        print(f"  {c:24} mode={mode:.5f} med={statistics.median(v):.5f} n={len(v)}  {chk}")
    _merge_distractor_table_corrective(table_path, samples)


if __name__ == "__main__":
    import sys

    from libero_infinity.simulator import TABLE_Z

    if "--fixture-topz-worker" in sys.argv:
        # Subprocess worker: print exactly one ``TOPZ_RESULT <float|None>`` line
        # for (task, fixture_class), isolated so a MuJoCo crash stays contained.
        _i = sys.argv.index("--fixture-topz-worker")
        _task, _fclass = sys.argv[_i + 1], sys.argv[_i + 2]
        try:
            _tz = _fixture_rest_top_inscene(_task, _fclass)
        except Exception as _exc:  # noqa: BLE001 — worker failure reported as None
            print(f"# topz-worker error {_fclass} @ {_task}: {_exc}")
            _tz = None
        print(f"TOPZ_RESULT {_tz}")
        raise SystemExit(0)

    if "--settle-z-worker" in sys.argv:
        # Subprocess worker: print one ``SETTLEZ <fixture> <delta>`` line per valid
        # on-fixture distractor settle sample, isolated so an ncon-overflow segfault
        # on one (task, fixture) stays contained.
        _i = sys.argv.index("--settle-z-worker")
        _task, _fclass = sys.argv[_i + 1], sys.argv[_i + 2]
        try:
            _ds = _settle_z_inscene(_task, _fclass)
        except Exception as _exc:  # noqa: BLE001 — worker failure reported as no lines
            print(f"# settle-z-worker error {_fclass} @ {_task}: {_exc}")
            _ds = []
        for _d in _ds:
            print(f"SETTLEZ {_fclass} {_d}")
        raise SystemExit(0)

    if "--distractor-settle-z" in sys.argv:
        _run_distractor_settle_z()
        raise SystemExit(0)

    if "--support-fixtures-only" in sys.argv:
        _run_support_fixtures_only()
        raise SystemExit(0)

    if "--distractor-footprints-only" in sys.argv:
        _run_distractor_footprints_only()
        raise SystemExit(0)

    if "--distractor-fixtures-only" in sys.argv:
        _run_distractor_fixtures_only()
        raise SystemExit(0)

    if "--distractor-table-only" in sys.argv:
        _seeds = _DISTRACTOR_FIXTURE_SEEDS
        for _a in sys.argv:
            if _a.startswith("--seeds="):
                _seeds = int(_a.split("=", 1)[1])
        _run_distractor_table_only(seeds=_seeds)
        raise SystemExit(0)

    if "--arena-tables-only" in sys.argv:
        _run_arena_tables_only()
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
        # seating z exactly (renderer/simulator lockstep). Rows are the dominant
        # settle MODE over contacted, in-band samples; the injected==settled
        # invariant is verified by the smoke's 5 mm pose_tolerance.
        if "--no-distractor-fixtures" not in sys.argv:
            dist_rows, fixture_geometry, table_rows = measure_distractor_fixtures(out["clearances"])
            vout["clearances"].update(dist_rows)
            vout["clearances"] = {k: vout["clearances"][k] for k in sorted(vout["clearances"])}
            vout["_meta"]["n_distractor_fixture_rows"] = len(dist_rows)
            # Add table-distractor clearances for distractor-only pool classes
            # missing from the canonical table (dest already written above).
            _merge_distractor_table_rows(dest, table_rows)

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
