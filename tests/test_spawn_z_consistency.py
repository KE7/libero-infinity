"""G4 family-C pose-frame consistency (Option A).

These tests pin the invariant introduced by ``asset_metadata.surface_spawn_z``:
the Scenic renderer (codegen) and the MuJoCo simulator (reset) must resolve the
*same* spawn z for the same ``(surface_z, asset_class)``, so the Scenic-sampled
pose and the post-reset MuJoCo pose live in one frame and ``pose_tolerance`` can
compare them 1-to-1 (validation plan §4; see
``rca/stage1_g4_consistency_pose_frame_mismatch.md``).
"""

from __future__ import annotations

import re

import pytest

from libero_infinity import asset_metadata
from libero_infinity.asset_metadata import (
    SPAWN_CLEARANCES,
    TABLE_SURFACE_Z,
    surface_spawn_z,
)

MEASURED_CLASSES = sorted(SPAWN_CLEARANCES)
SURFACES = [0.82, 0.86, 0.90, 0.41]


# ---------------------------------------------------------------------------
# 1. Renderer-side and simulator-side z agree to <= 1 mm (the core invariant).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset_class", MEASURED_CLASSES)
@pytest.mark.parametrize("surface_z", SURFACES)
def test_renderer_and_simulator_resolve_identical_spawn_z(
    asset_class: str, surface_z: float
) -> None:
    """The renderer and simulator must compute byte-equal spawn z (<= 1 mm).

    The renderer emits ``surface_spawn_z(TABLE_SURFACE_Z, class)`` at codegen;
    the simulator resolves z via ``_surface_spawn_z(surface_z, class)``. Both
    delegate to the shared pure helper, so they agree by construction — this
    test guards against future divergence (a re-introduced bespoke formula on
    either side).
    """
    from libero_infinity.simulator import _surface_spawn_z

    renderer_z = surface_spawn_z(surface_z, asset_class)
    simulator_z = _surface_spawn_z(surface_z, asset_class)
    assert abs(renderer_z - simulator_z) <= 1e-3, (
        f"{asset_class}@{surface_z}: renderer {renderer_z} vs simulator "
        f"{simulator_z} differ by > 1 mm"
    )


def test_unmeasured_class_uses_median_prior_not_bounding_box() -> None:
    """Unmeasured classes must fall back to the median measured clearance.

    The pre-fix ``bbox_height / 2`` model is the one the z-frame RCA refuted; a
    short distractor/variant would otherwise be placed ~5–9 cm too low and fail
    pose_tolerance. The data-derived median prior keeps unmeasured classes in the
    correct ~0.10 m table band.
    """
    from libero_infinity.asset_metadata import DEFAULT_CLEARANCE, spawn_clearance

    assert not asset_metadata.is_measured("__nonexistent_class__")
    assert spawn_clearance("__nonexistent_class__") == pytest.approx(DEFAULT_CLEARANCE)
    # The prior is the median of the measured registry, in the table band.
    assert 0.08 <= DEFAULT_CLEARANCE <= 0.16


def test_surface_spawn_z_is_pure_and_surface_additive() -> None:
    """surface_spawn_z is a pure function: shifting the surface shifts z 1:1."""
    for cls in MEASURED_CLASSES:
        z0 = surface_spawn_z(0.82, cls)
        z1 = surface_spawn_z(0.92, cls)
        assert abs((z1 - z0) - 0.10) <= 1e-9, cls
        # Deterministic / repeatable.
        assert surface_spawn_z(0.82, cls) == z0


# ---------------------------------------------------------------------------
# 2. The renderer actually emits a concrete resolved z, not the placeholder.
# ---------------------------------------------------------------------------


def test_renderer_emits_concrete_resolved_z_for_table_objects() -> None:
    """A rendered kitchen task must place movables at the resolved spawn z.

    Before Option A the renderer emitted ``... , TABLE_Z)`` for every object
    (a placeholder the simulator silently overrode). It must now emit the
    concrete float ``surface_spawn_z(TABLE_SURFACE_Z, class)`` for absolutely
    placed objects so the Scenic pose matches the MuJoCo pose.
    """
    from libero_infinity.compiler import compile_task_to_scenic
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    bddl = str(resolve_task_path("libero_goal/put_the_bowl_on_the_stove.bddl"))
    src = compile_task_to_scenic(TaskConfig.from_bddl(bddl), "position")

    # The akita_black_bowl is placed absolutely on the table; its resolved z
    # must appear as a concrete literal in the bowl's placement vector.
    #
    # PR #24 (FV Finding A, per-(variant, surface) clearance): ``main_table`` is
    # a WorkspaceNode whose object class is ``"table"``, so the renderer resolves
    # the bowl's support-surface class to ``"table"`` and emits the measured
    # per-(variant, surface) settled z for ``("akita_black_bowl"|"table")`` — not
    # the legacy class-only value. Resolve the expected z through the SAME call
    # the renderer uses (surface_class="table"), so the test tracks the resolved
    # spawn z regardless of whether the variant table is populated (the renderer
    # and this assertion fall back identically when it is absent).
    bowl_z = surface_spawn_z(TABLE_SURFACE_Z, "akita_black_bowl", "table")
    z_literal = f"{bowl_z:.4f}"
    obj_lines = [
        ln for ln in src.splitlines() if "new LIBEROObject" in ln and "akita_black_bowl_1" in ln
    ]
    assert obj_lines, "expected an akita_black_bowl_1 LIBEROObject declaration"
    line = obj_lines[0]
    assert z_literal in line, f"bowl placement should carry concrete z {z_literal}; got: {line}"

    # No absolutely-placed LIBEROObject may still use the bare TABLE_Z token in
    # its position vector (relative `offset by ... 0.0` placements are exempt).
    for ln in src.splitlines():
        if "new LIBEROObject" in ln and "at Vector(" in ln:
            head = ln.split("with", 1)[0]
            assert "TABLE_Z" not in head, f"placeholder z leaked: {ln}"


# ---------------------------------------------------------------------------
# 3. The table-surface constant must not drift between the three definitions.
# ---------------------------------------------------------------------------


def test_table_surface_z_matches_simulator_and_scenic_model() -> None:
    """TABLE_SURFACE_Z must equal simulator.TABLE_Z and the Scenic model TABLE_Z."""
    import pathlib

    from libero_infinity.simulator import TABLE_Z as SIM_TABLE_Z

    assert TABLE_SURFACE_Z == SIM_TABLE_Z

    model_path = (
        pathlib.Path(asset_metadata.__file__).resolve().parents[2]
        / "scenic"
        / "libero_model.scenic"
    )
    text = model_path.read_text()
    m = re.search(r"^TABLE_Z\s*=\s*([0-9.]+)", text, re.MULTILINE)
    assert m is not None, "could not find TABLE_Z in libero_model.scenic"
    assert float(m.group(1)) == TABLE_SURFACE_Z


def test_smoke_task_classes_have_measured_clearances() -> None:
    """Every movable class used by the validation SMOKE_TASKS must be measured.

    Falling back to the bounding-box approximation for a smoke class would
    re-introduce the z-frame mismatch for that class, so the registry must
    cover them all.
    """
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import SMOKE_TASKS, resolve_task_path

    missing: set[str] = set()
    for task in SMOKE_TASKS:
        cfg = TaskConfig.from_bddl(str(resolve_task_path(task)))
        for mo in cfg.movable_objects:
            if getattr(mo, "contained", False):
                continue
            cls = getattr(mo, "object_class", None) or getattr(mo, "category", None)
            if cls and not asset_metadata.is_measured(cls):
                missing.add(cls)
    assert not missing, f"SMOKE_TASKS classes lack measured clearances: {missing}"


# ---------------------------------------------------------------------------
# 4. ``floor`` arena (libero_object suite) — Regime A of the g4 pose-drift RCA.
#    The kitchen per-class clearance does NOT transfer to the floor arena (the
#    suite authors several classes in a different rest orientation), so ``floor``
#    must be threaded as a per-arena table class with its OWN measured
#    ``<class>|floor`` clearances. See rca/g4_task_pose_drift.md (Regime A).
# ---------------------------------------------------------------------------


def test_floor_arena_is_threaded_as_per_arena_table_class() -> None:
    """The libero_object ``floor`` arena must be threaded so the renderer resolves
    the measured ``<class>|floor`` clearance instead of the (non-transferring)
    canonical kitchen clearance + arena shift."""
    from libero_infinity.asset_metadata import PER_ARENA_TABLE_CLASSES

    assert "floor" in PER_ARENA_TABLE_CLASSES


def test_floor_arena_task_object_classes_have_measured_floor_clearance() -> None:
    """Every task-object class of the 10 libero_object basket scenes (and the
    object-axis variants the renderer can emit) must have a measured
    ``<class>|floor`` clearance, so the object-axis subsets resolve the suite's
    settled z rather than the kitchen-frame fallback that fails pose_tolerance."""
    from libero_infinity.asset_metadata import VARIANT_CLEARANCES

    # Canonical task-object classes carried by the basket scenes.
    floor_classes = {
        "bbq_sauce",
        "basket",
        "chocolate_pudding",
        "ketchup",
        "salad_dressing",
        "alphabet_soup",
        "cream_cheese",
        "milk",
        "tomato_sauce",
        "butter",
        "orange_juice",
    }
    missing = {c for c in floor_classes if f"{c}|floor" not in VARIANT_CLEARANCES}
    assert not missing, f"floor arena classes lack measured |floor clearance: {missing}"


def test_floor_arena_resolved_z_in_suite_band() -> None:
    """``surface_spawn_z(arena_surface_z('floor'), class, 'floor')`` must land in
    the libero_object suite's settled band (objects rest just above the ~0 frame,
    a few cm to ~0.13 m). A kitchen-frame fallback would land ~0.9 m too high."""
    from libero_infinity.asset_metadata import (
        VARIANT_CLEARANCES,
        arena_surface_z,
        surface_spawn_z,
    )

    surf = arena_surface_z("floor")
    for key in VARIANT_CLEARANCES:
        if not key.endswith("|floor"):
            continue
        cls = key.split("|", 1)[0]
        z = surface_spawn_z(surf, cls, "floor")
        # Suite objects settle between ~-0.02 m (flat boxes) and ~0.15 m (tall
        # bottles) in the floor world frame; well clear of the ~0.9 m kitchen z.
        assert -0.05 <= z <= 0.20, f"{cls}|floor resolved z {z:.4f} outside suite band"
