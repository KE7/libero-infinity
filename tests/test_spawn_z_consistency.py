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
    bowl_z = surface_spawn_z(TABLE_SURFACE_Z, "akita_black_bowl")
    z_literal = f"{bowl_z:.4f}"
    obj_lines = [
        ln
        for ln in src.splitlines()
        if "new LIBEROObject" in ln and "akita_black_bowl_1" in ln
    ]
    assert obj_lines, "expected an akita_black_bowl_1 LIBEROObject declaration"
    line = obj_lines[0]
    assert z_literal in line, (
        f"bowl placement should carry concrete z {z_literal}; got: {line}"
    )

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
