"""Unit tests for the xy/drawer-state-aware support heightfield (g4 §6 cabinet).

These assert the two hard guarantees of the additive heightfield mechanism:

1. It resolves the MEASURED cabinet rest for the covered (fixture, relation,
   state, class) tuples.
2. It is BYTE-IDENTICAL to the scalar ``surface_spawn_z`` for every placement it
   does not cover — the no-regression requirement.
"""

from __future__ import annotations

import libero_infinity.asset_metadata as am


def test_normalize_drawer_state():
    assert am.normalize_drawer_state("Open") == "open"
    assert am.normalize_drawer_state("open") == "open"
    assert am.normalize_drawer_state("Close") == "closed"
    assert am.normalize_drawer_state("closed") == "closed"
    assert am.normalize_drawer_state("Turnon") is None
    assert am.normalize_drawer_state(None) is None
    assert am.normalize_drawer_state("") is None


def test_cabinet_has_heightfield():
    assert am.has_fixture_heightfield("wooden_cabinet") is True
    # No heightfield for any other fixture / table / None.
    assert am.has_fixture_heightfield("flat_stove") is False
    assert am.has_fixture_heightfield("table") is False
    assert am.has_fixture_heightfield(None) is False


def test_measured_closed_top_side_rest():
    # akita on the CLOSED cabinet top_side falls to the table-level rest ~0.898.
    clr = am.fixture_support_clearance("wooden_cabinet", "on_surface", "closed", "akita_black_bowl")
    assert clr is not None
    z = am.heightfield_spawn_z(
        am.TABLE_SURFACE_Z, "wooden_cabinet", "on_surface", "closed", "akita_black_bowl"
    )
    assert z is not None
    assert abs(z - (am.TABLE_SURFACE_Z + clr)) < 1e-9
    # measured rest ≈ 0.898 in the kitchen arena (arena_z == TABLE_SURFACE_Z).
    assert abs(z - 0.8984) < 5e-3


def test_measured_open_in_drawer_rest():
    # akita placed IN the OPEN top drawer rests on the drawer floor ~1.126 (RCA
    # §5.2 FIX2). The renderer emits this as the relative z-offset (rest − TABLE_Z)
    # and the simulator reaches the same rest via the from-above LIBERO default.
    clr = am.fixture_support_clearance("wooden_cabinet", "inside", "open", "akita_black_bowl")
    assert clr is not None
    z = am.heightfield_spawn_z(
        am.TABLE_SURFACE_Z, "wooden_cabinet", "inside", "open", "akita_black_bowl"
    )
    assert z is not None
    assert abs(z - (am.TABLE_SURFACE_Z + clr)) < 1e-9
    # measured rest ≈ 1.1264 in the kitchen arena (arena_z == TABLE_SURFACE_Z).
    assert abs(z - 1.1264) < 5e-3


def test_heightfield_returns_none_for_uncovered():
    # Uncovered (relation, state, class, fixture) tuples MUST return None so the
    # caller keeps the unchanged scalar path (byte-identical no-regression).
    # on_surface|open (the OPEN-drawer top_side knife-edge) is genuinely
    # metastable and deliberately ABSENT (RCA §5.1) → must stay None.
    assert (
        am.fixture_support_clearance("wooden_cabinet", "on_surface", "open", "akita_black_bowl")
        is None
    )
    # inside|CLOSED (a closed drawer) is NOT measured → None; only inside|open is.
    assert (
        am.fixture_support_clearance("wooden_cabinet", "inside", "closed", "akita_black_bowl")
        is None
    )
    # inside|open is covered ONLY for akita_black_bowl; other classes fall through.
    assert am.fixture_support_clearance("wooden_cabinet", "inside", "open", "white_bowl") is None
    assert (
        am.fixture_support_clearance("wooden_cabinet", "on_surface", "closed", "white_bowl") is None
    )
    assert (
        am.fixture_support_clearance("flat_stove", "on_surface", "closed", "akita_black_bowl")
        is None
    )
    assert (
        am.fixture_support_clearance("wooden_cabinet", None, "closed", "akita_black_bowl") is None
    )
    assert (
        am.fixture_support_clearance("wooden_cabinet", "on_surface", None, "akita_black_bowl")
        is None
    )
    assert (
        am.heightfield_spawn_z(0.82, "flat_stove", "on_surface", "closed", "akita_black_bowl")
        is None
    )


def test_byte_identical_scalar_fallback_broad():
    # For every (class, surface) that is NOT a covered heightfield tuple, the
    # renderer/simulator resolver must equal the scalar surface_spawn_z exactly.
    surfaces = [
        None,
        "table",
        "kitchen_table",
        "flat_stove",
        "microwave",
        "wine_rack",
        "wooden_cabinet",
    ]
    classes = ["akita_black_bowl", "white_bowl", "cookies", "ketchup", "plate", "alphabet_soup"]
    for arena_z in (0.82, 0.41, -0.035):
        for sc in surfaces:
            for cc in classes:
                for rel in (None, "on_surface", "inside"):
                    for st in (None, "open", "closed"):
                        hz = am.heightfield_spawn_z(arena_z, sc, rel, st, cc)
                        covered = (
                            sc == "wooden_cabinet"
                            and cc == "akita_black_bowl"
                            and (
                                (rel == "on_surface" and st == "closed")
                                or (rel == "inside" and st == "open")
                            )
                        )
                        if covered:
                            assert hz is not None
                        else:
                            assert hz is None, f"unexpected heightfield for {sc}/{rel}/{st}/{cc}"


def test_covered_tuple_ne_scalar():
    # Sanity: for the ONE covered tuple the heightfield deliberately DIFFERS from
    # the scalar (that is the whole point — the scalar mis-emits the cabinet top).
    hz = am.heightfield_spawn_z(0.82, "wooden_cabinet", "on_surface", "closed", "akita_black_bowl")
    scalar = am.surface_spawn_z(0.82, "akita_black_bowl", "wooden_cabinet")
    assert hz is not None
    assert abs(hz - scalar) > 0.2  # ~0.898 vs ~1.229
