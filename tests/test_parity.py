"""Parity tests: ArticulationModel.canonical() vs legacy data sources.

These tests verify that the new ArticulationModel (single source of truth)
matches the legacy articulation data scattered across scene_semantics.py,
and task_semantics.py. If they fail, it means the
consolidation diverged from the legacy behaviour.
"""

from __future__ import annotations

import pytest

from libero_infinity.ir.nodes import ArticulationModel

# ---------------------------------------------------------------------------
# Fixture families parity
# ---------------------------------------------------------------------------


def test_parity_microwave_in_fixture_families() -> None:
    """microwave must be in ArticulationModel.fixture_families."""
    model = ArticulationModel.canonical()
    assert "microwave" in model.fixture_families, "microwave missing from fixture_families — diverged from legacy _ARTICULATABLE_FIXTURE_CLASSES"  # fmt: skip  # noqa: E501


def test_parity_wooden_cabinet_in_fixture_families() -> None:
    """wooden_cabinet must be in ArticulationModel.fixture_families."""
    model = ArticulationModel.canonical()
    assert "wooden_cabinet" in model.fixture_families


def test_parity_white_cabinet_in_fixture_families() -> None:
    """white_cabinet must be in ArticulationModel.fixture_families."""
    model = ArticulationModel.canonical()
    assert "white_cabinet" in model.fixture_families


def test_parity_flat_stove_in_fixture_families() -> None:
    """flat_stove must be in ArticulationModel.fixture_families."""
    model = ArticulationModel.canonical()
    assert "flat_stove" in model.fixture_families


def test_parity_exactly_four_articulatable_families() -> None:
    """There should be exactly 4 articulatable fixture families per spec."""
    model = ArticulationModel.canonical()
    assert len(model.fixture_families) == 4, (
        f"Expected exactly 4 fixture families, got {len(model.fixture_families)}: "
        f"{set(model.fixture_families.keys())}"
    )


# ---------------------------------------------------------------------------
# Articulation ranges parity: microwave
# ---------------------------------------------------------------------------


def test_parity_microwave_open_lo() -> None:
    """Microwave Open lo must be -2.094 (legacy scene_semantics.py ARTICULATION_RANGES)."""
    model = ArticulationModel.canonical()
    lo, _ = model.articulation_ranges["microwave"]["Open"]
    assert abs(lo - (-2.094)) < 1e-6, f"Microwave Open lo: expected -2.094, got {lo}"


def test_parity_microwave_open_hi() -> None:
    """Microwave Open hi must be -1.3 (legacy scene_semantics.py ARTICULATION_RANGES)."""
    model = ArticulationModel.canonical()
    _, hi = model.articulation_ranges["microwave"]["Open"]
    assert abs(hi - (-1.3)) < 1e-6, f"Microwave Open hi: expected -1.3, got {hi}"


def test_parity_microwave_close_lo() -> None:
    """Microwave Close lo must be -0.005 (legacy scene_semantics.py)."""
    model = ArticulationModel.canonical()
    lo, _ = model.articulation_ranges["microwave"]["Close"]
    assert abs(lo - (-0.005)) < 1e-6, f"Microwave Close lo: expected -0.005, got {lo}"


def test_parity_microwave_close_hi() -> None:
    """Microwave Close hi must be 0.0 (legacy scene_semantics.py)."""
    model = ArticulationModel.canonical()
    _, hi = model.articulation_ranges["microwave"]["Close"]
    assert abs(hi - 0.0) < 1e-6, f"Microwave Close hi: expected 0.0, got {hi}"


# ---------------------------------------------------------------------------
# Articulation ranges parity: wooden_cabinet
# ---------------------------------------------------------------------------


def test_parity_wooden_cabinet_open_lo() -> None:
    """Wooden cabinet Open lo must be -0.16 (legacy scene_semantics.py)."""
    model = ArticulationModel.canonical()
    lo, _ = model.articulation_ranges["wooden_cabinet"]["Open"]
    assert abs(lo - (-0.16)) < 1e-6, f"wooden_cabinet Open lo: expected -0.16, got {lo}"


def test_parity_wooden_cabinet_open_hi() -> None:
    """Wooden cabinet Open hi must be -0.14 (legacy scene_semantics.py)."""
    model = ArticulationModel.canonical()
    _, hi = model.articulation_ranges["wooden_cabinet"]["Open"]
    assert abs(hi - (-0.14)) < 1e-6, f"wooden_cabinet Open hi: expected -0.14, got {hi}"


# ---------------------------------------------------------------------------
# Articulation ranges parity: white_cabinet
# ---------------------------------------------------------------------------


def test_parity_white_cabinet_has_open_state() -> None:
    """white_cabinet must have an 'Open' state in articulation_ranges."""
    model = ArticulationModel.canonical()
    ranges = model.articulation_ranges.get("white_cabinet", {})
    assert "Open" in ranges, f"white_cabinet missing 'Open' state; has: {list(ranges.keys())}"


def test_parity_white_cabinet_open_range_valid() -> None:
    """white_cabinet Open range must have lo < hi."""
    model = ArticulationModel.canonical()
    lo, hi = model.articulation_ranges["white_cabinet"]["Open"]
    assert lo < hi, f"white_cabinet Open: lo={lo} >= hi={hi}"


# ---------------------------------------------------------------------------
# Articulation ranges parity: flat_stove
# ---------------------------------------------------------------------------


def test_parity_flat_stove_has_turnon_state() -> None:
    """flat_stove must have a 'Turnon' state in articulation_ranges."""
    model = ArticulationModel.canonical()
    ranges = model.articulation_ranges.get("flat_stove", {})
    assert "Turnon" in ranges, f"flat_stove missing 'Turnon' state; has: {list(ranges.keys())}"


def test_parity_flat_stove_has_turnoff_state() -> None:
    """flat_stove must have a 'Turnoff' state."""
    model = ArticulationModel.canonical()
    ranges = model.articulation_ranges.get("flat_stove", {})
    assert "Turnoff" in ranges, f"flat_stove missing 'Turnoff' state; has: {list(ranges.keys())}"


def test_parity_flat_stove_turnoff_range_valid() -> None:
    """flat_stove Turnoff must have lo < hi."""
    model = ArticulationModel.canonical()
    lo, hi = model.articulation_ranges["flat_stove"]["Turnoff"]
    assert lo < hi, f"flat_stove Turnoff: lo={lo} >= hi={hi}"


# ---------------------------------------------------------------------------
# Root workspace fixtures parity
# ---------------------------------------------------------------------------


def test_parity_table_in_root_workspace_fixtures() -> None:
    """'table' must be in root_workspace_fixtures (legacy: WORKSPACE_FIXTURES / root_fixtures)."""
    model = ArticulationModel.canonical()
    assert "table" in model.root_workspace_fixtures, "'table' missing from root_workspace_fixtures — diverged from legacy"  # fmt: skip  # noqa: E501


def test_parity_floor_in_root_workspace_fixtures() -> None:
    """'floor' must be in root_workspace_fixtures."""
    model = ArticulationModel.canonical()
    assert "floor" in model.root_workspace_fixtures


def test_parity_kitchen_table_in_root_workspace_fixtures() -> None:
    """'kitchen_table' must be in root_workspace_fixtures."""
    model = ArticulationModel.canonical()
    assert "kitchen_table" in model.root_workspace_fixtures


# ---------------------------------------------------------------------------
# is_articulatable parity
# ---------------------------------------------------------------------------


def test_parity_is_articulatable_microwave() -> None:
    model = ArticulationModel.canonical()
    assert model.is_articulatable("microwave")


def test_parity_is_articulatable_wooden_cabinet() -> None:
    model = ArticulationModel.canonical()
    assert model.is_articulatable("wooden_cabinet")


def test_parity_is_articulatable_white_cabinet() -> None:
    model = ArticulationModel.canonical()
    assert model.is_articulatable("white_cabinet")


def test_parity_is_articulatable_flat_stove() -> None:
    model = ArticulationModel.canonical()
    assert model.is_articulatable("flat_stove")


def test_parity_not_articulatable_table() -> None:
    model = ArticulationModel.canonical()
    assert not model.is_articulatable("table")


def test_parity_not_articulatable_unknown() -> None:
    model = ArticulationModel.canonical()
    assert not model.is_articulatable("banana_boat_fixture")


# ---------------------------------------------------------------------------
# get_family parity
# ---------------------------------------------------------------------------


def test_parity_get_family_microwave() -> None:
    """get_family('microwave') must return a non-None tuple."""
    model = ArticulationModel.canonical()
    family = model.get_family("microwave")
    assert family is not None
    family_name, kind = family
    assert family_name == "microwave"


def test_parity_get_family_unknown_returns_none() -> None:
    """get_family for unknown class must return None (no KeyError)."""
    model = ArticulationModel.canonical()
    result = model.get_family("does_not_exist")
    assert result is None


# ---------------------------------------------------------------------------
# Legacy compiler.py parity: generate_scenic backward compat
# ---------------------------------------------------------------------------


def test_parity_generate_scenic_exists() -> None:
    """generate_scenic() must exist in compiler.py for backward compat."""
    from libero_infinity.compiler import generate_scenic

    assert callable(generate_scenic)


def test_parity_compile_task_to_scenic_exists() -> None:
    """compile_task_to_scenic() must exist in compiler.py."""
    from libero_infinity.compiler import compile_task_to_scenic

    assert callable(compile_task_to_scenic)


def test_parity_compile_task_to_scenario_exists() -> None:
    """compile_task_to_scenario() must exist in compiler.py."""
    from libero_infinity.compiler import compile_task_to_scenario

    assert callable(compile_task_to_scenario)


def test_parity_compile_task_to_scenario_returns_scenario() -> None:
    """compile_task_to_scenario() must return a valid Scenic Scenario object.

    This test closes the critical gap: the compiler pipeline must produce a
    Scenario that Scenic can actually compile (no parse errors, no import
    errors). It uses a real BDDL file so the full pipeline is exercised.
    """
    pytest.importorskip("scenic", reason="scenic not installed")
    from conftest import LIVING_BASKET_BDDL
    from scenic.core.scenarios import Scenario  # type: ignore[import]

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.task_config import TaskConfig

    if LIVING_BASKET_BDDL is None or not LIVING_BASKET_BDDL.exists():
        pytest.skip("LIVING_BASKET_BDDL not found — vendored BDDL files missing")

    cfg = TaskConfig.from_bddl(LIVING_BASKET_BDDL)
    scenario = compile_task_to_scenario(cfg, "position")

    assert scenario is not None, "compile_task_to_scenario() returned None"
    assert isinstance(scenario, Scenario), f"Expected scenic.core.scenarios.Scenario, got {type(scenario)}"  # fmt: skip  # noqa: E501
