# BDDL filenames pinned at module scope are unavoidably long; relax E501.
# ruff: noqa: E501
"""Real-graph regression tests for the container-affordance variant filter.

These tests build a ``SemanticSceneGraph`` from an actual LIBERO BDDL via
the production ``build_semantic_scene_graph`` pipeline (as opposed to the
synthetic hand-constructed graphs in
``tests/test_object_language_fixes.py``). They lock in the corrective fix
for PR #7's inert filter; see
``rca/integration_smoke_pr7_filter_inert.md``.

The PR #7 filter probed for ``contained_in`` graph edges, but the real
graph builder emits ``goal_target`` edges whose ``dst_id`` is the full
``<node_id>_<site>`` region name. The repaired filter must therefore
inspect ``goal_target`` edges directly. These tests will FAIL on the
parent commit (inert filter) and PASS on the repair.
"""

from __future__ import annotations

import pathlib

import pytest

from libero_infinity.asset_registry import contain_region_sites
from libero_infinity.ir.graph_builder import build_semantic_scene_graph
from libero_infinity.planner.composition import plan_perturbations
from libero_infinity.task_config import TaskConfig

_BDDL_ROOT = pathlib.Path(
    "/home/batman/Documents/research/libero-infinity/libero-pro/libero/libero/bddl_files"
)
_BASKET = _BDDL_ROOT / "libero_object" / "pick_up_the_ketchup_and_place_it_in_the_basket.bddl"
_TRAY = (
    _BDDL_ROOT
    / "libero_90"
    / "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray.bddl"
)
_CADDY = (
    _BDDL_ROOT
    / "libero_90"
    / "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl"
)


def _plan_object_substitutions(bddl: pathlib.Path) -> dict[str, list[str]]:
    if not bddl.is_file():
        pytest.skip(f"BDDL fixture missing: {bddl}")
    cfg = TaskConfig.from_bddl(str(bddl))
    graph = build_semantic_scene_graph(cfg)
    plan = plan_perturbations(graph, frozenset(["object"]))
    return plan.object_substitutions


def test_basket_real_graph_filters_non_container_variants() -> None:
    subs = _plan_object_substitutions(_BASKET)
    assert "basket_1" in subs, f"basket_1 must have a substitution pool: {subs!r}"
    pool = subs["basket_1"]
    # white_storage_box and chefmate_8_frypan lack contain_region sites: must be filtered.
    assert "white_storage_box" not in pool, pool
    assert "chefmate_8_frypan" not in pool, pool
    _required = contain_region_sites("basket")
    for v in pool:
        _msg = f"variant {v!r} missing required contain_region sites"
        assert contain_region_sites(v) >= _required, _msg


def test_tray_real_graph_filters_non_container_variants() -> None:
    subs = _plan_object_substitutions(_TRAY)
    assert "wooden_tray_1" in subs, f"wooden_tray_1 must have a substitution pool: {subs!r}"
    pool = subs["wooden_tray_1"]
    assert "white_storage_box" not in pool, pool
    assert "chefmate_8_frypan" not in pool, pool
    _required = contain_region_sites("wooden_tray")
    for v in pool:
        _msg = f"variant {v!r} missing required contain_region sites"
        assert contain_region_sites(v) >= _required, _msg


def test_caddy_real_graph_preserves_directional_contain_regions() -> None:
    """Caddy already passed pre-repair (variant pool is effectively singleton);
    lock it in so future variant-pool expansions cannot regress."""
    subs = _plan_object_substitutions(_CADDY)
    if "desk_caddy_1" not in subs:
        # Pool may have been pinned to canonical singleton: that is acceptable.
        return
    required = contain_region_sites("desk_caddy")
    assert required, "desk_caddy must declare directional contain_region sites"
    for v in subs["desk_caddy_1"]:
        assert contain_region_sites(v) >= required, (
            f"variant {v!r} missing one of the directional contain_region sites "
            f"{sorted(required)!r}"
        )
