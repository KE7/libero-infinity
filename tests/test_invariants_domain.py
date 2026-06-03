"""Tests for G4 Family B (domain) invariants.

Uses lightweight dataclass mocks so the suite does not require MuJoCo / Scenic
sampling. Each test exercises positive, negative, and (where applicable) skip
cases per the validation plan's discipline rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from libero_infinity.validation.invariants import (
    assert_assets_in_registry,
    assert_bddl_objects_present,
    assert_domain,
    assert_goal_false_at_reset,
    assert_goal_reachable_soft,
    assert_no_initial_collisions,
    assert_on_predicates_z,
)

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


@dataclass
class _Obj:
    name: str
    object_class: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    z_top: float | None = None
    aabb: tuple[float, float, float, float, float, float] | None = None
    is_fixed: bool = False


@dataclass
class _Scene:
    objects: list[_Obj] = field(default_factory=list)


@dataclass
class _BDDLObj:
    instance_name: str
    object_class: str


@dataclass
class _BDDL:
    movable_objects: list[_BDDLObj] = field(default_factory=list)
    init_text: str = ""
    goal_text: str = ""


@dataclass
class _Contact:
    dist: float
    geom1: int = 0
    geom2: int = 1


@dataclass
class _MjData:
    contact: list[_Contact] = field(default_factory=list)

    @property
    def ncon(self) -> int:
        return len(self.contact)


class _Env:
    def __init__(self, success: bool):
        self._success = success

    def check_success(self) -> bool:
        return self._success


# ---------------------------------------------------------------------------
# B1 — bddl_objects_present
# ---------------------------------------------------------------------------


def test_bddl_objects_present_pass():
    bddl = _BDDL(movable_objects=[_BDDLObj("bowl_1", "akita_black_bowl")])
    scene = _Scene(objects=[_Obj("bowl_1", "akita_black_bowl")])
    r = assert_bddl_objects_present(bddl, scene)
    assert r.passed is True


def test_bddl_objects_present_fail():
    bddl = _BDDL(
        movable_objects=[_BDDLObj("bowl_1", "akita_black_bowl"), _BDDLObj("plate_1", "plate")]
    )
    scene = _Scene(objects=[_Obj("bowl_1", "akita_black_bowl")])
    r = assert_bddl_objects_present(bddl, scene)
    assert r.passed is False
    assert "plate_1" in r.payload["missing"]


def test_bddl_objects_present_skip_when_empty():
    r = assert_bddl_objects_present(_BDDL(), _Scene())
    assert r.passed is None


# ---------------------------------------------------------------------------
# B2 — assets_in_registry
# ---------------------------------------------------------------------------


def test_assets_in_registry_pass():
    scene = _Scene(objects=[_Obj("bowl_1", "akita_black_bowl"), _Obj("plate_1", "plate")])
    r = assert_assets_in_registry(scene, registry={"akita_black_bowl", "plate"})
    assert r.passed is True


def test_assets_in_registry_fail():
    scene = _Scene(objects=[_Obj("ufo_1", "alien_ufo")])
    r = assert_assets_in_registry(scene, registry={"plate"})
    assert r.passed is False
    assert r.payload["unknown"] == [("ufo_1", "alien_ufo")]


def test_assets_in_registry_skip_when_no_objects():
    r = assert_assets_in_registry(_Scene(), registry={"plate"})
    assert r.passed is None


def test_assets_in_registry_default_uses_libero_registry():
    # Smoke: a real LIBERO class should be accepted by the default registry.
    scene = _Scene(objects=[_Obj("bowl_1", "akita_black_bowl")])
    r = assert_assets_in_registry(scene)
    assert r.passed is True


# ---------------------------------------------------------------------------
# B3 — no_initial_collisions
# ---------------------------------------------------------------------------


def test_no_collisions_pass():
    data = _MjData(contact=[_Contact(dist=1e-5), _Contact(dist=0.0)])
    r = assert_no_initial_collisions(_Scene(), object(), data, tol=1e-4)
    assert r.passed is True


def test_no_collisions_fail():
    data = _MjData(contact=[_Contact(dist=-1e-3)])
    r = assert_no_initial_collisions(_Scene(), object(), data, tol=1e-4)
    assert r.passed is False
    assert r.payload["penetrating"][0]["dist"] == pytest.approx(-1e-3)


def test_no_collisions_skip_no_mujoco():
    r = assert_no_initial_collisions(_Scene(), None, None)
    assert r.passed is None


# ---------------------------------------------------------------------------
# B4 — on_predicates_z
# ---------------------------------------------------------------------------


def test_on_predicates_z_pass():
    bddl = _BDDL(init_text="(On bowl_1 plate_1)")
    scene = _Scene(
        objects=[
            _Obj("plate_1", "plate", position=(0, 0, 0.0), z_top=0.02),
            _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0.03)),
        ]
    )
    r = assert_on_predicates_z(bddl, scene)
    assert r.passed is True


def test_on_predicates_z_fail():
    bddl = _BDDL(init_text="(On bowl_1 plate_1)")
    scene = _Scene(
        objects=[
            _Obj("plate_1", "plate", position=(0, 0, 0.5), z_top=0.52),
            _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0.05)),
        ]
    )
    r = assert_on_predicates_z(bddl, scene)
    assert r.passed is False
    assert r.payload["violations"][0]["a"] == "bowl_1"


@dataclass
class _Fixture:
    """Mock fixture whose class name ends in ``Fixture`` so
    ``is_scene_fixture`` resolves it as scene structure (not a sampled asset).
    """

    name: str
    object_class: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    z_top: float | None = None
    aabb: tuple[float, float, float, float, float, float] | None = None
    is_fixed: bool = True


# Class name must end in "Fixture" for `is_scene_fixture` to detect it.
_Fixture.__name__ = "LIBEROFixture"


def test_on_predicates_z_resolves_region_via_fixture_prefix_pass():
    """`(On bowl main_table_bowl_region)` resolves to the `main_table` fixture.

    Regression test for RCA Finding 4 — On-predicate targets reference named
    regions/sides on a fixture (e.g. `main_table_bowl_region`,
    `wooden_cabinet_1_top_side`) that are not themselves materialised as
    Scenic objects. The check must resolve the longest fixture-name prefix.
    """
    bddl = _BDDL(init_text="(On bowl_1 main_table_bowl_region)")
    scene = _Scene(
        objects=[
            _Fixture("main_table", "main_table", position=(0, 0, 0.0), z_top=0.02),
            _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0.05)),
        ]
    )
    r = assert_on_predicates_z(bddl, scene)
    assert r.passed is True, r.detail


def test_on_predicates_z_resolves_region_via_fixture_prefix_fail():
    """Same prefix resolution path, but the bowl is below the table top."""
    bddl = _BDDL(init_text="(On bowl_1 main_table_bowl_region)")
    scene = _Scene(
        objects=[
            _Fixture("main_table", "main_table", position=(0, 0, 0.0), z_top=0.50),
            _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0.05)),
        ]
    )
    r = assert_on_predicates_z(bddl, scene)
    assert r.passed is False
    assert r.payload["violations"][0]["a"] == "bowl_1"


def test_on_predicates_z_resolves_longest_fixture_prefix():
    """When two fixtures share a prefix, the longest match wins.

    `wooden_cabinet_1_top_side` must resolve to `wooden_cabinet_1`, not
    `wooden_cabinet`.
    """
    bddl = _BDDL(init_text="(On bowl_1 wooden_cabinet_1_top_side)")
    scene = _Scene(
        objects=[
            _Fixture("wooden_cabinet", "cabinet", position=(0, 0, 0), z_top=1.0),
            _Fixture("wooden_cabinet_1", "cabinet", position=(0, 0, 0), z_top=0.02),
            _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0.05)),
        ]
    )
    r = assert_on_predicates_z(bddl, scene)
    # If the longer name resolved, z_top=0.02 → bowl at 0.05 passes.
    assert r.passed is True, r.detail


@dataclass
class _ObjWithParent:
    """Mock movable carrying ``support_parent_name`` (the implicit-support
    annotation Scenic LIBEROObjects emit for table-supported objects)."""

    name: str
    object_class: str
    position: tuple[float, float, float]
    support_parent_name: str = "main_table"
    z_top: float | None = None
    aabb: tuple[float, float, float, float, float, float] | None = None
    is_fixed: bool = False


def test_on_predicates_z_virtual_main_table_support():
    """`(On bowl_1 main_table_bowl_region)` resolves to a *virtual* support
    derived from any movable's `support_parent_name == "main_table"` — the
    table is rendered by LIBERO, not by the Scenic compiler, so it never
    appears in `scene.objects` as a fixture. Regression for the second half
    of RCA Finding 4 surfaced by the post-PR-21 smoke.
    """
    bddl = _BDDL(init_text="(On bowl_1 main_table_bowl_region)")
    scene = _Scene(
        objects=[
            _ObjWithParent("bowl_1", "akita_black_bowl", position=(0, 0, 0.82)),
        ]
    )
    r = assert_on_predicates_z(bddl, scene)
    assert r.passed is True, r.detail


def test_on_predicates_z_skip_no_predicates():
    r = assert_on_predicates_z(_BDDL(init_text="(AtPose bowl_1 region_1)"), _Scene())
    assert r.passed is None


# ---------------------------------------------------------------------------
# B5 — goal_false_at_reset
# ---------------------------------------------------------------------------


def test_goal_false_at_reset_pass():
    r = assert_goal_false_at_reset(_BDDL(goal_text="(On bowl_1 plate_1)"), _Env(success=False))
    assert r.passed is True


def test_goal_false_at_reset_fail_trivial():
    r = assert_goal_false_at_reset(_BDDL(goal_text="(On bowl_1 plate_1)"), _Env(success=True))
    assert r.passed is False


def test_goal_false_at_reset_skip_no_evaluator():
    r = assert_goal_false_at_reset(_BDDL(), object())
    assert r.passed is None


def test_goal_false_at_reset_uses_supplied_evaluator():
    calls = []

    def evaluator(bddl, env):
        calls.append((bddl, env))
        return False

    r = assert_goal_false_at_reset(_BDDL(), object(), goal_evaluator=evaluator)
    assert r.passed is True
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# B6 — goal_reachable_soft
# ---------------------------------------------------------------------------


def test_goal_reachable_soft_pass():
    bddl = _BDDL(goal_text="(On bowl_1 plate_1)")
    scene = _Scene(
        objects=[
            _Obj("plate_1", "plate", aabb=(-0.1, 0.1, -0.1, 0.1, 0.0, 0.02)),
            _Obj("bowl_1", "akita_black_bowl", aabb=(-0.05, 0.05, -0.05, 0.05, 0.0, 0.03)),
        ]
    )
    r = assert_goal_reachable_soft(bddl, scene)
    assert r.passed is True
    assert "bowl_1" in r.payload["resolved"]


def test_goal_reachable_soft_fail_occluded():
    bddl = _BDDL(goal_text="(On bowl_1 plate_1)")
    scene = _Scene(
        objects=[
            _Obj(
                "lid_1",
                "lid",
                aabb=(-1.0, 1.0, -1.0, 1.0, 0.5, 0.6),
                is_fixed=True,
            ),
            _Obj(
                "bowl_1",
                "akita_black_bowl",
                position=(0, 0, 0.0),
                z_top=0.1,
                aabb=(-0.05, 0.05, -0.05, 0.05, 0.0, 0.1),
            ),
        ]
    )
    r = assert_goal_reachable_soft(bddl, scene)
    assert r.passed is False
    assert r.payload["occluded"][0]["object"] == "bowl_1"


def test_goal_reachable_soft_skip_no_goal():
    r = assert_goal_reachable_soft(_BDDL(goal_text=""), _Scene())
    assert r.passed is None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_assert_domain_runs_all_seven():
    bddl = _BDDL(
        movable_objects=[_BDDLObj("bowl_1", "akita_black_bowl")],
        init_text="(On bowl_1 plate_1)",
        goal_text="(On bowl_1 plate_1)",
    )
    scene = _Scene(
        objects=[
            _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0.05), z_top=0.1),
            _Obj("plate_1", "plate", position=(0, 0, 0.0), z_top=0.02),
        ]
    )
    results = assert_domain(bddl, scene, registry={"akita_black_bowl", "plate"}, env=_Env(False))
    assert len(results) == 7
    names = [r.name for r in results]
    assert names == [
        "bddl_objects_present",
        "assets_in_registry",
        "no_initial_collisions",  # skip (no mjdata)
        "on_predicates_z",
        "goal_false_at_reset",
        "goal_reachable_soft",
        "goal_region_admits_object",  # skip (fake BDDL → no resolvable region)
    ]
    # Mujoco skipped (None); goal_region_admits_object skips (None) here because
    # the fake _BDDL is not a TaskConfig, so no goal region resolves.
    statuses = {r.name: r.passed for r in results}
    assert statuses["no_initial_collisions"] is None
    assert statuses["goal_region_admits_object"] is None
    skipped = {"no_initial_collisions", "goal_region_admits_object"}
    assert all(statuses[k] is True for k in statuses if k not in skipped)
