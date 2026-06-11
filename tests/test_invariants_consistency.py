"""Tests for G4 Family C (scene↔env consistency) invariants."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from libero_infinity.validation.invariants import (
    assert_class_match,
    assert_consistency,
    assert_pose_tolerance,
)


@dataclass
class _Obj:
    name: str
    object_class: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] | None = None


@dataclass
class _Scene:
    objects: list[_Obj] = field(default_factory=list)


class _Env:
    """Duck-typed env that exposes ``get_object_state(name) -> dict``."""

    def __init__(self, states: dict[str, dict]):
        self._states = states

    def get_object_state(self, name):
        return self._states.get(name)


def _quat_about_z(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# ---------------------------------------------------------------------------
# pose_tolerance
# ---------------------------------------------------------------------------


def test_pose_tolerance_pass_within_tols():
    o = _Obj("bowl_1", "bowl", position=(0.1, 0.2, 0.3), orientation=_quat_about_z(0.0))
    state = {"position": (0.1003, 0.2, 0.3), "orientation": _quat_about_z(0.5), "class": "bowl"}
    r = assert_pose_tolerance(o, state, pos_tol=5e-3, rot_tol_deg=1.0)
    assert r.passed is True


def test_pose_tolerance_fail_position():
    o = _Obj("bowl_1", "bowl", position=(0.0, 0.0, 0.0), orientation=_quat_about_z(0.0))
    state = {"position": (0.5, 0.0, 0.0), "orientation": _quat_about_z(0.0), "class": "bowl"}
    r = assert_pose_tolerance(o, state)
    assert r.passed is False
    assert r.payload["position_error"] == pytest.approx(0.5)


def test_pose_tolerance_fail_rotation():
    o = _Obj("bowl_1", "bowl", position=(0.0, 0.0, 0.0), orientation=_quat_about_z(0.0))
    state = {"position": (0.0, 0.0, 0.0), "orientation": _quat_about_z(45.0), "class": "bowl"}
    r = assert_pose_tolerance(o, state, rot_tol_deg=1.0)
    assert r.passed is False
    assert r.payload["rotation_error_deg"] == pytest.approx(45.0, abs=1e-3)


def test_pose_tolerance_no_rotation_data_passes_if_position_ok():
    # Skip-on-missing-rotation is part of the field-level diagnostic, NOT a
    # silent pass: rotation_error_deg=None means "no env rotation supplied";
    # we treat that as not-applicable for the rotation half.
    o = _Obj("bowl_1", "bowl", position=(0, 0, 0), orientation=None)
    state = {"position": (0, 0, 0), "orientation": None, "class": "bowl"}
    r = assert_pose_tolerance(o, state)
    assert r.passed is True
    assert r.payload["rotation_error_deg"] is None


# ---------------------------------------------------------------------------
# class_match
# ---------------------------------------------------------------------------


def test_class_match_pass():
    o = _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0))
    r = assert_class_match(o, {"class": "akita_black_bowl"})
    assert r.passed is True


def test_class_match_fail():
    o = _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0))
    r = assert_class_match(o, {"class": "plate"})
    assert r.passed is False
    assert r.payload["scenic_class"] == "akita_black_bowl"
    assert r.payload["env_class"] == "plate"


def test_class_match_fail_missing_env():
    o = _Obj("bowl_1", "akita_black_bowl", position=(0, 0, 0))
    r = assert_class_match(o, {"class": None})
    assert r.passed is False


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_assert_consistency_pass_for_each_object():
    scene = _Scene(
        objects=[
            _Obj("bowl_1", "bowl", position=(0, 0, 0), orientation=_quat_about_z(0)),
            _Obj("plate_1", "plate", position=(0.1, 0, 0), orientation=_quat_about_z(0)),
        ]
    )
    env = _Env(
        {
            "bowl_1": {"position": (0, 0, 0), "orientation": _quat_about_z(0), "class": "bowl"},
            "plate_1": {"position": (0.1, 0, 0), "orientation": _quat_about_z(0), "class": "plate"},
        }
    )
    results = assert_consistency(scene, env)
    assert len(results) == 4
    assert all(r.passed is True for r in results)


def test_assert_consistency_env_missing_is_failure_not_skip():
    scene = _Scene(objects=[_Obj("ghost", "bowl", position=(0, 0, 0))])
    env = _Env({})  # ghost not in env
    results = assert_consistency(scene, env)
    assert len(results) == 2
    assert all(r.passed is False for r in results)
    assert results[0].payload["reason"] == "env_missing"


def test_assert_consistency_detects_class_mismatch():
    scene = _Scene(
        objects=[_Obj("bowl_1", "bowl", position=(0, 0, 0), orientation=_quat_about_z(0))]
    )
    env = _Env(
        {"bowl_1": {"position": (0, 0, 0), "orientation": _quat_about_z(0), "class": "plate"}}
    )
    results = assert_consistency(scene, env)
    pose, cls = results
    assert pose.passed is True
    assert cls.passed is False


# ---------------------------------------------------------------------------
# LIBEROScenicEnv.get_object_state — RCA Finding 3 env-side accessor
# ---------------------------------------------------------------------------


def test_libero_scenic_env_get_object_state_returns_pose_and_class():
    """LIBEROScenicEnv must expose `get_object_state(name)` so the G4
    family-C consistency check can compare Scenic vs MuJoCo poses. Before
    this fix the consistency hook duck-typed three accessors, none of which
    LIBEROScenicEnv implemented — every check was uniform-False.
    """
    from types import SimpleNamespace

    from libero_infinity.gym_env import LIBEROScenicEnv

    # Bypass __init__: we exercise the accessor surface only, with a fake
    # sim/libero_env wired to the MuJoCo arrays the method reads.
    env = LIBEROScenicEnv.__new__(LIBEROScenicEnv)
    fake_data = SimpleNamespace(
        body_xpos={7: [1.0, 2.0, 3.0]},
        body_xquat={7: [1.0, 0.0, 0.0, 0.0]},
    )
    fake_libero_env = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=fake_data)))
    env._sim = SimpleNamespace(
        _body_ids={"akita_black_bowl_1": 7, "missing_bowl": None},
        libero_env=fake_libero_env,
    )
    env._effective_obj_classes = {"akita_black_bowl_1": "akita_black_bowl"}

    st = env.get_object_state("akita_black_bowl_1")
    assert st is not None
    assert st["position"] == (1.0, 2.0, 3.0)
    assert st["orientation"] == (1.0, 0.0, 0.0, 0.0)
    assert st["class"] == "akita_black_bowl"

    # Unknown / unresolved body returns None (caller surfaces as failure).
    assert env.get_object_state("missing_bowl") is None
    assert env.get_object_state("not_in_map") is None


def test_assert_consistency_uses_libero_env_accessor_real_results():
    """End-to-end check: with a LIBEROScenicEnv-shaped env exposing
    `get_object_state`, `assert_consistency` returns *real* per-object pose
    and class results (not uniform False with reason "env has no accessor").
    """
    from types import SimpleNamespace

    from libero_infinity.gym_env import LIBEROScenicEnv

    env = LIBEROScenicEnv.__new__(LIBEROScenicEnv)
    fake_data = SimpleNamespace(
        body_xpos={1: [0.10, 0.20, 0.30]},
        body_xquat={1: [1.0, 0.0, 0.0, 0.0]},
    )
    env._sim = SimpleNamespace(
        _body_ids={"bowl_1": 1},
        libero_env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=fake_data))),
    )
    env._effective_obj_classes = {"bowl_1": "akita_black_bowl"}

    scene = _Scene(
        objects=[
            _Obj(
                "bowl_1",
                "akita_black_bowl",
                position=(0.10, 0.20, 0.30),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )
        ]
    )
    # The duck-typed Scenic object exposes `name` (not `libero_name`) — that is
    # fine because `resolve_object_name` falls back to `name` for test doubles.
    results = assert_consistency(scene, env)
    assert len(results) == 2
    pose, cls = results
    assert pose.name == "pose_tolerance"
    assert pose.passed is True, pose.detail
    assert cls.name == "class_match"
    assert cls.passed is True, cls.detail
    # Critical: not the uniform-False env_missing path.
    assert pose.payload.get("reason") != "env_missing"


# ---------------------------------------------------------------------------
# WS-1: measured fixture geometry — no silent hand-coded fallback for the real
# support fixtures objects/distractors rest on across the scenario corpus.
# ---------------------------------------------------------------------------


def _corpus_support_fixture_classes() -> set[str]:
    """Every non-table fixture class present as a ``FixtureNode`` in the corpus.

    These are the surfaces the renderer can seat objects/distractors on; the
    workspace tables/floor are arena surfaces (served by the arena-table rows,
    not ``fixture_geometry.json``) and are excluded.
    """
    from libero_infinity.ir.graph_builder import build_semantic_scene_graph
    from libero_infinity.ir.nodes import ArticulationModel, FixtureNode
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path

    tables = set(ArticulationModel.canonical().root_workspace_fixtures)
    classes: set[str] = set()
    for task_rel in discover_all_tasks():
        try:
            cfg = TaskConfig.from_bddl(str(resolve_task_path(task_rel)))
            graph = build_semantic_scene_graph(cfg)
        except Exception:  # noqa: BLE001 — unbuildable task is not this test's concern
            continue
        for node in graph.nodes.values():
            cls = getattr(node, "object_class", None)
            if isinstance(node, FixtureNode) and cls and cls not in tables:
                classes.add(cls)
    return classes


def test_all_corpus_support_fixtures_have_measured_geometry():
    """Every real (non-table) support fixture used anywhere in the scenario
    corpus MUST carry MEASURED geometry in ``fixture_geometry.json`` — never the
    hand-coded ``_FIXTURE_DIMS_FALLBACK``. A fixture missing here silently
    degrades its spawn z / footprint to a guessed value (audit WS-1)."""
    from libero_infinity import asset_metadata

    support_fixtures = _corpus_support_fixture_classes()
    assert support_fixtures, "corpus scan found no support fixtures — scan is broken"
    unmeasured = sorted(f for f in support_fixtures if not asset_metadata.is_fixture_measured(f))
    assert not unmeasured, (
        f"corpus support fixtures missing MEASURED geometry (fell back to "
        f"_FIXTURE_DIMS_FALLBACK): {unmeasured}. Measure them with "
        f"`scripts/measure_spawn_clearances.py --support-fixtures-only`."
    )


def test_measured_fixture_geometry_rows_are_well_formed():
    """Each measured fixture row exposes a sane footprint, height and top_z."""
    from libero_infinity import asset_metadata

    assert asset_metadata.FIXTURE_GEOMETRY, "no measured fixtures loaded"
    for fclass, geom in asset_metadata.FIXTURE_GEOMETRY.items():
        fw, fl = asset_metadata.fixture_footprint(fclass)
        assert 0.0 < fw < 1.0 and 0.0 < fl < 1.0, (fclass, fw, fl)
        assert 0.0 < asset_metadata.fixture_height(fclass) < 1.0, fclass
        assert 0.0 <= asset_metadata.fixture_top_z_above_table(fclass) < 1.0, fclass


def test_fixture_fallback_warns_only_for_unmeasured_real_fixtures(caplog):
    """A real fixture without measured geometry WARNS (once); a measured fixture
    and a workspace table (an arena surface) do NOT warn."""
    import logging

    from libero_infinity import asset_metadata

    # Measured fixture: no warning.
    asset_metadata._warned_fixture_fallback.discard("flat_stove")
    with caplog.at_level(logging.WARNING, logger="libero_infinity.asset_metadata"):
        asset_metadata.fixture_footprint("flat_stove")
    assert not any("fixture_geometry" in r.message for r in caplog.records)

    # Workspace table (arena surface, fallback is the accepted source): no warning.
    caplog.clear()
    asset_metadata._warned_fixture_fallback.discard("kitchen_table")
    with caplog.at_level(logging.WARNING, logger="libero_infinity.asset_metadata"):
        asset_metadata.fixture_footprint("kitchen_table")
    assert not any("no MEASURED" in r.message for r in caplog.records)

    # Genuinely-unmeasured real fixture: warns exactly once, then deduplicates.
    caplog.clear()
    fake = "totally_unmeasured_fixture_xyz"
    asset_metadata._warned_fixture_fallback.discard(fake)
    with caplog.at_level(logging.WARNING, logger="libero_infinity.asset_metadata"):
        asset_metadata.fixture_footprint(fake)
        asset_metadata.fixture_height(fake)
    warns = [r for r in caplog.records if "no MEASURED" in r.message and fake in r.message]
    assert len(warns) == 1, f"expected one deduped warning, got {len(warns)}"
