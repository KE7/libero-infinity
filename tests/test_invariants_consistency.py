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
