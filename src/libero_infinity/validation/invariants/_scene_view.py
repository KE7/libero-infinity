"""Scene adapter — projects a Scenic ``Scene`` onto the richer per-axis schema
expected by the G4 family-A identity invariants.

Why this exists
---------------
The G4 identity invariants (``identity.py``) duck-type a scene with the
attributes ``objects``, ``fixtures``, ``distractors``, ``lights``, ``cameras``,
``robot``, ``background``. The renderer (``renderer/scenic_renderer.py``) emits
a Scenic program whose generated ``Scene`` object exposes only:

* ``scene.objects`` — a flat tuple containing **all** Scenic objects, including
  ``LIBEROFixture`` instances, regular ``LIBEROObject`` task objects, and
  ``LIBEROObject`` distractors (``libero_name`` starts with ``"distractor_"``).
* ``scene.params`` — a mapping with global per-axis parameters (camera angles,
  lighting intensities, robot init qpos, background textures, articulation
  joint targets, sampled object asset classes, …).

Light / camera / robot / background are **not** materialised as Scenic objects
on the Scene — they live in ``scene.params`` because the simulator consumes
them as bulk parameters at env-construction time, not as Scenic geometry.

Per the RCA at ``~/.omar/ea/4/validation_run/rca/stage3_g4_identity_adapter_gap.md``
the renderer correctly gates per-axis randomness on ``plan.active_axes`` —
when an axis is inactive, **nothing** is emitted for it. So the no-axes
baseline and any inactive-axis-only perturbed sample agree on every per-axis
decision *by construction*. The identity hook could not see that agreement
because its readers (``_lights``, ``_cameras``, ``_distractor_set``,
``_fixture_joint_states``) returned ``_MISSING`` on every Scenic Scene.

This adapter materialises the missing schema on top of a Scenic Scene:

* ``fixtures``    — Scenic objects whose declared class name ends in
                    ``Fixture`` (e.g. ``LIBEROFixture``), augmented with
                    ``joint_states`` synthesised from ``params['articulation_<name>']``.
* ``objects``     — non-fixture, non-distractor Scenic objects (the task objects).
* ``distractors`` — Scenic objects whose ``libero_name`` starts with
                    ``"distractor_"``.
* ``lights``      — synthesised from lighting params (``light_intensity``,
                    ``light_*_offset``, ``ambient_level``); ``()`` when no
                    lighting params are present.
* ``cameras``     — synthesised from camera params (``cam_azimuth``,
                    ``cam_elevation``, ``cam_distance``); ``()`` when absent.
* ``robot``       — synthesised from robot params (``robot_init_qpos``,
                    ``robot_init_radius``, ``robot_model``); ``None`` when
                    absent.
* ``background``  — string derived from ``params['wall_texture']`` /
                    ``['floor_texture']``; ``None`` when absent.

The crucial invariant: **two scenes that did not activate axis X both expose
identical (often empty) per-axis content here**, so the existing per-axis
identity assertions (which compare baseline ↔ perturbed) correctly return
``True``. Inactive on both → vacuous identity, surfaced as empty-tuple equality
rather than ``<missing>`` mismatch.

This module is duck-typed by intent: scenes that already expose the richer
schema (e.g. the existing ``SimpleNamespace`` unit-test fixtures in
``tests/test_invariants_identity.py``) are returned unchanged by
:func:`wrap_scene` so the per-axis assertion logic is unchanged across both
worlds.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_scenic_scene(scene: Any) -> bool:
    """A Scenic ``Scene`` exposes ``.objects`` and ``.params`` but no ``.fixtures``."""
    if scene is None:
        return False
    if hasattr(scene, "fixtures") or hasattr(scene, "lights") or hasattr(scene, "cameras"):
        return False
    return hasattr(scene, "objects") and hasattr(scene, "params")


def _scenic_class_name(obj: Any) -> str:
    """Return the Scenic class name for a sampled object.

    Scenic objects carry their declared class in ``_class_`` and/or
    ``__class__.__name__``. ``LIBEROFixture`` and ``LIBEROObject`` are the two
    relevant kinds emitted by ``renderer.scenic_renderer``.
    """
    cls = getattr(obj, "_class_", None) or getattr(obj, "__class__", None)
    return getattr(cls, "__name__", "") if cls is not None else ""


def _libero_name(obj: Any) -> str | None:
    """Return the ``libero_name`` declared on a Scenic LIBEROObject/LIBEROFixture."""
    name = getattr(obj, "libero_name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _is_fixture(obj: Any) -> bool:
    return _scenic_class_name(obj).endswith("Fixture")


def _is_distractor(obj: Any) -> bool:
    name = _libero_name(obj)
    return name is not None and name.startswith("distractor_")


def _is_task_object(obj: Any) -> bool:
    return (not _is_fixture(obj)) and (not _is_distractor(obj)) and _libero_name(obj) is not None


# ---------------------------------------------------------------------------
# Per-object views (expose the field names identity.py reads)
# ---------------------------------------------------------------------------


def _obj_position(obj: Any) -> tuple[float, float, float] | None:
    pos = getattr(obj, "position", None)
    if pos is None:
        return None
    # Scenic Vector supports .x .y .z and indexing.
    try:
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    except (TypeError, IndexError, KeyError):
        try:
            return (float(pos.x), float(pos.y), float(pos.z))
        except AttributeError:
            return None


def _obj_asset_class(obj: Any) -> str | None:
    ac = getattr(obj, "asset_class", None)
    if isinstance(ac, str) and ac:
        return ac
    return None


def _obj_material(obj: Any, params: Mapping[str, Any]) -> Any:
    """Best-effort material projection.

    Scenic emits no per-object material; texture is a global scene parameter
    (``table_texture`` for objects/fixtures, ``wall_texture``/``floor_texture``
    for the background). When the texture axis is inactive nothing is emitted,
    so both baseline and perturbed see the same absence — identity passes.
    """
    # Per-object: object-class-keyed override (renderer does not emit one
    # today; reserved for future per-object material attributes).
    direct = getattr(obj, "material", None)
    if direct is not None:
        return direct
    # Fall back to global table_texture (texture axis bulk param).
    return params.get("table_texture") if params is not None else None


@dataclass(frozen=True)
class _ObjectView:
    """Adapter exposing identity-friendly attribute names on a Scenic object."""

    name: str
    class_id: str | None
    position: tuple[float, float, float] | None
    material: Any

    # Mirror under both class_id and asset/asset_name to match the legacy
    # identity readers' alias preferences.
    @property
    def asset(self) -> Any:
        return self.class_id

    @property
    def asset_name(self) -> Any:
        return self.class_id

    @property
    def id(self) -> str:
        return self.name


@dataclass(frozen=True)
class _FixtureView:
    name: str
    class_id: str | None
    position: tuple[float, float, float] | None
    material: Any
    joint_states: Mapping[str, float]
    is_fixed: bool = True

    @property
    def joints(self) -> Mapping[str, float]:
        return self.joint_states

    @property
    def id(self) -> str:
        return self.name


@dataclass(frozen=True)
class _DistractorView:
    name: str
    class_id: str | None
    position: tuple[float, float, float] | None

    @property
    def id(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Param-derived axes
# ---------------------------------------------------------------------------


# Canonical light layout — when the lighting axis is inactive, no params are
# emitted, and both sides see ``()``. When it IS active (and therefore not
# checked by the identity invariant) we still expose a single synthetic light
# so downstream consumers can introspect it.
_LIGHT_PARAM_KEYS = (
    "light_intensity",
    "light_x_offset",
    "light_y_offset",
    "light_z_offset",
    "ambient_level",
)
_CAMERA_PARAM_KEYS = ("cam_azimuth", "cam_elevation", "cam_distance")
_ROBOT_PARAM_KEYS = ("robot_init_qpos", "robot_init_radius", "robot_model")
_BACKGROUND_PARAM_KEYS = ("wall_texture", "floor_texture")


def _any_present(params: Mapping[str, Any], keys: Iterable[str]) -> bool:
    return any(k in params for k in keys)


def _build_lights(params: Mapping[str, Any]) -> tuple:
    if not _any_present(params, _LIGHT_PARAM_KEYS):
        return ()
    intensity = params.get("light_intensity")
    pos = (
        float(params.get("light_x_offset", 0.0) or 0.0),
        float(params.get("light_y_offset", 0.0) or 0.0),
        float(params.get("light_z_offset", 0.0) or 0.0),
    )
    return (
        SimpleNamespace(
            name="scene_key_light",
            position=pos,
            intensity=float(intensity) if intensity is not None else 0.0,
        ),
    )


def _build_cameras(params: Mapping[str, Any]) -> tuple:
    if not _any_present(params, _CAMERA_PARAM_KEYS):
        return ()
    az = float(params.get("cam_azimuth", 0.0) or 0.0)
    el = float(params.get("cam_elevation", 0.0) or 0.0)
    dist = float(params.get("cam_distance", 0.0) or 0.0)
    # Encode (az, el, dist) as a 4-tuple "rotation" placeholder + position.
    return (
        SimpleNamespace(
            name="agentview",
            position=(0.0, -dist, 0.0),
            rotation=(az, el, dist, 0.0),
        ),
    )


def _build_robot(params: Mapping[str, Any]) -> Any:
    if not _any_present(params, _ROBOT_PARAM_KEYS):
        # Inactive on both sides → expose a stable canonical sentinel so the
        # identity assertion sees equal-on-equal rather than ``<missing>``.
        return SimpleNamespace(
            name="__inactive__",
            model="__inactive__",
            init_qpos=None,
            init_joint_config=None,
        )
    qpos = params.get("robot_init_qpos")
    if qpos is not None:
        qpos = tuple(float(q) for q in qpos)
    return SimpleNamespace(
        name=params.get("robot_model", "panda"),
        model=params.get("robot_model", "panda"),
        init_qpos=qpos,
        init_joint_config=qpos,
    )


def _build_background(params: Mapping[str, Any]) -> Any:
    if not _any_present(params, _BACKGROUND_PARAM_KEYS):
        # Stable canonical sentinel — identical on both sides when inactive.
        return SimpleNamespace(
            name="__inactive__",
            id="__inactive__",
            wall=None,
            floor=None,
        )
    wall = params.get("wall_texture")
    floor = params.get("floor_texture")
    # Encode as a deterministic string so dotted lookup ``background`` and
    # ``background.name`` both resolve.
    name = f"wall={wall};floor={floor}"
    return SimpleNamespace(name=name, id=name, wall=wall, floor=floor)


def _fixture_joint_states(name: str, params: Mapping[str, Any]) -> Mapping[str, float]:
    """Read ``param articulation_<name>`` (the only articulation field the
    renderer emits per fixture). Returns ``{}`` when absent."""
    from re import sub

    san = sub(r"[^A-Za-z0-9_]", "_", name)
    key = f"articulation_{san}"
    if key not in params:
        return {}
    return {"joint": float(params[key])}


# ---------------------------------------------------------------------------
# SceneView
# ---------------------------------------------------------------------------


class SceneView:
    """Read-only adapter exposing the identity-invariant schema over a Scenic Scene."""

    __slots__ = ("_scene", "_params", "_objects", "_fixtures", "_distractors")

    def __init__(self, scene: Any) -> None:
        self._scene = scene
        self._params: Mapping[str, Any] = dict(getattr(scene, "params", {}) or {})
        raw_objs = list(getattr(scene, "objects", ()) or ())

        fixtures: list[_FixtureView] = []
        objects: list[_ObjectView] = []
        distractors: list[_DistractorView] = []

        for o in raw_objs:
            name = _libero_name(o)
            if name is None:
                # Skip Scenic primitives that aren't LIBEROObjects (e.g. region
                # markers). Identity invariants are scoped to LIBERO-tagged
                # entities only.
                continue
            pos = _obj_position(o)
            cls = _scenic_class_name(o)
            asset_class = _obj_asset_class(o)

            if _is_fixture(o):
                fixtures.append(
                    _FixtureView(
                        name=name,
                        class_id=asset_class or cls or None,
                        position=pos,
                        material=_obj_material(o, self._params),
                        joint_states=_fixture_joint_states(name, self._params),
                    )
                )
            elif _is_distractor(o):
                distractors.append(
                    _DistractorView(
                        name=name,
                        class_id=asset_class or cls or None,
                        position=pos,
                    )
                )
            else:
                objects.append(
                    _ObjectView(
                        name=name,
                        class_id=asset_class or cls or None,
                        position=pos,
                        material=_obj_material(o, self._params),
                    )
                )

        # Stable ordering so the diff is deterministic.
        self._objects = tuple(sorted(objects, key=lambda o: o.name))
        self._fixtures = tuple(sorted(fixtures, key=lambda o: o.name))
        self._distractors = tuple(sorted(distractors, key=lambda o: o.name))

    # ---- duck-typed identity-schema surface --------------------------------

    @property
    def objects(self) -> tuple:
        return self._objects

    @property
    def fixtures(self) -> tuple:
        return self._fixtures

    @property
    def distractors(self) -> tuple:
        return self._distractors

    @property
    def lights(self) -> tuple:
        return _build_lights(self._params)

    @property
    def cameras(self) -> tuple:
        return _build_cameras(self._params)

    @property
    def robot(self) -> Any:
        return _build_robot(self._params)

    @property
    def background(self) -> Any:
        return _build_background(self._params)

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def wrap_scene(scene: Any) -> Any:
    """Return a scene that exposes the richer identity-schema.

    * Already-richer scenes (e.g. ``SimpleNamespace`` with ``.fixtures`` /
      ``.lights`` / ``.cameras``) are returned unchanged.
    * Bare Scenic ``Scene`` objects (only ``.objects`` + ``.params``) are
      wrapped in a :class:`SceneView`.
    * ``None`` returns ``None`` so callers can short-circuit upstream.
    """
    if scene is None:
        return None
    if _is_scenic_scene(scene):
        return SceneView(scene)
    return scene
