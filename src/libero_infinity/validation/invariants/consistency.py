"""G4 Family C — Scenic-scene ↔ LIBERO-env consistency invariants.

After ``env.reset()``, every Scenic-sampled object should be reflected in the
LIBERO environment within tight tolerances. Two checks per object:

* ``assert_pose_tolerance``  — position within ``pos_tol`` m, rotation within
  ``rot_tol_deg`` degrees (axis-angle norm).
* ``assert_class_match``     — same asset class string.

The env interface is duck-typed via ``_env_get_object(env, name)`` which
accepts any of:
    * ``env.get_object_state(name) -> {"position": ..., "orientation": ..., "class": ...}``
    * ``env.objects[name]``                 (attribute-style)
    * ``env.get_pose(name) -> (pos, quat)`` + ``env.get_class(name) -> str``

If no env data is available for ``name``, the result is a *failure* (the env
silently missing an object is a bug, not a skip).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from ._result import AssertionResult
from .domain import _iter_scene_objects, _obj_class, _obj_position

DEFAULT_POS_TOL = 5e-3
DEFAULT_ROT_TOL_DEG = 1.0

__all__ = [
    "AssertionResult",
    "CONSISTENCY_ASSERTIONS",
    "assert_pose_tolerance",
    "assert_class_match",
    "assert_consistency",
]


# ---------------------------------------------------------------------------
# Env duck-typed accessors
# ---------------------------------------------------------------------------


class EnvObjectMissing(LookupError):
    """Raised when the env has no record of a Scenic-named object."""


def _env_get_object(env: Any, name: str) -> dict[str, Any]:
    """Return a normalized ``{position, orientation, class}`` dict for ``name``.

    Tries a sequence of duck-typed accessors. Raises :class:`EnvObjectMissing`
    if none yield data — that is a genuine consistency *failure*, not a skip.
    """
    if env is None:
        raise EnvObjectMissing(f"env is None; cannot resolve {name!r}")

    # 1) get_object_state(name) -> dict-like
    fn = getattr(env, "get_object_state", None)
    if callable(fn):
        st = fn(name)
        if st is not None:
            return _normalize_state(st)

    # 2) env.objects mapping
    objs = getattr(env, "objects", None)
    if isinstance(objs, dict) and name in objs:
        return _normalize_state(objs[name])

    # 3) get_pose + get_class
    gp = getattr(env, "get_pose", None)
    gc = getattr(env, "get_class", None)
    if callable(gp) and callable(gc):
        pose = gp(name)
        cls = gc(name)
        if pose is not None:
            return _normalize_state({"pose": pose, "class": cls})

    raise EnvObjectMissing(f"env has no accessor that resolves {name!r}")


def _normalize_state(st: Any) -> dict[str, Any]:
    """Coerce various env-object representations into ``{position, orientation, class}``."""
    if isinstance(st, dict):
        out = dict(st)
        if "pose" in out and "position" not in out:
            pose = out.pop("pose")
            if pose is not None and len(pose) >= 2:
                out["position"] = pose[0]
                out["orientation"] = pose[1]
        return out
    # Attribute-style fallback
    out = {}
    for key, attrs in (
        ("position", ("position", "pos", "xyz")),
        ("orientation", ("orientation", "quat", "rotation", "ori")),
        ("class", ("object_class", "class_", "asset_class", "cls", "class")),
    ):
        for a in attrs:
            v = getattr(st, a, None)
            if v is not None:
                out[key] = v
                break
    return out


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _coerce_quat(o: Any) -> tuple[float, float, float, float] | None:
    """Return (w, x, y, z) from various orientation representations.

    Supports:
      * 4-tuple/list — assumed (w, x, y, z) (consistent with MuJoCo).
      * scalar yaw   — single float, axis = +Z.
      * dict with ``quat`` or ``yaw``.
    """
    if o is None:
        return None
    if isinstance(o, dict):
        if "quat" in o:
            return _coerce_quat(o["quat"])
        if "yaw" in o:
            return _coerce_quat(float(o["yaw"]))
        return None
    if isinstance(o, (int, float)):
        yaw = float(o)
        half = yaw / 2.0
        return (math.cos(half), 0.0, 0.0, math.sin(half))
    try:
        seq = list(o)
    except TypeError:
        return None
    if len(seq) == 4:
        try:
            return tuple(float(v) for v in seq)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _quat_angle_deg(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> float:
    """Return the rotation angle (degrees) between two unit quaternions."""
    # Normalize defensively (callers may pass un-normalised).
    def _norm(q):
        n = math.sqrt(sum(c * c for c in q))
        if n == 0.0:
            return q
        return tuple(c / n for c in q)

    a = _norm(q1)
    b = _norm(q2)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


# ---------------------------------------------------------------------------
# C1 — pose tolerance
# ---------------------------------------------------------------------------


def assert_pose_tolerance(
    scenic_obj: Any,
    env_obj_state: dict[str, Any],
    *,
    pos_tol: float = DEFAULT_POS_TOL,
    rot_tol_deg: float = DEFAULT_ROT_TOL_DEG,
) -> AssertionResult:
    """Compare Scenic vs env pose for a single object."""
    name = getattr(scenic_obj, "name", "?")
    s_pos = _obj_position(scenic_obj)
    e_pos = env_obj_state.get("position")
    payload: dict[str, Any] = {"name": name, "pos_tol": pos_tol, "rot_tol_deg": rot_tol_deg}
    if s_pos is None or e_pos is None:
        return AssertionResult(
            name="pose_tolerance",
            passed=False,
            detail=f"{name}: missing position data (scenic={s_pos}, env={e_pos}).",
            payload=payload,
        )
    try:
        e_pos_t = (float(e_pos[0]), float(e_pos[1]), float(e_pos[2]))
    except (TypeError, ValueError, IndexError):
        return AssertionResult(
            name="pose_tolerance",
            passed=False,
            detail=f"{name}: env position not a 3-vector ({e_pos!r}).",
            payload=payload,
        )
    dpos = tuple(s - e for s, e in zip(s_pos, e_pos_t))
    pos_err = math.sqrt(sum(c * c for c in dpos))
    payload["scenic_position"] = s_pos
    payload["env_position"] = e_pos_t
    payload["position_error"] = pos_err

    s_ori = _coerce_quat(
        getattr(scenic_obj, "orientation", None) or getattr(scenic_obj, "yaw", None)
    )
    e_ori = _coerce_quat(env_obj_state.get("orientation"))
    rot_err_deg: float | None = None
    if s_ori is not None and e_ori is not None:
        rot_err_deg = _quat_angle_deg(s_ori, e_ori)
        payload["rotation_error_deg"] = rot_err_deg
    else:
        payload["rotation_error_deg"] = None

    pos_ok = pos_err <= pos_tol
    rot_ok = rot_err_deg is None or rot_err_deg <= rot_tol_deg
    if pos_ok and rot_ok:
        return AssertionResult(
            name="pose_tolerance",
            passed=True,
            detail=(
                f"{name}: pos_err={pos_err:.5f}m"
                + (f", rot_err={rot_err_deg:.3f}°" if rot_err_deg is not None else "")
                + " (within tol)."
            ),
            payload=payload,
        )
    return AssertionResult(
        name="pose_tolerance",
        passed=False,
        detail=(
            f"{name}: pos_err={pos_err:.5f}m vs tol {pos_tol}"
            + (
                f"; rot_err={rot_err_deg:.3f}° vs tol {rot_tol_deg}°"
                if rot_err_deg is not None
                else "; no rotation data"
            )
        ),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# C2 — class match
# ---------------------------------------------------------------------------


def assert_class_match(scenic_obj: Any, env_obj_state: dict[str, Any]) -> AssertionResult:
    """Scenic object's asset class must equal the env's class string."""
    name = getattr(scenic_obj, "name", "?")
    s_cls = _obj_class(scenic_obj)
    e_cls = env_obj_state.get("class")
    if s_cls is None or e_cls is None:
        return AssertionResult(
            name="class_match",
            passed=False,
            detail=f"{name}: missing class info (scenic={s_cls!r}, env={e_cls!r}).",
            payload={"name": name, "scenic_class": s_cls, "env_class": e_cls},
        )
    if s_cls != e_cls:
        return AssertionResult(
            name="class_match",
            passed=False,
            detail=f"{name}: class mismatch (scenic={s_cls!r}, env={e_cls!r}).",
            payload={"name": name, "scenic_class": s_cls, "env_class": e_cls},
        )
    return AssertionResult(
        name="class_match",
        passed=True,
        detail=f"{name}: class={s_cls}.",
        payload={"name": name, "scenic_class": s_cls, "env_class": e_cls},
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


CONSISTENCY_ASSERTIONS: tuple[str, ...] = ("pose_tolerance", "class_match")


def assert_consistency(
    scene: Any,
    env: Any,
    *,
    pos_tol: float = DEFAULT_POS_TOL,
    rot_tol_deg: float = DEFAULT_ROT_TOL_DEG,
    names: Iterable[str] | None = None,
) -> list[AssertionResult]:
    """Iterate scene objects and produce one (pose, class) result per object.

    Missing env entries are reported as *failures* — silent omission would
    defeat the purpose of this check.
    """
    objs = _iter_scene_objects(scene)
    if names is not None:
        keep = set(names)
        objs = [o for o in objs if getattr(o, "name", None) in keep]
    results: list[AssertionResult] = []
    for o in objs:
        nm = getattr(o, "name", "?")
        try:
            state = _env_get_object(env, nm)
        except EnvObjectMissing as exc:
            results.append(
                AssertionResult(
                    name="pose_tolerance",
                    passed=False,
                    detail=f"{nm}: env has no record ({exc}).",
                    payload={"name": nm, "reason": "env_missing"},
                )
            )
            results.append(
                AssertionResult(
                    name="class_match",
                    passed=False,
                    detail=f"{nm}: env has no record ({exc}).",
                    payload={"name": nm, "reason": "env_missing"},
                )
            )
            continue
        results.append(assert_pose_tolerance(o, state, pos_tol=pos_tol, rot_tol_deg=rot_tol_deg))
        results.append(assert_class_match(o, state))
    return results
