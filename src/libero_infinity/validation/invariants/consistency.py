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
from ._scene_view import is_scene_fixture, resolve_object_name
from .domain import _iter_scene_objects, _obj_class, _obj_position

DEFAULT_POS_TOL = 5e-3
DEFAULT_ROT_TOL_DEG = 1.0

# ---------------------------------------------------------------------------
# Alt-rest (valid alternate physical rest) acceptance
# ---------------------------------------------------------------------------
# The strict pose gate (``pos_err <= DEFAULT_POS_TOL`` AND
# ``rot_err <= DEFAULT_ROT_TOL_DEG``) demands that the SETTLED pose be an exact
# fixed point of the EMITTED pose. That is stricter than the physical claim g4
# actually needs to make — "the object was placed correctly on its support" —
# because a tall / multi-face object can deterministically settle into a genuine
# SECOND stable rest (a second energy minimum): still upright, still on the same
# support surface, still inside its sampled placement footprint, but with its
# origin slid a few mm past the 5 mm gate (RCA g4_fixed_point_settle.md §3,
# g4_metastable_residual.md §6 — these are stable fixed points, rot_err = 0°,
# that iterating the settle does NOT relax back).
#
# The alt-rest path admits EXACTLY those settles and NOTHING else. A settle that
# misses the strict gate is accepted as a valid alternate rest ONLY when it is
# simultaneously:
#   1. AT REST     — end-of-settle DISPLACEMENT (over the last few settle steps)
#                    below the convergence thresholds (settled to a fixed point,
#                    not a transient still moving when the settle stopped),
#   2. UPRIGHT     — rotation error vs the emitted canonical orientation within
#                    the SAME 1° tolerance the strict gate uses (rejects tips),
#   3. ON SUPPORT  — vertical deviation within a fraction of the object's own
#                    height (rejects fall-through-floor and wrong-support climbs),
#   4. IN REGION   — horizontal drift within the object's own planar footprint
#                    (capped absolutely) (rejects out-of-region slides / far
#                    perches — the object did not travel to a distinct location).
# Every threshold is anchored to the object's own measured geometry or the
# existing rotation tolerance — none is tuned to the residual tail. The class is
# checked separately (``assert_class_match``); the alt-rest path scores the SAME
# body_id, so it can never admit a wrong object.
#
# This is a strict SUPERSET of the old gate (``passed = strict OR alt_rest``), so
# it is a guaranteed NET-ADD: anything that passed the strict gate still passes.
ALT_REST_UPRIGHT_TOL_DEG = DEFAULT_ROT_TOL_DEG  # upright == strict rot tol (1°)
ALT_REST_Z_HEIGHT_FRAC = 0.25  # |settled_z - emitted_z| <= 1/4 object height
ALT_REST_XY_ABS_CAP = 0.05  # absolute cap on horizontal drift (m)
# "At rest" = converged: the NET DRIFT of the vibration-averaged position over
# the settle tail (mean position of the first vs second half of the last window
# of settle steps; see simulator._settle_convergence) must be small. Measured
# (scripts/measure_altrest_thresholds.py, living_room tall/multi-face objects):
# there is a DETERMINISTIC contact-vibration floor of ~8–10 mm / <0.1° for large
# upright objects (a limit-cycle driven by the held arm + contact solver — the
# object is NOT translating away, its net-from-emitted xy is ~0). The genuine
# alternate rests we admit net-drift 2–4 mm / ≤1.4°; a still-settling / climbing
# object drifts 19–22 mm / 2–8°. The thresholds sit ABOVE the vibration floor and
# below gross motion, so the convergence gate catches gross non-convergence while
# the vibration floor never false-rejects a resting object. (The upright /
# on-support / in-region gates below, plus the pipeline's 50-step settle and the
# 35°/0.20 m resampler, are the primary assurance the pose is a valid rest; this
# gate adds an explicit "did the object stop moving?" check on top.)
ALT_REST_CONV_LIN_TOL = 0.012  # metres of NET position drift over the settle tail
ALT_REST_CONV_ANG_TOL = 2.0  # degrees of NET orientation drift over the settle tail
DEFAULT_ACCEPT_ALT_REST = True

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


def _obj_extents(o: Any) -> tuple[float, float, float] | None:
    """Return the object's MEASURED ``(width, length, height)`` in metres, or None.

    The alt-rest on-support / in-region bounds must be anchored to the object's
    real geometry. The renderer emits task objects WITHOUT explicit bbox
    specifiers (they carry the ``LIBEROObject`` defaults 0.08/0.08/0.06), so we
    resolve the true dimensions from the asset registry via the object's class,
    and fall back to the ``width/length/height`` attributes only when the class
    is not in the registry (e.g. distractors / unit-test doubles that set them).
    Missing on both paths ⇒ ``None`` (the alt-rest path then declines to the
    strict gate).
    """
    cls = None
    for attr in ("asset_class", "object_class", "class_", "cls"):
        v = getattr(o, attr, None)
        if isinstance(v, str) and v:
            cls = v
            break
    if cls is not None:
        try:
            from libero_infinity.asset_registry import OBJECT_DIMENSIONS, get_dimensions

            if cls in OBJECT_DIMENSIONS:
                w, length, h = get_dimensions(cls)
                return (float(w), float(length), float(h))
        except Exception:  # noqa: BLE001 — registry issue ⇒ fall back to attrs
            pass
    w = getattr(o, "width", None)
    length = getattr(o, "length", None)
    h = getattr(o, "height", None)
    try:
        return (float(w), float(length), float(h))
    except (TypeError, ValueError):
        return None


def _alt_rest_valid(
    scenic_obj: Any,
    env_obj_state: dict[str, Any],
    *,
    s_pos: tuple[float, float, float],
    e_pos: tuple[float, float, float],
    upright_err_deg: float | None,
    upright_tol_deg: float,
    z_height_frac: float,
    xy_abs_cap: float,
    conv_lin_tol: float,
    conv_ang_tol: float,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Decide whether a non-strict settle is a VALID ALTERNATE PHYSICAL REST.

    Returns ``(accepted, reject_reason, info)``. Accepted ⇔ the object is at
    rest, upright, on its support, and within its own footprint of the emitted
    xy (see the module-level acceptance contract). ``reject_reason`` is the FIRST
    failed condition (so a rejected settle is never silently masked — the reason
    is surfaced in the payload). Conservative on missing data: any absent signal
    (rotation, convergence, extents) ⇒ decline (fall back to strict).
    """
    info: dict[str, Any] = {}

    # (2) UPRIGHT — the settled orientation must be within the upright tolerance
    # of the object's canonical (as-placed) orientation. ``upright_err_deg`` is
    # computed by the caller, preferring env-settled-vs-canonical (robust) over
    # the scenic-vs-env term (vacuous for real objects whose Scenic orientation
    # does not coerce). Unknown ⇒ decline.
    if upright_err_deg is None:
        return (False, "no_rotation_data", info)
    info["upright_err_deg"] = upright_err_deg
    if upright_err_deg > upright_tol_deg:
        return (False, "tipped", info)

    # (1) AT REST — end-of-settle convergence displacement must be known and
    # below the convergence tolerances (object settled to a fixed point).
    lin = env_obj_state.get("settle_conv_lin")
    ang = env_obj_state.get("settle_conv_ang")
    if lin is None or ang is None:
        return (False, "no_convergence_signal", info)
    lin = float(lin)
    ang = float(ang)
    info["settle_conv_lin"] = lin
    info["settle_conv_ang"] = ang
    if lin > conv_lin_tol or ang > conv_ang_tol:
        return (False, "not_converged", info)

    # Object geometry is required for the on-support / in-region bounds.
    ext = _obj_extents(scenic_obj)
    if ext is None:
        return (False, "no_extents", info)
    width, length, height = ext
    if height <= 0.0 or max(width, length) <= 0.0:
        return (False, "bad_extents", info)

    # (3) ON SUPPORT — vertical deviation within a fraction of object height.
    # Rejects fall-through-floor (big drop) and wrong-support climb (big rise);
    # admits the small z-shift of a secondary flat / perched rest on the SAME
    # surface. Anchored to the object's own height, not a magic constant.
    dz = abs(s_pos[2] - e_pos[2])
    z_band = z_height_frac * height
    info["dz"] = dz
    info["z_band"] = z_band
    if dz > z_band:
        return (False, "off_support", info)

    # (4) IN REGION — horizontal drift within the object's own planar footprint
    # (half-extent), capped absolutely. Rejects out-of-region slides / distant
    # perches; admits a settle to an adjacent contact within the placement spot.
    planar_half = 0.5 * max(width, length)
    xy_bound = min(planar_half, xy_abs_cap)
    xy = math.hypot(s_pos[0] - e_pos[0], s_pos[1] - e_pos[1])
    info["xy_drift"] = xy
    info["xy_bound"] = xy_bound
    if xy > xy_bound:
        return (False, "out_of_region", info)

    return (True, None, info)


def assert_pose_tolerance(
    scenic_obj: Any,
    env_obj_state: dict[str, Any],
    *,
    pos_tol: float = DEFAULT_POS_TOL,
    rot_tol_deg: float = DEFAULT_ROT_TOL_DEG,
    accept_alt_rest: bool = DEFAULT_ACCEPT_ALT_REST,
    upright_tol_deg: float = ALT_REST_UPRIGHT_TOL_DEG,
    z_height_frac: float = ALT_REST_Z_HEIGHT_FRAC,
    xy_abs_cap: float = ALT_REST_XY_ABS_CAP,
    conv_lin_tol: float = ALT_REST_CONV_LIN_TOL,
    conv_ang_tol: float = ALT_REST_CONV_ANG_TOL,
) -> AssertionResult:
    """Compare Scenic vs env pose for a single object.

    A settle passes when EITHER the strict exact-pose gate holds
    (``pos_err <= pos_tol`` AND ``rot_err <= rot_tol_deg``) OR — when
    ``accept_alt_rest`` — it is a VALID ALTERNATE PHYSICAL REST (at rest,
    upright, on its support surface, within its sampled footprint; see
    ``_alt_rest_valid`` and the module-level acceptance contract). The gate is a
    strict superset of the old one, so enabling alt-rest can never flip a
    strict-passing object to fail. Both the strict result and the alt-rest
    verdict/reject-reason are reported in the payload for audit.

    Set ``accept_alt_rest=False`` to recover the exact legacy strict-only gate
    (used by the OLD-vs-NEW A/B harness).
    """
    name = resolve_object_name(scenic_obj) or "?"
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
    strict_ok = pos_ok and rot_ok
    payload["strict_pass"] = strict_ok

    # "Upright" reference for the alt-rest path: prefer env-settled orientation
    # vs the object's CANONICAL (as-placed) orientation — this is robust and is
    # exactly the tip the resampler / RCA measure. Fall back to the scenic-vs-env
    # term only when no canonical is supplied (e.g. unit-test doubles). This is
    # ONLY used to gate the alt-rest acceptance; the strict gate is untouched.
    c_ori = _coerce_quat(env_obj_state.get("canonical_orientation"))
    upright_err_deg: float | None
    if c_ori is not None and e_ori is not None:
        upright_err_deg = _quat_angle_deg(e_ori, c_ori)
    else:
        upright_err_deg = rot_err_deg
    payload["upright_error_deg"] = upright_err_deg

    # Alt-rest acceptance — only consulted when the strict gate misses, so the
    # gate is a strict superset (guaranteed net-add).
    alt_ok = False
    alt_reason: str | None = None
    if accept_alt_rest and not strict_ok:
        alt_ok, alt_reason, alt_info = _alt_rest_valid(
            scenic_obj,
            env_obj_state,
            s_pos=s_pos,
            e_pos=e_pos_t,
            upright_err_deg=upright_err_deg,
            upright_tol_deg=upright_tol_deg,
            z_height_frac=z_height_frac,
            xy_abs_cap=xy_abs_cap,
            conv_lin_tol=conv_lin_tol,
            conv_ang_tol=conv_ang_tol,
        )
        payload["alt_rest_info"] = alt_info
    payload["alt_rest_pass"] = alt_ok
    payload["alt_rest_reject_reason"] = alt_reason

    passed = strict_ok or alt_ok
    payload["accept_mode"] = "strict" if strict_ok else ("alt_rest" if alt_ok else "reject")

    if passed:
        mode = "within tol" if strict_ok else "valid alternate rest"
        return AssertionResult(
            name="pose_tolerance",
            passed=True,
            detail=(
                f"{name}: pos_err={pos_err:.5f}m"
                + (f", rot_err={rot_err_deg:.3f}°" if rot_err_deg is not None else "")
                + f" ({mode})."
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
            + (f"; alt_rest rejected: {alt_reason}" if alt_reason is not None else "")
        ),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# C2 — class match
# ---------------------------------------------------------------------------


def assert_class_match(scenic_obj: Any, env_obj_state: dict[str, Any]) -> AssertionResult:
    """Scenic object's asset class must equal the env's class string."""
    name = resolve_object_name(scenic_obj) or "?"
    s_cls = _obj_class(scenic_obj) or None
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
    accept_alt_rest: bool = DEFAULT_ACCEPT_ALT_REST,
    names: Iterable[str] | None = None,
) -> list[AssertionResult]:
    """Iterate scene objects and produce one (pose, class) result per object.

    Missing env entries are reported as *failures* — silent omission would
    defeat the purpose of this check.
    """
    # Fixtures are scene structure injected from the BDDL :fixtures block, not
    # Scenic-sampled task assets — the Scenic↔env consistency check is scoped
    # to the sampled movable objects (and distractors).
    objs = [o for o in _iter_scene_objects(scene) if not is_scene_fixture(o)]
    if names is not None:
        keep = set(names)
        objs = [o for o in objs if resolve_object_name(o) in keep]
    results: list[AssertionResult] = []
    for o in objs:
        nm = resolve_object_name(o) or "?"
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
        results.append(
            assert_pose_tolerance(
                o,
                state,
                pos_tol=pos_tol,
                rot_tol_deg=rot_tol_deg,
                accept_alt_rest=accept_alt_rest,
            )
        )
        results.append(assert_class_match(o, state))
    return results
