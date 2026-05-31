"""Robot link-footprint metadata — the world-frame AABB + linearized
forward-kinematics model the Scenic renderer uses to place the *perturbed robot
init pose* into the sampling require graph (Fix 1 of the consolidated
placement-clearance PR; RCA Finding B).

The data is **measured** from the authoritative LIBERO/robosuite Panda model by
``scripts/measure_robot_link_footprints.py`` and stored in
``data/robot_link_footprints.json``. Re-run that generator after a robot-model
or ``RobotInitPlan`` envelope change. Nothing here is hardcoded geometry — this
module only loads and exposes the measured tables.
"""

from __future__ import annotations

import json
import pkgutil
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotLink:
    """One robot link's measured footprint + linearized FK model.

    All quantities are in the MuJoCo world frame at the canonical home pose.

    Attributes:
        name:  MuJoCo body name.
        x0, y0, z0:  Link-origin world position at the canonical qpos.
        jx, jy, jz:  Position Jacobian rows (length n_dof) of the link origin
            w.r.t. the arm joints — so ``world_pos ≈ (x0,y0,z0) + J @ dq`` for a
            joint delta ``dq`` from canonical.
        hx, hy, hz:  Conservative per-axis half-extents that outer-bound the
            link's true world geometry over the WHOLE perturbation envelope
            (geometry size + linearization residual folded together).
        z_min, z_max:  Swept world-z range of the link geometry over the
            envelope — used for an exact static z-separation prune.
    """

    name: str
    x0: float
    y0: float
    z0: float
    jx: tuple[float, ...]
    jy: tuple[float, ...]
    jz: tuple[float, ...]
    hx: float
    hy: float
    hz: float
    z_min: float
    z_max: float

    def has_footprint(self) -> bool:
        """True iff the link has measurable geometry (skip pure site/frame bodies)."""
        if not (self.hx > 0.0 or self.hy > 0.0 or self.hz > 0.0):
            return False
        return self.z_min < self.z_max


@dataclass(frozen=True)
class RobotFootprint:
    """Measured footprint model for one robot model (e.g. Panda)."""

    robot_model: str
    canonical_qpos: tuple[float, ...]
    radius_lo: float
    radius_hi: float
    n_dof: int
    table_world_z: float
    links: tuple[RobotLink, ...]

    def active_links(self) -> tuple[RobotLink, ...]:
        """Links with real geometry (drops zero-footprint site/frame bodies)."""
        return tuple(lk for lk in self.links if lk.has_footprint())


def _load() -> dict[str, RobotFootprint]:
    raw = pkgutil.get_data("libero_infinity", "data/robot_link_footprints.json")
    if raw is None:
        return {}
    data = json.loads(raw)
    out: dict[str, RobotFootprint] = {}
    for model_name, spec in data.get("robots", {}).items():
        links = tuple(
            RobotLink(
                name=str(lk["name"]),
                x0=float(lk["x0"]),
                y0=float(lk["y0"]),
                z0=float(lk["z0"]),
                jx=tuple(float(v) for v in lk["jx"]),
                jy=tuple(float(v) for v in lk["jy"]),
                jz=tuple(float(v) for v in lk["jz"]),
                hx=float(lk["hx"]),
                hy=float(lk["hy"]),
                hz=float(lk["hz"]),
                z_min=float(lk["z_min"]),
                z_max=float(lk["z_max"]),
            )
            for lk in spec.get("links", [])
        )
        out[str(model_name)] = RobotFootprint(
            robot_model=str(model_name),
            canonical_qpos=tuple(float(q) for q in spec.get("canonical_qpos", ())),
            radius_lo=float(spec.get("radius_lo", 0.0)),
            radius_hi=float(spec.get("radius_hi", 0.0)),
            n_dof=int(spec.get("n_dof", 0)),
            table_world_z=float(spec.get("table_world_z", 0.0)),
            links=links,
        )
    return out


ROBOT_FOOTPRINTS: dict[str, RobotFootprint] = _load()


def get_robot_footprint(robot_model: str) -> RobotFootprint | None:
    """Return the measured footprint model for ``robot_model`` (or None)."""
    return ROBOT_FOOTPRINTS.get(robot_model)
