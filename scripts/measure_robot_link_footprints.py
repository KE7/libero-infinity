"""Generate ``data/robot_link_footprints.json`` — the per-robot-link world-frame
AABB + linearized forward-kinematics model used by the Scenic renderer to put
the *perturbed robot init pose* into the sampling require graph (Fix 1 of the
consolidated placement-clearance PR; see
``rca/stage1_g5_pose_tolerance_object_axis_and_settle_drift.md`` Finding B).

Why this exists
---------------
The robot-axis perturbation jitters the Panda's init joint vector inside a ball
of radius ``[radius_lo, radius_hi]`` (rad) around the canonical home qpos. Scenic
currently has **no** constraint coupling that perturbed arm pose to the placed
objects, so it accepts samples where a perturbed link's volume intersects a
task object / distractor / fixture; MuJoCo's settle then resolves the
penetration by shoving the object 40–260 mm in xy (the dominant
``pose_tolerance`` failure mode).

To put the arm in the require graph we need, for every arm link the
perturbation can move:

  * its world-frame origin at the canonical pose (``x0, y0, z0``),
  * the position Jacobian of that origin w.r.t. the 7 arm joints
    (``jx, jy, jz`` — each length 7), so the renderer can emit the link's
    *perturbed* world position as a **linear** Scenic expression of the same
    sampled joint deltas it already emits, and
  * a conservative per-axis half-extent (``hx, hy, hz``) that **outer-bounds**
    the link's true world geometry over the entire perturbation envelope —
    folding in both the link's geometric size and the linearization residual,
    so the emitted box is guaranteed to contain the real link for every sample
    (no false "clear", i.e. the constraint never accepts a real collision).

The renderer emits a 3-D separating-axis (SAT) clause per (link, object):

    require (abs(Lx - ox) > hx + ow/2)
         or (abs(Ly - oy) > hy + ol/2)
         or (abs(Lz - oz) > hz + oh/2)

The **z** term is essential, not optional: at the canonical home pose the arm
sits ~30 cm *above* the table objects, so a pure-xy shadow constraint would
reject every object beneath the (stationary) arm. The z projection lets the
sampler accept poses where the link is genuinely above the object, and only
rejects poses where the perturbed link actually dips into the object's slab.

Reproducible: deterministic seed, fixed envelope from the planner. Re-run after
a robot-model or ``RobotInitPlan`` envelope change:

    PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/measure_robot_link_footprints.py
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

# Deterministic envelope sampling — no Math.random / Date.now equivalents.
_N_SAMPLES = 1500
_SEED = 0

# Bodies excluded from the clearance model: the static pedestal/base and link0
# never move under arm-joint perturbation (Jacobian ≡ 0) and sit ~0.66 m behind
# the workspace, far from any placeable object. Everything else that belongs to
# the arm or gripper is measured.
_EXCLUDE_SUFFIXES = ("base", "link0")


def _body_world_aabb(model, data, body_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """World-frame AABB (min, max) of every geom attached to ``body_id``.

    Uses MuJoCo's local geom AABB rotated into the world frame when available
    (tight), falling back to the geom bounding sphere ``geom_rbound`` (a
    conservative outer bound). Returns ``None`` if the body has no geoms.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = False
    geom_aabb = getattr(model, "geom_aabb", None)
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        found = True
        gpos = np.asarray(data.geom_xpos[gi], dtype=float)
        gmat = np.asarray(data.geom_xmat[gi], dtype=float).reshape(3, 3)
        if geom_aabb is not None:
            cen = np.asarray(geom_aabb[gi][:3], dtype=float)
            half = np.asarray(geom_aabb[gi][3:6], dtype=float)
            world_cen = gpos + gmat @ cen
            # World AABB half-extent of an oriented box: |R| @ half.
            world_half = np.abs(gmat) @ half
            lo = np.minimum(lo, world_cen - world_half)
            hi = np.maximum(hi, world_cen + world_half)
        else:
            r = float(model.geom_rbound[gi])
            lo = np.minimum(lo, gpos - r)
            hi = np.maximum(hi, gpos + r)
    if not found:
        return None
    return lo, hi


def measure() -> dict:
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.planner.axes import _PANDA_INIT_QPOS
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    # A robot-axis envelope is robot-model-global (task-invariant): the link
    # footprints depend only on the canonical qpos + radius ball, never on the
    # scene. Build one kitchen task purely to obtain the authoritative MuJoCo
    # Panda model.
    bddl = str(resolve_task_path("libero_goal/put_the_bowl_on_the_stove.bddl"))
    cfg = TaskConfig.from_bddl(bddl)
    import random

    random.seed(_SEED)
    scenario = compile_task_to_scenario(cfg, "position")
    scene, _ = scenario.generate(maxIterations=2000)
    env = make_env(scene, bddl_path=bddl)
    env.reset()

    sim = env._sim.libero_env.env.sim
    model = sim.model
    data = sim.data
    robot = env._sim.libero_env.env.robots[0]
    joint_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    n_dof = int(joint_idx.shape[0])

    # Resolve the radius envelope from the planner (single source of truth).
    radius_lo, radius_hi = _resolve_radius_envelope()

    canon = np.asarray(_PANDA_INIT_QPOS, dtype=float)

    names = [model.body_id2name(i) for i in range(model.nbody)]
    link_bodies: list[tuple[str, int]] = []
    for bid, nm in enumerate(names):
        if not nm:
            continue
        if "robot0" not in nm and "gripper0" not in nm:
            continue
        if any(nm.endswith(sfx) for sfx in _EXCLUDE_SUFFIXES):
            continue
        link_bodies.append((nm, bid))

    def _forward(q: np.ndarray) -> None:
        data.qpos[joint_idx] = q
        data.qvel[joint_idx] = 0.0
        sim.forward()

    # Canonical-pose origins.
    _forward(canon)
    table_world_z = float(data.body_xpos[model.body_name2id("robot0_base")][2])
    origin0: dict[str, np.ndarray] = {}
    for nm, bid in link_bodies:
        origin0[nm] = np.asarray(data.body_xpos[bid], dtype=float).copy()

    # Position Jacobian of each link origin w.r.t. the 7 arm joints
    # (central finite difference at the canonical pose).
    eps = 1e-4
    jac: dict[str, np.ndarray] = {nm: np.zeros((3, n_dof)) for nm, _ in link_bodies}
    for k in range(n_dof):
        qp = canon.copy()
        qp[k] += eps
        _forward(qp)
        plus = {nm: np.asarray(data.body_xpos[bid], dtype=float).copy() for nm, bid in link_bodies}
        qm = canon.copy()
        qm[k] -= eps
        _forward(qm)
        for nm, bid in link_bodies:
            minus = np.asarray(data.body_xpos[bid], dtype=float)
            jac[nm][:, k] = (plus[nm] - minus) / (2.0 * eps)
    _forward(canon)

    # Sample the perturbation envelope; for each sample compute the true world
    # AABB of each link and its deviation from the LINEAR prediction, so the
    # stored half-extent outer-bounds the link for every sample in the ball.
    rng = np.random.default_rng(_SEED)
    half_dev: dict[str, np.ndarray] = {nm: np.zeros(3) for nm, _ in link_bodies}
    z_lo: dict[str, float] = {nm: math.inf for nm, _ in link_bodies}
    z_hi: dict[str, float] = {nm: -math.inf for nm, _ in link_bodies}

    for s in range(_N_SAMPLES):
        d = rng.standard_normal(n_dof)
        d /= np.linalg.norm(d) + 1e-12
        # Bias toward the envelope boundary (radius_hi) where the linearization
        # residual is largest — the stored pad must cover the worst case.
        r = radius_hi if (s % 2 == 0) else rng.uniform(radius_lo, radius_hi)
        dq = r * d
        q = canon + dq
        _forward(q)
        for nm, bid in link_bodies:
            aabb = _body_world_aabb(model, data, bid)
            if aabb is None:
                continue
            lo, hi = aabb
            pred = origin0[nm] + jac[nm] @ dq
            # Per-axis max distance from the linear-predicted centre to either
            # AABB face = conservative half-extent around the predicted centre.
            dev = np.maximum(np.abs(hi - pred), np.abs(lo - pred))
            half_dev[nm] = np.maximum(half_dev[nm], dev)
            z_lo[nm] = min(z_lo[nm], float(lo[2]))
            z_hi[nm] = max(z_hi[nm], float(hi[2]))

    env.close()

    links_out = []
    for nm, _bid in link_bodies:
        o = origin0[nm]
        links_out.append(
            {
                "name": nm,
                "x0": round(float(o[0]), 6),
                "y0": round(float(o[1]), 6),
                "z0": round(float(o[2]), 6),
                "jx": [round(float(v), 6) for v in jac[nm][0]],
                "jy": [round(float(v), 6) for v in jac[nm][1]],
                "jz": [round(float(v), 6) for v in jac[nm][2]],
                "hx": round(float(half_dev[nm][0]), 6),
                "hy": round(float(half_dev[nm][1]), 6),
                "hz": round(float(half_dev[nm][2]), 6),
                "z_min": round(float(z_lo[nm]), 6),
                "z_max": round(float(z_hi[nm]), 6),
            }
        )

    return {
        "_meta": {
            "description": "Per-robot-link world-frame AABB + linearized FK "
            "Jacobian over the robot-axis perturbation envelope. Generated by "
            "scripts/measure_robot_link_footprints.py from the authoritative "
            "LIBERO/robosuite Panda model. hx/hy/hz conservatively outer-bound "
            "the link geometry over the whole [radius_lo, radius_hi] ball.",
            "n_samples": _N_SAMPLES,
            "seed": _SEED,
        },
        "robots": {
            "Panda": {
                "canonical_qpos": [round(float(q), 8) for q in canon],
                "radius_lo": radius_lo,
                "radius_hi": radius_hi,
                "n_dof": n_dof,
                "table_world_z": round(table_world_z, 6),
                "links": links_out,
            }
        },
    }


def _resolve_radius_envelope() -> tuple[float, float]:
    """Read the robot-axis radius envelope from the planner (single source)."""
    from libero_infinity.planner.axes import plan_robot

    plan = plan_robot(None, frozenset({"robot"}), None)  # type: ignore[arg-type]
    if plan is None:
        raise RuntimeError("planner returned no RobotInitPlan for the robot axis")
    return float(plan.radius_lo), float(plan.radius_hi)


if __name__ == "__main__":
    out = measure()
    dest = pathlib.Path("src/libero_infinity/data/robot_link_footprints.json")
    dest.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    p = out["robots"]["Panda"]
    print(f"\nWrote {dest}: {len(p['links'])} links, radius [{p['radius_lo']}, {p['radius_hi']}]")
    for lk in p["links"]:
        print(
            f"  {lk['name']:24} x0={lk['x0']:+.3f} y0={lk['y0']:+.3f} z0={lk['z0']:+.3f} "
            f"h=({lk['hx']:.3f},{lk['hy']:.3f},{lk['hz']:.3f}) "
            f"z=[{lk['z_min']:.3f},{lk['z_max']:.3f}]"
        )
