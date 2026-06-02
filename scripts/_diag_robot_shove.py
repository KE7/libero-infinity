"""RCA Finding-B: reproduce the 2 genuine robot-shove distractor fails and the
pure-z robot-subset fails, and measure the ACTUAL contact mechanism.

For each (task, subset, seed) condition, after env.reset() (which settles):
  * distractor injected (scene) xyz vs settled (body) xyz, xy & z deltas,
  * the perturbed robot init qpos applied (and whether it was clipped vs the
    Scenic-sampled param), and
  * for every active robot link body, the min per-axis AABB gap to the
    distractor body — negative gap on all 3 axes == real geometric overlap
    (the shove source). Reports the nearest link + whether the renderer's
    static z-prune would have guarded that link.
"""

from __future__ import annotations

import random
import sys

import numpy as np

from libero_infinity.compiler import compile_task_to_scenario
from libero_infinity.gym_env import make_env
from libero_infinity.task_config import TaskConfig
from libero_infinity.validation.invariants._scene_view import (
    is_scene_fixture,
    resolve_object_name,
)
from libero_infinity.validation.invariants.consistency import (
    _env_get_object,
    assert_pose_tolerance,
)
from libero_infinity.validation.invariants.domain import _iter_scene_objects
from libero_infinity.validation.sweep import resolve_task_path

# (task, subset, seed) conditions to probe — the 2 genuine xy-shoves + controls.
CONDITIONS = [
    ("libero_goal/push_the_plate_to_the_front_of_the_stove.bddl", "robot,distractor", 2),
    ("libero_goal/put_the_wine_bottle_on_the_rack.bddl", "robot,distractor", 2),
    # pure-z robot-subset control (xy~0 in v5):
    ("libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl", "robot,distractor", 0),
    # distractor-only control (no robot) — same scene, same z residual:
    ("libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl", "distractor", 0),
]


def body_world_aabb(sim, bid):
    """World-frame AABB (lo, hi) over all geoms of body `bid`."""
    model = sim.model
    raw = model._model
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid:
            continue
        aabb = raw.geom_aabb[g]
        c = np.asarray(aabb[:3])
        half = np.asarray(aabb[3:])
        xpos = np.asarray(sim.data.geom_xpos[g])
        rot = np.asarray(sim.data.geom_xmat[g]).reshape(3, 3)
        center = xpos + rot @ c
        ext = np.abs(rot) @ half
        lo = np.minimum(lo, center - ext)
        hi = np.maximum(hi, center + ext)
    if not np.isfinite(lo).all():
        return None
    return lo, hi


def main():
    for task, subset, seed in CONDITIONS:
        bddl = str(resolve_task_path(task))
        print(f"\n==== {task.split('/')[-1]}  subset={subset}  seed={seed}")
        try:
            cfg = TaskConfig.from_bddl(bddl)
            random.seed(seed)
            scn = compile_task_to_scenario(cfg, subset)
            scene, _ = scn.generate(maxIterations=6000)
            env = make_env(scene, bddl_path=bddl)
            env.reset()
        except Exception as exc:
            print(f"  build fail {type(exc).__name__}: {exc}")
            continue

        sim = env._sim.libero_env.env.sim
        params = getattr(scene, "params", {})

        # Sampled vs applied robot qpos (clip check).
        applied = getattr(env._sim, "_applied_robot_init_qpos", None)
        sampled = params.get("robot_init_qpos")
        if applied is not None and sampled is not None:
            sa = np.asarray(sampled, dtype=float)
            ap = np.asarray(applied, dtype=float)
            clip = np.max(np.abs(sa - ap)) if sa.shape == ap.shape else float("nan")
            print(f"  robot qpos clip |sampled-applied|_max = {clip:.5f} rad  radius={params.get('robot_init_radius')}")

        # Robot link bodies.
        link_bids = {}
        for bid in range(sim.model.nbody):
            nm = sim.model.body_id2name(bid)
            if nm and (nm.startswith("robot0_") or nm.startswith("gripper0_")):
                link_bids[nm] = bid

        # Distractors via the SAME path the smoke uses.
        for o in _iter_scene_objects(scene):
            if is_scene_fixture(o):
                continue
            nm = resolve_object_name(o) or "?"
            if not str(nm).startswith("distractor_"):
                continue
            try:
                st = _env_get_object(env, nm)
            except Exception:
                continue
            res = assert_pose_tolerance(o, st)
            p = res.payload
            sp = p.get("scenic_position")
            ep = p.get("env_position")
            if sp is None or ep is None:
                continue
            inj = np.array([float(sp[0]), float(sp[1]), float(sp[2])])
            set_pos = np.array([float(ep[0]), float(ep[1]), float(ep[2])])
            dxy = np.hypot(set_pos[0] - inj[0], set_pos[1] - inj[1]) * 1000
            dz = (set_pos[2] - inj[2]) * 1000
            passed = res.passed
            # settled body for AABB
            bid = None
            for b in range(sim.model.nbody):
                bn = sim.model.body_id2name(b)
                if bn and nm in bn:
                    bid = b
                    break
            d_aabb = body_world_aabb(sim, bid) if bid is not None else None
            print(f"  {nm}: passed={passed} inj=({inj[0]:.3f},{inj[1]:.3f},{inj[2]:.3f}) "
                  f"settle_xy_delta={dxy:.1f}mm z_delta={dz:+.1f}mm")
            if d_aabb is None:
                continue
            dlo, dhi = d_aabb
            # nearest link by min positive 3-axis gap (negative == overlap).
            best = None
            for lname, lbid in link_bids.items():
                lab = body_world_aabb(sim, lbid)
                if lab is None:
                    continue
                llo, lhi = lab
                # per-axis separation: positive if gap, negative if overlap
                gap = np.array([
                    max(dlo[a] - lhi[a], llo[a] - dhi[a]) for a in range(3)
                ])
                # overlap iff all gaps < 0; separation distance = max gap
                sep = float(np.max(gap))
                if best is None or sep < best[1]:
                    best = (lname, sep, gap)
            if best is not None:
                lname, sep, gap = best
                tag = "OVERLAP" if sep < 0 else "clear"
                print(f"      POST-settle nearest link {lname}: sep={sep*1000:+.1f}mm "
                      f"(per-axis gap mm: x={gap[0]*1000:+.1f} y={gap[1]*1000:+.1f} z={gap[2]*1000:+.1f}) [{tag}]")

            # --- DECISIVE: reconstruct the INITIAL (pre-settle) geometric config:
            # perturbed arm at applied qpos + distractor at INJECTED pose, no step.
            if applied is not None and bid is not None:
                try:
                    qadr = []
                    for jn in [f"robot0_joint{k}" for k in range(1, 8)]:
                        jid = sim.model.joint_name2id(jn)
                        qadr.append(int(sim.model.jnt_qposadr[jid]))
                    for k, ad in enumerate(qadr):
                        sim.data.qpos[ad] = float(applied[k])
                    # set distractor free joint to injected pose (pos only).
                    jadr = int(sim.model.jnt_qposadr[int(sim.model.body_jntadr[bid])])
                    sim.data.qpos[jadr:jadr + 3] = inj
                    sim.data.qvel[:] = 0.0
                    sim.forward()
                    d0 = body_world_aabb(sim, bid)
                    if d0 is not None:
                        dlo0, dhi0 = d0
                        best0 = None
                        for lname2, lbid2 in link_bids.items():
                            lab2 = body_world_aabb(sim, lbid2)
                            if lab2 is None:
                                continue
                            llo2, lhi2 = lab2
                            gap2 = np.array([
                                max(dlo0[a] - lhi2[a], llo2[a] - dhi0[a]) for a in range(3)
                            ])
                            sep2 = float(np.max(gap2))
                            if best0 is None or sep2 < best0[1]:
                                best0 = (lname2, sep2, gap2)
                        if best0 is not None:
                            ln2, sp2, gp2 = best0
                            tag2 = "INITIAL-OVERLAP(shove!)" if sp2 < 0 else "initial-clear(NOT a shove)"
                            print(f"      PRE-settle nearest link {ln2}: sep={sp2*1000:+.1f}mm "
                                  f"(x={gp2[0]*1000:+.1f} y={gp2[1]*1000:+.1f} z={gp2[2]*1000:+.1f}) [{tag2}]")
                except Exception as exc:
                    print(f"      (pre-settle reconstruction failed: {type(exc).__name__}: {exc})")
        env.close()


if __name__ == "__main__":
    sys.exit(main())
