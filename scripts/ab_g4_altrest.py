"""OLD-vs-NEW A/B for the g4 pose_tolerance alt-rest scoring change.

For a representative corpus slice (tasks x subsets x seeds, all arenas) it scores
every task object under BOTH gates via the SAME single reset:

  OLD = assert_pose_tolerance(..., accept_alt_rest=False)   # strict 5mm/1deg
  NEW = assert_pose_tolerance(..., accept_alt_rest=True)    # strict OR alt-rest

and tallies, per (arena, class):
  * FAIL->PASS   (the closed tail — what NEW newly admits)
  * PASS->FAIL   (MUST be 0 — proves NET-ADD)
and per-reject-reason histogram for objects NEW still fails (audit the gate).

It ALSO runs an injected-bad-case check: for each real settled object it fabricates
(a) a fall-through, (b) an out-of-region slide, (c) a tipped pose, (d) a moving
(non-converged) state, and asserts NEW REJECTS every one — proving the alt-rest
path is not a blanket mask.

Usage:
  ab_g4_altrest.py --tasks-file f.txt --subsets position,object --seeds 4 --json-out /tmp/ab.jsonl
"""

import argparse
import collections
import json
import random

import numpy as np

ARENA_TASKS = {
    "living_room": [
        "libero_90/LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.bddl",
        "libero_90/LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket.bddl",
        "libero_90/LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl",
    ],
    "kitchen": [
        "libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl",
        "libero_90/KITCHEN_SCENE3_turn_on_the_stove.bddl",
        "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
    ],
    "table": [
        "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
        "libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl",
        "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    ],
    "study": [
        "libero_90/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
    ],
}


def _extents(o):
    try:
        return float(o.width), float(o.length), float(o.height)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="position,object")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tasks-file", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--arenas", default="living_room,kitchen,table,study")
    args = ap.parse_args()

    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import (
        is_scene_fixture,
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.consistency import (
        assert_pose_tolerance,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path

    arenas = [a for a in args.arenas.split(",") if a.strip()]
    task_arena = []
    if args.tasks_file:
        with open(args.tasks_file) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    task_arena.append(("custom", ln))
    else:
        for a in arenas:
            for t in ARENA_TASKS.get(a, []):
                task_arena.append((a, t))

    subsets = [tok.replace("+", ",") for tok in args.subsets.split(",")]

    flip_fp = collections.Counter()  # (arena,cls) -> FAIL->PASS
    flip_pf = collections.Counter()  # (arena,cls) -> PASS->FAIL (must be 0)
    old_pass = collections.Counter()
    new_pass = collections.Counter()
    total = collections.Counter()
    reject_reasons = collections.Counter()
    inj = collections.Counter()  # injected-bad -> {rejected, wrongly_accepted}
    rows = []
    n_reset_fail = 0

    for arena, task in task_arena:
        try:
            path = resolve_task_path(task)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP resolve {task}: {exc}")
            continue
        for sub in subsets:
            for seed in range(args.seeds):
                random.seed(seed)
                np.random.seed(seed)
                try:
                    cfg = TaskConfig.from_bddl(str(path))
                    scn = compile_task_to_scenario(cfg, sub)
                    scene, _ = scn.generate(maxIterations=8000)
                    env = make_env(scene, bddl_path=str(path))
                    env.reset()
                except Exception:  # noqa: BLE001
                    n_reset_fail += 1
                    continue
                try:
                    es = getattr(env, "realized_scene", None) or scene
                    for o in _iter_scene_objects(es):
                        if is_scene_fixture(o):
                            continue
                        nm = resolve_object_name(o) or "?"
                        cls = getattr(o, "asset_class", "?")
                        try:
                            st = env.get_object_state(nm)
                        except Exception:  # noqa: BLE001
                            continue
                        if st is None:
                            continue
                        r_old = assert_pose_tolerance(o, st, accept_alt_rest=False)
                        r_new = assert_pose_tolerance(o, st, accept_alt_rest=True)
                        key = (arena, cls)
                        total[key] += 1
                        old_pass[key] += int(bool(r_old.passed))
                        new_pass[key] += int(bool(r_new.passed))
                        if (not r_old.passed) and r_new.passed:
                            flip_fp[key] += 1
                            rows.append(
                                {
                                    "flip": "FAIL->PASS",
                                    "arena": arena,
                                    "class": cls,
                                    "task": task,
                                    "sub": sub,
                                    "seed": seed,
                                    "pos_err": r_new.payload.get("position_error"),
                                    "rot_err": r_new.payload.get("rotation_error_deg"),
                                    "alt_info": r_new.payload.get("alt_rest_info"),
                                }
                            )
                        if r_old.passed and (not r_new.passed):
                            flip_pf[key] += 1
                            rows.append(
                                {
                                    "flip": "PASS->FAIL",
                                    "arena": arena,
                                    "class": cls,
                                    "task": task,
                                    "sub": sub,
                                    "seed": seed,
                                }
                            )
                        if not r_new.passed:
                            reject_reasons[r_new.payload.get("alt_rest_reject_reason")] += 1

                        # -------- injected-bad-case check (NEW must reject) --------
                        ext = _extents(o)
                        if ext is not None and st.get("position") is not None:
                            _injected_bad_checks(o, st, ext, assert_pose_tolerance, inj)
                finally:
                    env.close()

    print(f"\nreset failures: {n_reset_fail}")
    print("\n=== per (arena, class): OLD -> NEW pass, flips ===")
    grand = {"old": 0, "new": 0, "tot": 0, "fp": 0, "pf": 0}
    for key in sorted(total):
        arena, cls = key
        o, n, t = old_pass[key], new_pass[key], total[key]
        fp, pf = flip_fp[key], flip_pf[key]
        grand["old"] += o
        grand["new"] += n
        grand["tot"] += t
        grand["fp"] += fp
        grand["pf"] += pf
        flag = "  <== PASS->FAIL!!" if pf else ""
        if fp or pf:
            print(
                f"  {arena:12s} {cls:20s} old={o:3d}/{t:<3d} new={n:3d}/{t:<3d} "
                f"FAIL->PASS={fp:2d} PASS->FAIL={pf}{flag}"
            )
    g = grand
    print(
        f"\nTOTAL objects={g['tot']}  OLD pass={g['old']} ({100*g['old']/max(g['tot'],1):.2f}%)  "
        f"NEW pass={g['new']} ({100*g['new']/max(g['tot'],1):.2f}%)"
    )
    print(f"  FAIL->PASS (closed tail) = {g['fp']}")
    print(f"  PASS->FAIL (must be 0)   = {g['pf']}")
    print("\nNEW-still-fail reject-reason histogram:")
    for reason, c in reject_reasons.most_common():
        print(f"  {str(reason):20s} {c}")
    print("\nInjected-bad-case check (NEW must REJECT all):")
    for k in sorted(inj):
        print(f"  {k:28s} {inj[k]}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(rows)} flip rows -> {args.json_out}")


def _injected_bad_checks(o, st, ext, assert_pose_tolerance, inj):
    """Fabricate physically-invalid settles from the object's EMITTED pose; NEW
    must reject every one. Each fabrication perturbs ONE axis into invalidity
    while keeping the others clean, so the alt-rest path is what must reject it.
    """
    import math

    from libero_infinity.validation.invariants.domain import _obj_position

    width, length, height = ext
    ep = _obj_position(o)  # emitted (scenic) pose — the reference
    if ep is None:
        return
    # Upright reference = the object's real canonical orientation from the env
    # (falls back to identity). The alt-rest path measures upright as
    # env-settled-vs-canonical, so non-tipped fabrications set env orientation =
    # canonical (upright_err 0) to isolate the axis under test.
    canon = st.get("canonical_orientation") or (1.0, 0.0, 0.0, 0.0)
    good_conv = {"settle_conv_lin": 1e-4, "settle_conv_ang": 1e-2}

    def state(pos, ori=canon, **extra):
        s = {
            "position": pos,
            "orientation": ori,
            "class": st.get("class"),
            "canonical_orientation": canon,
        }
        s.update(good_conv)
        s.update(extra)
        return s

    # A tiny in-region slide (>5mm so strict fails ⇒ alt is what must decide),
    # small enough to stay within the footprint bound.
    in_region = min(0.4 * max(width, length), 0.02)

    # (a) fall-through: env z 2x height below emitted (everything else clean).
    p = (ep[0], ep[1], ep[2] - 2.0 * max(height, 0.02))
    ra = assert_pose_tolerance(o, state(p), accept_alt_rest=True)
    inj["fall_through_rejected" if not ra.passed else "fall_through_WRONGLY_ACCEPTED"] += 1

    # (b) out-of-region: slide 3x footprint + 10cm horizontally.
    slide = 3.0 * max(width, length) + 0.10
    p = (ep[0] + slide, ep[1], ep[2])
    rb = assert_pose_tolerance(o, state(p), accept_alt_rest=True)
    inj["out_of_region_rejected" if not rb.passed else "out_of_region_WRONGLY_ACCEPTED"] += 1

    # (c) tipped: env orientation 30deg from canonical, at an in-region slide
    # (so strict fails on position and only the alt upright check can reject).
    half = math.radians(30.0) / 2.0
    tip = (math.cos(half), math.sin(half), 0.0, 0.0)  # 30° about x vs identity
    p = (ep[0] + in_region, ep[1], ep[2])
    rc = assert_pose_tolerance(
        o, state(p, ori=tip, canonical_orientation=(1.0, 0.0, 0.0, 0.0)), accept_alt_rest=True
    )
    inj["tipped_rejected" if not rc.passed else "tipped_WRONGLY_ACCEPTED"] += 1

    # (d) not-converged: a normally-valid in-region slide BUT large settle drift —
    # proves the convergence gate is what rejects (control (d') below accepts it).
    p = (ep[0] + in_region, ep[1], ep[2])
    rd = assert_pose_tolerance(
        o, state(p, settle_conv_lin=0.5, settle_conv_ang=5.0), accept_alt_rest=True
    )
    inj["not_converged_rejected" if not rd.passed else "not_converged_WRONGLY_ACCEPTED"] += 1
    # (d') control: SAME in-region slide, converged, upright ⇒ should be ACCEPTED
    # (proves the alt-rest path admits genuine converged rests, not a blanket reject).
    re = assert_pose_tolerance(o, state(p), accept_alt_rest=True)
    inj["control_valid_altrest_accepted" if re.passed else "control_valid_altrest_REJECTED"] += 1


if __name__ == "__main__":
    main()
