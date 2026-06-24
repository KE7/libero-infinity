"""Generalized G4 pose_tolerance measurement / verify harness (multi-arena).

Like ``verify_g4_floor.py`` but parameterized by an explicit task list (so it
can target any arena's failing tasks) and emitting per-(arena, class, support)
diagnostics: scenic_z, settled_z, dz, xy, and the per-(arena,class) settled-z
spread (determinism check) needed to decide measured-clearance vs metastable.

Usage:
  verify_g4_arena.py --tasks t1,t2,... --subsets position,object --seeds 3 --tag before
  verify_g4_arena.py --tasks-file f.txt --json-out /tmp/rows.jsonl
"""

import argparse
import collections
import json
import math
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    ap.add_argument("--tasks-file", default="")
    ap.add_argument("--subsets", default="position")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--only-classes", default="", help="comma list; restrict reported objects")
    args = ap.parse_args()

    from libero_infinity.compiler import build_semantic_scene_graph, compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.renderer.scenic_renderer import _workspace_class
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

    tasks = []
    if args.tasks_file:
        with open(args.tasks_file) as f:
            tasks += [ln.strip() for ln in f if ln.strip()]
    if args.tasks:
        tasks += [t for t in args.tasks.split(",") if t.strip()]
    tasks = sorted(set(tasks))
    only = {c for c in args.only_classes.split(",") if c.strip()}
    subsets = [tok.replace("+", ",") for tok in args.subsets.split(",")]

    # arena per task
    arena_of = {}
    for t in tasks:
        try:
            cfg = TaskConfig.from_bddl(str(resolve_task_path(t)))
            arena_of[t] = _workspace_class(build_semantic_scene_graph(cfg)) or "?"
        except Exception:
            arena_of[t] = "?"

    npass = nfail = g3_fail = 0
    xy_max = dz_max = 0.0
    # per (arena,class): pass/fail, settled_z samples, dz samples
    agg = collections.defaultdict(
        lambda: {"pass": 0, "fail": 0, "settled": [], "dz": [], "scenic": []}
    )
    rows_out = []
    for t in tasks:
        bddl = str(resolve_task_path(t))
        arena = arena_of[t]
        for sub in subsets:
            for seed in range(args.seeds):
                try:
                    cfg = TaskConfig.from_bddl(bddl)
                    random.seed(seed)
                    np.random.seed(seed)
                    scn = compile_task_to_scenario(cfg, sub)
                    scene, _ = scn.generate(maxIterations=8000)
                except Exception:
                    g3_fail += 1
                    continue
                try:
                    env = make_env(scene, bddl_path=bddl)
                    env.reset()
                except Exception:
                    g3_fail += 1
                    continue
                es = getattr(env, "realized_scene", None) or scene
                for o in _iter_scene_objects(es):
                    if is_scene_fixture(o):
                        continue
                    nm = resolve_object_name(o) or "?"
                    cls = getattr(o, "asset_class", "?")
                    if only and cls not in only:
                        continue
                    try:
                        st = _env_get_object(env, nm)
                    except Exception:
                        continue
                    res = assert_pose_tolerance(o, st)
                    p = res.payload
                    sp, ep = p.get("scenic_position"), p.get("env_position")
                    key = (arena, cls)
                    if sp and ep:
                        xy = math.hypot(sp[0] - ep[0], sp[1] - ep[1])
                        dz = sp[2] - ep[2]
                        xy_max = max(xy_max, xy)
                        dz_max = max(dz_max, abs(dz))
                        agg[key]["settled"].append(ep[2])
                        agg[key]["scenic"].append(sp[2])
                        agg[key]["dz"].append(dz)
                        rows_out.append(
                            {
                                "task": t,
                                "arena": arena,
                                "class": cls,
                                "name": nm,
                                "subset": sub,
                                "seed": seed,
                                "scenic_z": sp[2],
                                "settled_z": ep[2],
                                "dz_mm": dz * 1000,
                                "xy_mm": xy * 1000,
                                "passed": bool(res.passed),
                            }
                        )
                    if res.passed:
                        npass += 1
                        agg[key]["pass"] += 1
                    else:
                        nfail += 1
                        agg[key]["fail"] += 1
                env.close()

    tot = npass + nfail
    print(f"\n=== verify_g4_arena [{args.tag}] ===")
    print(
        f"task-object pose_tolerance: {npass}/{tot} = {100*npass/tot:.2f}% pass"
        if tot
        else "no rows"
    )
    print(f"max xy err: {xy_max*1000:.2f}mm  (must stay <5mm)")
    print(f"max |dz|:   {dz_max*1000:.2f}mm")
    print(f"g3/build fail conditions: {g3_fail}")
    print("\nper (arena,class): pass/fail | settled_z mean (spread mm) | dz mean/max mm")
    for (arena, cls), v in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1]["fail"])):
        s = v["settled"]
        if not s:
            continue
        spread = (max(s) - min(s)) * 1000
        smean = sum(s) / len(s)
        dz = v["dz"]
        dzmean = sum(dz) / len(dz) * 1000
        dzmax = max(abs(x) for x in dz) * 1000
        flag = "  <-- METASTABLE?" if spread > 5 else ("  <-- FAIL" if v["fail"] else "")
        print(
            f"  [{arena:18s}] {cls:30s} P={v['pass']:4d} F={v['fail']:4d} | "
            f"settled={smean:.4f} spread={spread:6.1f}mm | dz μ={dzmean:7.1f} max={dzmax:7.1f}mm{flag}"
        )
    print(
        "JSON "
        + json.dumps(
            {
                "pass": npass,
                "fail": nfail,
                "xy_max_mm": xy_max * 1000,
                "dz_max_mm": dz_max * 1000,
                "g3_fail": g3_fail,
            }
        )
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows_out)} rows to {args.json_out}")


if __name__ == "__main__":
    main()
