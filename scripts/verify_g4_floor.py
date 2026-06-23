"""Before/after verification for the G4 pose_tolerance floor-arena fix.

For the 10 libero_object basket tasks across position/object/combined subsets and
several seeds, build the real LIBERO env, reset+settle, and score
``assert_pose_tolerance`` on the TASK objects. Reports pass-rate, max xy error
(must stay <5mm), and max |dz|. Also reports Scenic sample success (g3 proxy)."""
import argparse, random, collections, json, math
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="position,object,position+object+camera+lighting+texture")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.invariants._scene_view import is_scene_fixture, resolve_object_name
    from libero_infinity.validation.invariants.consistency import _env_get_object, assert_pose_tolerance
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.sweep import resolve_task_path, discover_all_tasks

    TASKS = sorted(t for t in discover_all_tasks() if t.startswith("libero_object/") and "basket" in t)
    subsets = [s.replace("+", ",") for s in args.subsets.split(",")] if "+" not in args.subsets else [s.replace("+", ",") for s in args.subsets.split(",")]
    # robust subset parse: split top-level by comma but keep '+'-joined combos
    subsets = []
    for tok in args.subsets.split(","):
        subsets.append(tok.replace("+", ","))
    npass = nfail = 0
    xy_max = dz_max = 0.0
    g3_fail = 0
    fails = collections.Counter()
    for t in TASKS:
        bddl = str(resolve_task_path(t))
        for sub in subsets:
            for seed in range(args.seeds):
                try:
                    cfg = TaskConfig.from_bddl(bddl)
                    random.seed(seed); np.random.seed(seed)
                    scn = compile_task_to_scenario(cfg, sub)
                    scene, _ = scn.generate(maxIterations=8000)
                except Exception:
                    g3_fail += 1; continue
                try:
                    env = make_env(scene, bddl_path=bddl); env.reset()
                except Exception:
                    g3_fail += 1; continue
                es = getattr(env, "realized_scene", None) or scene
                for o in _iter_scene_objects(es):
                    if is_scene_fixture(o): continue
                    nm = resolve_object_name(o) or "?"
                    try: st = _env_get_object(env, nm)
                    except Exception: continue
                    res = assert_pose_tolerance(o, st); p = res.payload
                    sp, ep = p.get("scenic_position"), p.get("env_position")
                    if sp and ep:
                        xy = math.hypot(sp[0]-ep[0], sp[1]-ep[1]); xy_max = max(xy_max, xy)
                        dz_max = max(dz_max, abs(sp[2]-ep[2]))
                    if res.passed: npass += 1
                    else:
                        nfail += 1
                        fails[f"{getattr(o,'asset_class','?')}"] += 1
                env.close()
    tot = npass + nfail
    print(f"\n=== verify_g4_floor [{args.tag}] ===")
    print(f"task-object pose_tolerance: {npass}/{tot} = {100*npass/tot:.2f}% pass" if tot else "no rows")
    print(f"max xy err: {xy_max*1000:.2f}mm  (must stay <5mm)")
    print(f"max |dz|:   {dz_max*1000:.2f}mm")
    print(f"g3/build fail conditions: {g3_fail}")
    if fails: print("fails by class:", dict(fails.most_common()))
    print("JSON " + json.dumps({"pass":npass,"fail":nfail,"xy_max_mm":xy_max*1000,"dz_max_mm":dz_max*1000,"g3_fail":g3_fail}))

if __name__ == "__main__":
    main()
