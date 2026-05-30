"""Canonical SMOKE_TASKS x sampled-subsets pose_tolerance smoke, categorized.

Reproduces the RCA's "5 SMOKE_TASKS x 8 subsets" framing (prior 0/97) and
tallies pose_tolerance True/False, split by whether the object axis substituted
a runtime variant (Scenic asset_class != env class) — so the systemic z-frame
fix (non-substituted objects) is reported separately from the object-axis
substitution interaction.

Usage:
    PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/smoke_categorized.py \
        --subsets 8 --out /tmp/smoke_cat.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=int, default=8)
    ap.add_argument("--out", default="/tmp/smoke_cat.jsonl")
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Use non-displacing axis subsets only (exclude robot/distractor, "
        "which physically displace objects post-settle — Finding B). Isolates "
        "the z-frame fix's domain.",
    )
    args = ap.parse_args()

    # Non-displacing axis subsets: position/object/camera/lighting/texture/
    # background/articulation, never robot or distractor.
    CLEAN_SUBSETS = [
        ("position",),
        ("position", "object"),
        ("position", "camera", "lighting"),
        ("object", "texture", "background"),
        ("position", "object", "camera", "lighting", "texture", "background", "articulation"),
        ("position", "articulation"),
    ]

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
    from libero_infinity.validation.sweep import (
        SMOKE_TASKS,
        resolve_task_path,
        sample_subsets,
    )

    subsets = CLEAN_SUBSETS if args.clean else sample_subsets(args.subsets, seed=0)

    tally = {
        "non_subst": [0, 0],   # [True, False] for objects with no variant swap
        "subst": [0, 0],       # object-axis variant swapped
        "contained": [0, 0],   # contained / non-table support
    }
    rows = 0
    n_true = n_false = 0
    with open(args.out, "w") as fh:
        for task in SMOKE_TASKS:
            bddl = str(resolve_task_path(task))
            cfg = TaskConfig.from_bddl(bddl)
            contained = {
                mo.instance_name
                for mo in cfg.movable_objects
                if getattr(mo, "contained", False)
            }
            for subset in subsets:
                try:
                    random.seed(0)
                    scn = compile_task_to_scenario(cfg, ",".join(subset))
                    scene, _ = scn.generate(maxIterations=2000)
                    env = make_env(scene, bddl_path=bddl)
                    env.reset()
                except Exception as exc:  # noqa: BLE001
                    print(f"# build failed {task}[{subset}]: {exc}", flush=True)
                    continue
                row = {"task": task, "subset": list(subset), "objects": []}
                for o in _iter_scene_objects(scene):
                    if is_scene_fixture(o):
                        continue
                    nm = resolve_object_name(o) or "?"
                    try:
                        st = _env_get_object(env, nm)
                    except Exception:
                        continue
                    res = assert_pose_tolerance(o, st)
                    scn_cls = getattr(o, "asset_class", None)
                    env_cls = st.get("class")
                    sp = getattr(o, "support_parent_name", "")
                    is_contained = nm in contained or (
                        bool(sp) and "table" not in str(sp).lower()
                    )
                    substituted = (
                        scn_cls is not None
                        and env_cls is not None
                        and scn_cls != env_cls
                    )
                    cat = (
                        "contained"
                        if is_contained
                        else ("subst" if substituted else "non_subst")
                    )
                    tally[cat][0 if res.passed else 1] += 1
                    if res.passed:
                        n_true += 1
                    else:
                        n_false += 1
                    row["objects"].append(
                        {
                            "name": nm,
                            "scn_cls": scn_cls,
                            "env_cls": env_cls,
                            "cat": cat,
                            "passed": res.passed,
                            "pos_err": res.payload.get("position_error"),
                        }
                    )
                env.close()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows += 1
                print(
                    f"[{rows}] {task.split('/')[-1][:38]:38} {','.join(subset)[:22]:22} "
                    f"running {n_true}T/{n_false}F",
                    flush=True,
                )

    total = n_true + n_false
    print("\n===== CATEGORIZED POSE_TOLERANCE =====")
    print(f"conditions: {rows}   movables: {total}")
    print(f"OVERALL: {n_true}/{total} True ({100 * n_true / max(total, 1):.0f}%)")
    for cat, (t, f) in tally.items():
        tot = t + f
        rate = f"{100 * t / tot:.0f}%" if tot else "n/a"
        print(f"  {cat:12} {t}T / {f}F   ({rate} pass)")
    print(f"\nJSONL: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
