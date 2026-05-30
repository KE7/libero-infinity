"""Diagnose G4 family-C pose_tolerance failures: categorize each movable by
failure mode (z-frame, xy drift, rotation, contained, substituted variant) and
report the position/rotation error so we can separate the systemic z-frame fix
from acknowledged secondary issues (post-settle drift, contained objects,
runtime-sampled object-axis variants).

Usage:
    PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/diagnose_pose_failures.py
"""
from __future__ import annotations

import random

TASKS_SUBSETS = [
    ("libero_goal/put_the_bowl_on_the_stove.bddl", "position"),
    ("libero_goal/put_the_bowl_on_the_stove.bddl", "object"),
    ("libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl", "object"),
    ("libero_90/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.bddl", "position"),
    ("libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl", "object"),
]


def main() -> None:
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.validation.sweep import resolve_task_path
    from libero_infinity.validation.invariants._scene_view import (
        is_scene_fixture,
        resolve_object_name,
    )
    from libero_infinity.validation.invariants.domain import _iter_scene_objects
    from libero_infinity.validation.invariants.consistency import (
        _env_get_object,
        assert_pose_tolerance,
    )

    cat_counts: dict[str, list[int]] = {}

    def bump(cat: str, passed: bool) -> None:
        c = cat_counts.setdefault(cat, [0, 0])
        c[0 if passed else 1] += 1

    for task_rel, subset in TASKS_SUBSETS:
        bddl = str(resolve_task_path(task_rel))
        try:
            cfg = TaskConfig.from_bddl(bddl)
            random.seed(0)
            scn = compile_task_to_scenario(cfg, subset)
            scene, _ = scn.generate(maxIterations=2000)
            env = make_env(scene, bddl_path=bddl)
            env.reset()
        except Exception as exc:  # noqa: BLE001
            print(f"# build failed {task_rel}[{subset}]: {exc}")
            continue
        contained = {
            mo.instance_name for mo in cfg.movable_objects if getattr(mo, "contained", False)
        }
        print(f"\n=== {task_rel.split('/')[-1][:48]} [{subset}] ===")
        print(f"{'name':24} {'scn_cls':18} {'env_cls':18} {'perr_mm':>8} {'rerr_deg':>8} {'pass':>5} {'cat'}")
        for o in _iter_scene_objects(scene):
            if is_scene_fixture(o):
                continue
            nm = resolve_object_name(o) or "?"
            try:
                st = _env_get_object(env, nm)
            except Exception:
                continue
            res = assert_pose_tolerance(o, st)
            p = res.payload
            perr = p.get("position_error")
            rerr = p.get("rotation_error_deg")
            scn_cls = getattr(o, "asset_class", "?")
            env_cls = st.get("class")
            substituted = scn_cls != env_cls
            is_contained = nm in contained or bool(getattr(o, "support_parent_name", "")) and "table" not in str(getattr(o, "support_parent_name", "")).lower()
            if is_contained:
                cat = "contained/supported"
            elif substituted:
                cat = "obj-variant"
            elif perr is not None and perr > 0.05:
                cat = "z-frame(>50mm)"
            elif perr is not None and perr > 0.005 and (rerr is None or rerr <= 1.0):
                cat = "xy-drift(5-50mm)"
            elif rerr is not None and rerr > 1.0:
                cat = "rotation"
            else:
                cat = "ok"
            bump(cat, res.passed)
            pm = f"{perr * 1000:.1f}" if perr is not None else "   -"
            rd = f"{rerr:.2f}" if rerr is not None else "   -"
            print(f"{nm:24} {str(scn_cls):18} {str(env_cls):18} {pm:>8} {rd:>8} {str(res.passed):>5} {cat}")
        env.close()

    print("\n===== FAILURE CATEGORIES (pass/fail) =====")
    for cat, (t, f) in sorted(cat_counts.items()):
        print(f"  {cat:22} {t}T / {f}F")


if __name__ == "__main__":
    main()
