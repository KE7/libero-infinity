"""Settle-measure the distractor contact rest height on a fixture top.

Builds a distractor scene for TASK, resets (inject + settle), and for every
distractor whose support_surface_class == FIXTURE reads:
  injected z (scenic_position), settled z (env_position), the distractor's own
  on-table body-origin clearance, and derives
      settle_top_z = settled_z - on_table_clearance - TABLE_Z
the distractor-independent contact surface height above the table.

Usage (isolated subprocess):
    PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/_diag_settle_topz.py \
        <task_rel> <fixture_class> [--seeds 6]
Emits machine-readable lines: SETTLE <fixture> <seed> <dist_name> <class>
<injected_z> <settled_z> <on_table> <settle_top_z>
"""
from __future__ import annotations

import argparse
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("fixture")
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    from libero_infinity.asset_metadata import TABLE_SURFACE_Z, spawn_clearance
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.simulator import TABLE_Z
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

    bddl = str(resolve_task_path(args.task))
    for seed in range(args.seeds):
        try:
            cfg = TaskConfig.from_bddl(bddl)
            random.seed(seed)
            scn = compile_task_to_scenario(cfg, "distractor")
            scene, _ = scn.generate(maxIterations=4000)
            env = make_env(scene, bddl_path=bddl)
            env.reset()
        except Exception as exc:  # noqa: BLE001
            print(f"# build-fail seed={seed}: {type(exc).__name__}: {exc}")
            continue
        eval_scene = getattr(env, "realized_scene", None) or scene
        for o in _iter_scene_objects(eval_scene):
            if is_scene_fixture(o):
                continue
            nm = resolve_object_name(o) or "?"
            if not nm.startswith("distractor"):
                continue
            ssc = getattr(o, "support_surface_class", "") or ""
            if ssc != args.fixture:
                continue
            cls = getattr(o, "asset_class", "") or "distractor"
            try:
                st = _env_get_object(env, nm)
                res = assert_pose_tolerance(o, st)
            except Exception:  # noqa: BLE001
                continue
            p = res.payload
            sp = p.get("scenic_position")
            ep = p.get("env_position")
            if sp is None or ep is None:
                continue
            inj_z = float(sp[2])
            set_z = float(ep[2])
            on_table = spawn_clearance(cls, None)
            settle_top_z = set_z - on_table - TABLE_Z
            print(
                f"SETTLE {args.fixture} {seed} {nm} {cls} "
                f"{inj_z:.5f} {set_z:.5f} {on_table:.5f} {settle_top_z:.5f} "
                f"passed={res.passed} TABLE_Z={TABLE_Z} TABLE_SURFACE_Z={TABLE_SURFACE_Z}"
            )
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
