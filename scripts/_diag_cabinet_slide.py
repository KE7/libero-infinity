"""Reproduce the 285-299mm distractor slides in put_the_bowl_on_top_of_the_cabinet.
Dump every distractor (active+inactive) with injected vs settled x/y/z, assigned
support, and nearest fixture, for the exact v8-failing subsets/seeds."""
from __future__ import annotations
import random
import numpy as np

CONDS = [
    ("libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl", ("robot", "distractor")),
    ("libero_goal/put_the_bowl_on_top_of_the_cabinet.bddl",
     ("position", "object", "robot", "camera", "lighting", "texture", "distractor", "background")),
]
SEEDS = [0, 1, 2, 3, 4]


def main():
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    for task_rel, subset in CONDS:
        bddl = str(resolve_task_path(task_rel))
        for seed in SEEDS:
            try:
                cfg = TaskConfig.from_bddl(bddl)
                random.seed(seed)
                scn = compile_task_to_scenario(cfg, ",".join(subset))
                scene, _ = scn.generate(maxIterations=4000)
                env = make_env(scene, bddl_path=bddl)
                env.reset()
            except Exception as exc:
                print(f"# build-fail {subset[:1]} s{seed}: {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                continue
            sim = env._sim.libero_env.env.sim
            active = getattr(env._sim, "_active_distractor_names", set())
            ndist = None
            try:
                ndist = scene.params.get("n_distractors")
            except Exception:
                pass
            # fixture positions
            fix = {}
            for o in scene.objects:
                fn = getattr(o, "libero_name", "")
                oc = getattr(o, "object_class", "") or getattr(o, "asset_class", "")
            print(f"\n=== {subset[0]} s{seed} n_distractors={ndist} active={sorted(active)} ===", flush=True)
            for o in scene.objects:
                nm = getattr(o, "libero_name", "")
                if not nm.startswith("distractor_"):
                    continue
                cls = getattr(o, "asset_class", "") or ""
                sc = getattr(o, "support_surface_class", "") or ""
                fi = getattr(o, "support_parent_name", "") or ""
                surf = f"{sc}({fi})" if sc and fi else "TABLE"
                act = "ACTIVE" if nm in active else "inactive"
                try:
                    inj = np.array(o.position, dtype=float)
                except Exception:
                    inj = None
                bid = None
                for c in (nm, nm + "_main"):
                    try:
                        bid = sim.model.body_name2id(c); break
                    except Exception:
                        pass
                if bid is None or inj is None:
                    print(f"  {nm:12} {cls:18} {surf:26} {act} inj=({inj[0]:.3f},{inj[1]:.3f},{inj[2]:.3f}) [no body]")
                    continue
                bp = sim.data.body_xpos[bid]
                dx, dy, dz = (bp[0]-inj[0])*1000, (bp[1]-inj[1])*1000, (bp[2]-inj[2])*1000
                xy = float(np.hypot(dx, dy))
                flag = "  <== SLIDE" if xy > 30 else ""
                print(f"  {nm:12} {cls:18} {surf:26} {act} inj=({inj[0]:+.3f},{inj[1]:+.3f},{inj[2]:.3f}) "
                      f"settle=({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:.3f}) dx={dx:+.0f} dy={dy:+.0f} dz={dz:+.0f} xy={xy:.0f}{flag}", flush=True)
            env.close()


if __name__ == "__main__":
    main()
