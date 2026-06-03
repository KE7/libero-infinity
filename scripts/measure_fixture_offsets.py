"""Measure each fixture class's geom-AABB CENTER OFFSET relative to the body
position the renderer uses in clearance constraints (the graph FixtureNode
init_x/init_y == the value emitted as `<fixture>.position`).

The distractor↔fixture / object↔fixture / robot↔fixture clearance assumes the
fixture geometry is centered on its body origin. For irregular fixtures
(flat_stove) the real collision geometry is offset ~100 mm, so the guarded box
misses part of the real fixture and a table distractor is injected PENETRATING it
-> chaotic settle launch (PR#24 285-339 mm gross fail, RCA robot_distractor_settle).

Static measurement: fixtures are immovable, so geom positions after reset are
stable. No distractor live-stepping (avoids the MuJoCo contact-arena overflow).

Usage: MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python scripts/measure_fixture_offsets.py
"""
from __future__ import annotations
import json
import random
import statistics
import numpy as np

# reuse the measure-task corpus + AABB helpers from the main generator
import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "_msc", pathlib.Path(__file__).with_name("measure_spawn_clearances.py")
)
_msc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_msc)  # type: ignore


def main():
    from libero_infinity.ir.graph_builder import build_semantic_scene_graph
    from libero_infinity.gym_env import make_env
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import discover_all_tasks, resolve_task_path
    from libero_infinity.compiler import compile_task_to_scenario
    from libero_infinity.ir.nodes import FixtureNode

    avail = set(discover_all_tasks())
    offsets: dict[str, list[tuple[float, float]]] = {}
    foots: dict[str, list[tuple[float, float, float]]] = {}

    for task_rel in _msc.MEASURE_TASKS:
        if task_rel not in avail:
            continue
        bddl = str(resolve_task_path(task_rel))
        try:
            cfg = TaskConfig.from_bddl(bddl)
            graph = build_semantic_scene_graph(cfg)
        except Exception as exc:
            print(f"# graph fail {task_rel}: {exc}")
            continue
        # fixture instance -> (class, init_x, init_y) from the graph (the
        # reference the renderer emits as <fixture>.position)
        finfo = {}
        for node in graph.nodes.values():
            cls = getattr(node, "object_class", None)
            ix = getattr(node, "init_x", None)
            iy = getattr(node, "init_y", None)
            inst = getattr(node, "instance_name", None)
            # FixtureNode duck-type: has init_x/init_y and is a fixture
            if inst and ix is not None and iy is not None and isinstance(node, FixtureNode):
                finfo[inst] = (cls, float(ix), float(iy))
        if not finfo:
            continue
        try:
            random.seed(0)
            scenario = compile_task_to_scenario(cfg, "distractor")
            scene, _ = scenario.generate(maxIterations=8000)
            env = make_env(scene, bddl_path=bddl)
            env.reset()
        except Exception as exc:
            print(f"# build fail {task_rel}: {exc}")
            continue
        sim = env._sim.libero_env.env.sim
        for inst, (cls, ix, iy) in finfo.items():
            if not cls:
                continue
            faabb = _msc._fixture_world_aabb(sim, inst)
            if faabb is None:
                continue
            cx = (faabb[0] + faabb[1]) / 2.0
            cy = (faabb[2] + faabb[3]) / 2.0
            offsets.setdefault(cls, []).append((cx - ix, cy - iy))
            foots.setdefault(cls, []).append(
                (faabb[1] - faabb[0], faabb[3] - faabb[2], faabb[5] - faabb[4])
            )
            print(f"{task_rel.split('/')[-1][:38]:38} {cls:16} init=({ix:+.4f},{iy:+.4f}) "
                  f"aabb_ctr=({cx:+.4f},{cy:+.4f}) off=({cx-ix:+.4f},{cy-iy:+.4f}) "
                  f"foot=({faabb[1]-faabb[0]:.4f},{faabb[3]-faabb[2]:.4f})", flush=True)
        env.close()

    print("\n=== PER-CLASS MEDIAN OFFSET + FOOTPRINT ===")
    result = {}
    for cls in sorted(offsets):
        ox = statistics.median(o[0] for o in offsets[cls])
        oy = statistics.median(o[1] for o in offsets[cls])
        fw = statistics.median(f[0] for f in foots[cls])
        fl = statistics.median(f[1] for f in foots[cls])
        result[cls] = {"offset": [round(ox, 5), round(oy, 5)], "footprint": [round(fw, 5), round(fl, 5)]}
        print(f"  {cls:18} offset=({ox:+.4f},{oy:+.4f}) footprint=({fw:.4f},{fl:.4f}) "
              f"n={len(offsets[cls])}")
    print("\nJSON:\n" + json.dumps(result, indent=2))

    # --write: merge ONLY the offset field into existing fixture entries, where the
    # freshly measured footprint AGREES with the stored (validated) footprint
    # (≤10 mm) — guaranteeing offset and footprint describe the SAME geom config.
    # Never touches footprint/height/top_z. Add-missing semantics for offset only.
    if "--write" in __import__("sys").argv:
        fg_path = pathlib.Path("src/libero_infinity/data/fixture_geometry.json")
        fg = json.loads(fg_path.read_text())
        wrote = []
        for cls, r in result.items():
            ent = fg["fixtures"].get(cls)
            if ent is None:
                continue
            stored_fp = ent.get("footprint")
            if not (isinstance(stored_fp, list) and len(stored_fp) >= 2):
                continue
            if abs(stored_fp[0] - r["footprint"][0]) > 0.01 or abs(stored_fp[1] - r["footprint"][1]) > 0.01:
                print(f"# skip {cls}: measured footprint {r['footprint']} disagrees with stored {stored_fp}")
                continue
            ent["offset"] = r["offset"]
            wrote.append(cls)
        fg_path.write_text(json.dumps(fg, indent=2) + "\n")
        print(f"\n# wrote offset for: {wrote}")


if __name__ == "__main__":
    main()
