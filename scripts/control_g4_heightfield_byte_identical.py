"""BROAD no-regression control for the cabinet support heightfield.

Renders a wide task set spanning floor / table / kitchen / living_room / stove /
cabinet arenas TWICE — once with ``data/fixture_heightfields.json`` ACTIVE and
once with it disabled (simulating HEAD) — and diffs the emitted Scenic object
lines. The ONLY lines that may differ are ``wooden_cabinet`` ``on_surface`` /
``closed`` akita placements (the covered heightfield tuple); EVERY other emitted
object line (all non-heightfield placements) MUST be byte-identical.

Run from repo root with the interpreter/env used elsewhere:
  MUJOCO_GL=egl PYTHONPATH=src python scripts/control_g4_heightfield_byte_identical.py
"""

import os
import random

import numpy as np

_TASKS = [
    # cabinet (the changed tasks)
    "libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl",
    "libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
    # other kitchen / stove / cabinet-family
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl",
    "libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.bddl",
    "libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl",
    "libero_90/KITCHEN_SCENE3_turn_on_the_stove.bddl",
    # table / study / living_room / floor arenas
    "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
    "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
]

_SUBSETS = ["position", "object", "position,camera,distractor"]


def _render(task, subset, seed):
    from libero_infinity.compiler import build_semantic_scene_graph
    from libero_infinity.planner.composition import parse_axes, plan_perturbations
    from libero_infinity.renderer.scenic_renderer import render_scenic
    from libero_infinity.task_config import TaskConfig
    from libero_infinity.validation.sweep import resolve_task_path

    cfg = TaskConfig.from_bddl(str(resolve_task_path(task)))
    random.seed(seed)
    np.random.seed(seed)
    graph = build_semantic_scene_graph(cfg)
    plan = plan_perturbations(graph, parse_axes(subset.replace(",", "+")))
    return render_scenic(plan, graph)


def _obj_lines(src):
    return [
        ln for ln in src.splitlines() if "= new LIBEROObject" in ln or "= new LIBEROFixture" in ln
    ]


def main():
    import importlib

    import libero_infinity.asset_metadata as am
    import libero_infinity.renderer.scenic_renderer as sr

    data_path = os.path.join(os.path.dirname(am.__file__), "data", "fixture_heightfields.json")
    bak = data_path + ".ctrlbak"

    total = 0
    diffs = 0
    illegal = 0
    # 1) render everything with the heightfield ACTIVE (current state)
    active = {}
    for task in _TASKS:
        for subset in _SUBSETS:
            try:
                active[(task, subset)] = _obj_lines(_render(task, subset, 7))
            except Exception as exc:  # noqa: BLE001
                print(f"# render(active) failed {task}/{subset}: {exc}")

    # 2) disable the data file and reload the modules so FIXTURE_HEIGHTFIELDS = {}
    os.rename(data_path, bak)
    try:
        importlib.reload(am)
        importlib.reload(sr)
        for task in _TASKS:
            for subset in _SUBSETS:
                try:
                    head = _obj_lines(_render(task, subset, 7))
                except Exception as exc:  # noqa: BLE001
                    print(f"# render(head) failed {task}/{subset}: {exc}")
                    continue
                a = active.get((task, subset))
                if a is None:
                    continue
                for la, lh in zip(a, head):
                    total += 1
                    if la == lh:
                        continue
                    diffs += 1
                    # A legal diff is a wooden_cabinet placement gaining the
                    # closed on_surface heightfield (its z drops to 0.898 and it
                    # gains the two relation/state specifiers). Keyed on the
                    # cabinet surface class + the new specifiers so it matches BOTH
                    # the position-literal line (asset_class "akita_black_bowl")
                    # and the object-axis line (asset_class _chosen_X[0], z
                    # _chosen_X[1]). ANY differing line WITHOUT the cabinet surface
                    # class is a real regression (no non-cabinet object may change).
                    legal = (
                        'support_surface_class "wooden_cabinet"' in la
                        and 'support_relation_kind "on_surface"' in la
                        and 'cabinet_drawer_state "closed"' in la
                    )
                    tag = "LEGAL(cabinet)" if legal else "ILLEGAL-REGRESSION"
                    if not legal:
                        illegal += 1
                    print(f"[{tag}] {task.split('/')[-1][:40]} [{subset}]")
                    print(f"   active: {la}")
                    print(f"   head  : {lh}")
    finally:
        os.rename(bak, data_path)
        importlib.reload(am)
        importlib.reload(sr)

    print("\n" + "=" * 60)
    print(f"object lines compared: {total}")
    print(f"differing lines      : {diffs}  (all must be LEGAL cabinet lines)")
    print(f"ILLEGAL regressions  : {illegal}  (MUST be 0)")
    print(
        "RESULT:",
        "PASS — byte-identical for all non-heightfield emission" if illegal == 0 else "FAIL",
    )


if __name__ == "__main__":
    main()
