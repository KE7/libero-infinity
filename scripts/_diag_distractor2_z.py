"""Pin down distractor_2's z-penetration: its asset class, assigned support,
the renderer's spawn z (surface_spawn_z) vs the true settled rest height.
Decides whether the residual is a missing measured (class,fixture) spawn-z row.
"""
from __future__ import annotations

import random

from libero_infinity.asset_metadata import TABLE_SURFACE_Z, surface_spawn_z
from libero_infinity.ir.graph_builder import build_semantic_scene_graph
from libero_infinity.planner.composition import plan_perturbations
from libero_infinity.renderer.scenic_renderer import _distractor_slots
from libero_infinity.task_config import TaskConfig
from libero_infinity.validation.sweep import resolve_task_path

TASK = "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
SUBSET = "robot,distractor"
SEED = 2

bddl = str(resolve_task_path(TASK))
cfg = TaskConfig.from_bddl(bddl)
random.seed(SEED)
graph = build_semantic_scene_graph(cfg)
plan = plan_perturbations(graph, SUBSET)
print("distractor_budget:", plan.distractor_budget)
print("distractor_classes:", plan.distractor_classes)
print("TABLE_SURFACE_Z:", TABLE_SURFACE_Z)
slots = _distractor_slots(plan, graph)
for s in slots:
    print(f"\nslot {s.index}: surface_class={s.surface_class} fixture={s.fixture_name} "
          f"z_lo={s.z_lo:.4f} z_hi={s.z_hi:.4f}")
    for c in (plan.distractor_classes or []):
        z = surface_spawn_z(TABLE_SURFACE_Z, c, s.surface_class)
        print(f"    surface_spawn_z({c!r}, {s.surface_class!r}) = {z:.4f}")
