"""Unit tests for the PR #24 follow-up clearance fixes (FV panel CRITICALs).

Covered:
* Fix B (FV SMT G): the robot<->distractor clearance must emit the distractor's
  world settle z (~0.92), not the Scenic SAFE_REGION TABLE_Z (~0.82).
* Fix C (FV MC #6): clearance footprint half-extents must be the max over the
  object's substitution pool, not the canonical class only.
* Fix D (FV MC #3): variant choosers + surface z must be keyed per object
  INSTANCE, not per class.

These render Scenic source text only (no MuJoCo), so they are fast and run in
the `pytest -k 'clearance or scenic or placement'` slice.
"""
from __future__ import annotations

import re

import pytest

from libero_infinity.asset_metadata import TABLE_SURFACE_Z, surface_spawn_z
from libero_infinity.compiler import (
    build_semantic_scene_graph,
    plan_perturbations,
    render_scenic,
)
# Pull node types + helpers from the renderer namespace so the test resolves the
# exact symbols the renderer uses, regardless of their defining module.
from libero_infinity.renderer.scenic_renderer import (  # type: ignore
    MovableSupportNode,
    ObjectNode,
    _sanitize,
    _to_var,
    get_dimensions,
)
from libero_infinity.task_config import TaskConfig
from libero_infinity.validation.sweep import resolve_task_path

_TASK = "libero_goal/put_the_bowl_on_the_plate.bddl"


def _render(request: str):
    cfg = TaskConfig.from_bddl(str(resolve_task_path(_TASK)))
    graph = build_semantic_scene_graph(cfg)
    plan = plan_perturbations(graph, request)
    return plan, graph, render_scenic(plan, graph)


def _poolmax_dims(plan, graph):
    """instance var name -> elementwise max footprint over its substitution pool."""
    out: dict[str, tuple[float, float, float]] = {}
    obj_axis = "object" in plan.active_axes
    for _nid, node in graph.nodes.items():
        if not isinstance(node, (ObjectNode, MovableSupportNode)):
            continue
        if isinstance(node, ObjectNode) and getattr(node, "contained", False):
            continue
        obj_class = node.object_class or node.instance_name
        pool = [obj_class]
        if obj_axis:
            pool += list(plan.object_substitutions.get(node.instance_name) or [])
        dims = tuple(max(get_dimensions(c)[k] for c in pool) for k in range(3))
        out[_to_var(node.instance_name)] = dims
    return out


def test_clearance_distractor_world_z():
    """Fix B (FV SMT G): no Scenic-frame distractor z; world settle z emitted."""
    plan, graph, text = _render("position,distractor,robot")
    # The Scenic SAFE_REGION z must NOT leak into any clearance clause.
    assert "distractor_0.position.z" not in text
    dz = surface_spawn_z(TABLE_SURFACE_Z, "distractor", None)
    assert 0.90 <= dz <= 0.99, dz
    # If the robot-clearance clauses were emitted (robot footprint present), the
    # distractor target must carry the world settle z as a constant.
    if "_robot_dq" in text and "distractor_0" in text:
        assert f"{dz:.4f}" in text


def test_clearance_variant_footprint_max_over_pool():
    """Fix C (FV MC #6): object<->object thresholds use the max-over-pool footprint."""
    plan, graph, text = _render("position,object")
    dims = _poolmax_dims(plan, graph)
    pat = re.compile(
        r"abs\((\w+)\.position\.x - (\w+)\.position\.x\) > ([\d.]+)\) "
        r"or \(abs\(\1\.position\.y - \2\.position\.y\) > ([\d.]+)\)"
    )
    checked = 0
    for a, b, nx, ny in pat.findall(text):
        if a not in dims or b not in dims:
            continue  # fixture / distractor pair — different footprint source
        exp_x = (dims[a][0] + dims[b][0]) / 2.0
        exp_y = (dims[a][1] + dims[b][1]) / 2.0
        assert abs(float(nx) - exp_x) < 1e-3, (a, b, nx, exp_x)
        assert abs(float(ny) - exp_y) < 1e-3, (a, b, ny, exp_y)
        checked += 1
    assert checked >= 1, "no object<->object clearance clause found to validate"


def test_scenic_per_instance_chooser():
    """Fix D (FV MC #3): one variant chooser per object INSTANCE, not per class."""
    plan, graph, text = _render("position,object")
    subs = plan.object_substitutions or {}
    if not subs:
        pytest.skip("task exposes no object-axis substitutions")
    # Resolve each substituted obj_name to its instance name.
    insts = []
    for obj_name in subs:
        node = graph.get_node(obj_name)
        if node is not None:
            insts.append(node.instance_name)
    # Every substituted instance gets its own instance-keyed chooser.
    for inst in insts:
        assert f"_chosen_{_sanitize(inst)} = Uniform(" in text, inst
    # Exactly one chooser per distinct instance (no per-class collapse).
    n_choosers = len(re.findall(r"^_chosen_\w+ = Uniform\(", text, re.MULTILINE))
    assert n_choosers == len(set(insts)), (n_choosers, insts)
