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
    """Fix B (FV SMT G) + Fix 2: distractors carry a world-frame settle z.

    Under Fix 2 each distractor is placed ``at Vector(Range, Range,
    resolved_spawn_z)`` (NOT ``in SAFE_REGION``), so ``distractor_i.position.z``
    IS the world settle z (~0.92 on the table, higher on a fixture) — the SAME
    value the simulator settles to. The robot↔distractor clearance therefore
    references ``distractor_i.position.z`` directly, which is now correct (it is
    no longer the Scenic SAFE_REGION TABLE_Z ~0.82). This pins that the
    distractor z is a world-frame value and the bare table-surface z never
    appears as a distractor's emitted spawn z.
    """
    plan, graph, text = _render("position,distractor,robot")
    dz = surface_spawn_z(TABLE_SURFACE_Z, "distractor", None)
    assert 0.90 <= dz <= 0.99, dz
    # The table-assigned distractor (slot 0) draws its z from a correlated
    # (class, world_spawn_z) Uniform — every emitted z must be world-frame
    # (> TABLE_SURFACE_Z), never the bare SAFE_REGION TABLE_SURFACE_Z.
    m = re.search(r"_distractor_0_choice = Uniform\((.*)\)", text)
    assert m, "distractor_0 must use a correlated (class, z) chooser"
    zs = [float(z) for z in re.findall(r'"[\w]+",\s*([\d.]+)\)', m.group(1))]
    assert zs, m.group(1)
    assert all(z > TABLE_SURFACE_Z + 0.02 for z in zs), zs
    assert f"{TABLE_SURFACE_Z:.4f}" not in m.group(1)
    # If the robot-clearance clauses were emitted, the distractor target now
    # carries its world-frame position.z directly (Fix 2 lockstep).
    if "_robot_dq" in text and "distractor_0" in text:
        assert "distractor_0.position.z" in text


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
