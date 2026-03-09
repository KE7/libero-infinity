"""Scenic renderer for the Libero-Infinity compiler pipeline.

Renders a PerturbationPlan + SemanticSceneGraph into a valid Scenic 3 program.
The renderer is a pure function — zero conditional logic on task semantics.
"""

from libero_infinity.renderer.scenic_renderer import render_scenic

__all__ = ["render_scenic"]
