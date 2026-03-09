"""LIBERO-Infinity: Scenic 3-based open-ended evaluation for robotic manipulation.

Embeds the Scenic 3 probabilistic programming language into robot evaluation:
point at any LIBERO task and sample infinite statistically-diverse test scenes.

Core capabilities:
  - Position distributions over the full table workspace
  - Object asset distributions sampled at eval time
  - Compositional perturbations via Scenic's scenario system
  - Falsification search via VerifAI integration
"""

import warnings

# Suppress gym 0.25.2 deprecation warning (pinned for robosuite 1.4.0 compatibility).
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

__all__ = [
    "asset_registry",
    "bddl_preprocessor",
    "compiler",
    "eval",
    "gym_env",
    "perturbation_audit",
    "simulator",
    "task_config",
    "task_semantics",
    "task_reverser",
    "vision_validation",
]
