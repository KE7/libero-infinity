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

# Install patched MJCFs for fixtures whose vendored XMLs lack <inertial> and a
# flat base geom (desk_caddy, wooden_two_layer_shelf). See asset_overrides.py
# and rca/g5_settle_drift_caddy.md.
try:
    from libero_infinity.asset_overrides import install_mjcf_overrides

    install_mjcf_overrides()
except Exception:
    # Patch is best-effort at import time; failures are non-fatal so that
    # libero_infinity stays importable in environments without LIBERO yet
    # bootstrapped. install_mjcf_overrides() can be called manually later.
    pass

__all__ = [
    "asset_overrides",
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
