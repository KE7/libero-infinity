"""verifai_position.scenic — VerifAI cross-entropy adversarial search.

Uses VerifaiRange instead of Range so that Scenic's cross-entropy sampler
can concentrate on failure-inducing regions across iterations.

When used without VerifAI, VerifaiRange falls back to uniform sampling
(identical to Range). With VerifAI installed and feedback provided via
scenario.generate(feedback=rho), the CE sampler narrows the distribution
toward high-failure regions.

Usage::

    scenario = scenic.scenarioFromFile(
        "scenic/verifai_position.scenic",
        params={"bddl_path": "...", "verifaiSamplerType": "ce"},
    )
    for i in range(n_samples):
        scene, _ = scenario.generate(feedback=last_rho)
        ... run episode ...
        last_rho = 0.0 if success else 1.0
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# VerifAI sampler configuration
# ──────────────────────────────────────────────────────────────────────────────
param verifaiSamplerType = "ce"    # cross-entropy sampler

param task = "adversarial_position_search"
param bddl_path = ""
param min_clearance = 0.12
param ood_margin = 0.15

# ──────────────────────────────────────────────────────────────────────────────
# Use VerifaiRange for CE-concentrating sampling (falls back to uniform)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from verifai.scenic_interop import VerifaiRange
except ImportError:
    VerifaiRange = Range

_min_clearance = globalParameters.min_clearance
_ood_margin = globalParameters.ood_margin

bowl = new LIBEROObject with libero_name "akita_black_bowl_1",
                         with asset_class "akita_black_bowl",
                         with width 0.10, with length 0.10, with height 0.06,
                         at Vector(VerifaiRange(TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05),
                                   VerifaiRange(TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05),
                                   TABLE_Z)

plate = new LIBEROObject with libero_name "plate_1",
                          with asset_class "plate",
                          with width 0.20, with length 0.20, with height 0.02,
                          at Vector(VerifaiRange(TABLE_X_MIN + 0.08, TABLE_X_MAX - 0.08),
                                    VerifaiRange(TABLE_Y_MIN + 0.08, TABLE_Y_MAX - 0.08),
                                    TABLE_Z)

ramekin = new LIBEROObject with libero_name "glazed_rim_porcelain_ramekin_1",
                            with asset_class "glazed_rim_porcelain_ramekin",
                            with width 0.08, with length 0.08, with height 0.05,
                            at Vector(VerifaiRange(TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05),
                                      VerifaiRange(TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05),
                                      TABLE_Z)

# Pairwise clearance
require (distance from bowl to plate)    > _min_clearance
require (distance from bowl to ramekin)  > _min_clearance
require (distance from plate to ramekin) > _min_clearance

# Workspace bounds are already enforced by the VerifaiRange limits above.
