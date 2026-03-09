"""combined_perturbation.scenic — Layer 3c: joint spatial × identity perturbation.

Composes position_perturbation and object_perturbation into a single scenario
using Scenic 3's modular scenario system, jointly sampling both object positions
and identities in a single pass. Scenic's compose block handles the sequencing
natively.

What varies per sampled scene
──────────────────────────────
  bowl.position      → uniform in TABLE_REGION (spatial perturbation)
  plate.position     → uniform in TABLE_REGION
  ramekin.position   → uniform in TABLE_REGION (if present in task)
  bowl.asset_class   → Uniform over ASSET_VARIANTS["akita_black_bowl"]
  [scene.params]     → chosen_asset, task, bddl_path accessible in Python

Python usage
────────────
    scenario = scenic.scenarioFromFile(
        "scenic/combined_perturbation.scenic",
        params={
            "task": "put_the_bowl_on_the_plate",
            "bddl_path": "...",
            "perturb_class": "akita_black_bowl",
            "min_clearance": 0.12,
        },
    )
    scene, n_iters = scenario.generate(maxIterations=2000)

    chosen_asset = scene.params["chosen_asset"]
    # → e.g. "white_bowl"

    # 1. Patch the BDDL for the chosen asset
    with patched_bddl(bddl_path, {"akita_black_bowl": chosen_asset}) as tmp:
        sim.bddl_path = tmp
        sim.setup()   # injects sampled positions

    # 2. Run policy evaluation on the configured scene

Scenario composition notes (Scenic 3)
──────────────────────────────────────
* A `scenario` block with a `setup:` section runs its body once when instantiated.
* `compose:` blocks orchestrate sub-scenarios sequentially or in parallel.
* `do ScenarioA(); do ScenarioB()` runs them in sequence.
* Shared objects (bowl, plate) defined in setup: are accessible in compose: .
* Objects created in sub-scenarios are merged into the parent scene.

Implementation note: rather than sub-scenario composition (which requires
careful variable scoping), this file inlines both distributions in a single
Main scenario.  This is simpler and avoids Scenic scoping pitfalls when
sharing object references across sub-scenarios.
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# Global parameters
# ──────────────────────────────────────────────────────────────────────────────
param task              = "put_the_bowl_on_the_plate"
param bddl_path         = ""
param perturb_class     = "akita_black_bowl"
param include_canonical = True
param min_clearance     = 0.12
param ood_margin        = 0.15

# Training positions (for anti-canonical soft constraints)
param bowl_train_x   =  0.12
param bowl_train_y   = -0.05
param plate_train_x  =  0.04
param plate_train_y  = -0.02

# ──────────────────────────────────────────────────────────────────────────────
# Asset sampling (object perturbation axis)
# ──────────────────────────────────────────────────────────────────────────────
_all_variants = ASSET_VARIANTS.get(globalParameters.perturb_class,
                                   [globalParameters.perturb_class])
_variants = (_all_variants if globalParameters.include_canonical
             else _all_variants[1:])

chosen_asset = Uniform(*_variants)
param chosen_asset = chosen_asset

# ──────────────────────────────────────────────────────────────────────────────
# Object declarations (position + identity simultaneously perturbed)
# ──────────────────────────────────────────────────────────────────────────────

# Extract param values into locals at compile time — use these in require
# closures to avoid issues with globalParameters access at runtime.
_min_clearance = globalParameters.min_clearance
_ood_margin    = globalParameters.ood_margin
_btx = globalParameters.bowl_train_x
_bty = globalParameters.bowl_train_y
_ptx = globalParameters.plate_train_x
_pty = globalParameters.plate_train_y

bowl = new LIBEROObject with libero_name "akita_black_bowl_1",
                         with asset_class chosen_asset,
                         with width 0.10, with length 0.10, with height 0.06,
                         in SAFE_REGION

plate = new LIBEROObject with libero_name "plate_1",
                          with asset_class "plate",
                          with width 0.20, with length 0.20, with height 0.02,
                          in PLATE_SAFE_REGION

ramekin = new LIBEROObject with libero_name "glazed_rim_porcelain_ramekin_1",
                            with asset_class "glazed_rim_porcelain_ramekin",
                            with width 0.08, with length 0.08, with height 0.05,
                            in SAFE_REGION

# ──────────────────────────────────────────────────────────────────────────────
# Hard constraints (spatial plausibility)
# ──────────────────────────────────────────────────────────────────────────────

require (distance from bowl to plate)    > _min_clearance
require (distance from bowl to ramekin)  > _min_clearance
require (distance from plate to ramekin) > _min_clearance

# Workspace bounds enforced by `in SAFE_REGION` / `in PLATE_SAFE_REGION`.

# ──────────────────────────────────────────────────────────────────────────────
# Soft constraints (OOD bias)
# ──────────────────────────────────────────────────────────────────────────────

bowl_train_pt  = new Point at Vector(_btx, _bty, TABLE_Z)
plate_train_pt = new Point at Vector(_ptx, _pty, TABLE_Z)

require[0.8] distance from bowl  to bowl_train_pt  > _ood_margin
require[0.7] distance from plate to plate_train_pt > _ood_margin

# ──────────────────────────────────────────────────────────────────────────────
# Scenario composition example (for multi-phase evaluation)
# ──────────────────────────────────────────────────────────────────────────────
# Uncomment to use Scenic's modular scenario system instead of inlined setup:
#
# scenario PositionPerturb():
#     setup:
#         bowl_pos = new Point in TABLE_REGION
#         plate_pos = new Point in TABLE_REGION
#         require (distance from bowl_pos to plate_pos) > globalParameters.min_clearance
#
# scenario ObjectPerturb():
#     setup:
#         _v = ASSET_VARIANTS.get(globalParameters.perturb_class,
#                                  [globalParameters.perturb_class])
#         chosen = Uniform(*_v)
#         param chosen_asset = chosen
#
# scenario Main():
#     compose:
#         do ObjectPerturb()
#         do PositionPerturb()
