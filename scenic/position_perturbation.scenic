"""position_perturbation.scenic — Layer 3a: spatial distribution over object placements.

Samples object positions from a continuous uniform distribution over the full table
workspace, subject to hard physical-plausibility constraints enforced by Scenic's
rejection sampler.

Instantiation (Python)
──────────────────────
    import scenic
    scenario = scenic.scenarioFromFile(
        "scenic/position_perturbation.scenic",
        params={
            "task": "put_the_bowl_on_the_plate",
            "bddl_path": "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
        },
    )
    scene, n_iters = scenario.generate()

Per-scene outputs (accessible on scene.objects)
────────────────────────────────────────────────
  bowl.position   → sampled (x, y, TABLE_Z) satisfying all constraints
  plate.position  → sampled (x, y, TABLE_Z)
  ramekin.position → sampled (x, y, TABLE_Z)
  goal_fixture.position → sampled (x, y, TABLE_Z) when optional goal_fixture_* params are set

Design notes
────────────
* All three objects are placed independently uniform in SAFE_REGION (or
  PLATE_SAFE_REGION for large objects) — the `in` specifier handles
  workspace boundary enforcement, no explicit boundary requires needed.
* Optional goal fixtures (drawer, stove, microwave, cabinet, rack) can be
  position-perturbed too by passing `goal_fixture_*` params.
* Hard `require` constraints enforce minimum pairwise clearance.
* The soft `require[0.8]` constraint pushes bowl away from its training
  position without completely forbidding it there — this is deliberately
  weaker than a hard constraint so the distribution still covers the
  training region (needed for proper out-of-distribution evaluation, not
  just out-of-sample evaluation).
* Scenic's rejection sampler will resample until all hard constraints are
  satisfied.  Expected rejections: ~3–8 per accepted sample given the
  clearance constraints above.

To add more objects: copy the pattern below and add pairwise distance
constraints for each new pair.
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# Global parameters — overridable from Python via scenarioFromFile(params=...)
# ──────────────────────────────────────────────────────────────────────────────
param task = "put_the_bowl_on_the_plate"
param bddl_path = ""

# Training positions (from canonical BDDL region centres) used for the
# soft anti-training constraint.  Values are (x, y) in table-relative frame.
param bowl_train_x = 0.12
param bowl_train_y = -0.05
param plate_train_x = 0.04
param plate_train_y = -0.02
param ramekin_train_x = -0.09
param ramekin_train_y = 0.0

# Minimum clearance between any two objects (metres, centre-to-centre).
# Must be ≥ the largest per-pair footprint clearance across all object pairs:
#   bowl (0.10×0.10) + plate (0.20×0.20): hypot(0.10,0.10)/2 + hypot(0.20,0.20)/2 ≈ 0.212 m
# 0.22 provides a small buffer above that maximum.
param min_clearance = 0.22

# Soft constraint: prefer objects at least this far from training position
param ood_margin = 0.15

# Optional goal fixture (for fixture-backed goals such as drawers/stoves)
param goal_fixture_name = ""
param goal_fixture_class = ""
param goal_fixture_width = 0.20
param goal_fixture_length = 0.20
param goal_fixture_height = 0.10
param goal_fixture_train_x = 0.0
param goal_fixture_train_y = 0.0
param goal_fixture_workspace_margin = 0.11

# ──────────────────────────────────────────────────────────────────────────────
# Object declarations — position sampled uniform in TABLE_REGION
# ──────────────────────────────────────────────────────────────────────────────

bowl = new LIBEROObject with libero_name "akita_black_bowl_1",
                         with asset_class "akita_black_bowl",
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
# Extract param values into local variables at compile time.
# Use locals (not globalParameters) inside require closures to avoid scoping
# issues with Scenic's deferred require evaluation.
# ──────────────────────────────────────────────────────────────────────────────
_min_clearance = globalParameters.min_clearance
_ood_margin    = globalParameters.ood_margin
_btx = globalParameters.bowl_train_x
_bty = globalParameters.bowl_train_y
_ptx = globalParameters.plate_train_x
_pty = globalParameters.plate_train_y
_has_goal_fixture = globalParameters.goal_fixture_name != ""
_gfw = globalParameters.goal_fixture_width
_gfl = globalParameters.goal_fixture_length
_gfh = globalParameters.goal_fixture_height
_gfx = globalParameters.goal_fixture_train_x
_gfy = globalParameters.goal_fixture_train_y
_gf_margin = globalParameters.goal_fixture_workspace_margin
_gf_x_lo = TABLE_X_MIN + (_gfw / 2.0) + _gf_margin
_gf_x_hi = TABLE_X_MAX - (_gfw / 2.0) - _gf_margin
_gf_y_lo = TABLE_Y_MIN + (_gfl / 2.0) + _gf_margin
_gf_y_hi = TABLE_Y_MAX - (_gfl / 2.0) - _gf_margin

if _has_goal_fixture:
    goal_fixture = new LIBEROFixture with libero_name globalParameters.goal_fixture_name,
                                    with asset_class globalParameters.goal_fixture_class,
                                    with width _gfw, with length _gfl, with height _gfh,
                                    at Vector(Range(_gf_x_lo, _gf_x_hi),
                                              Range(_gf_y_lo, _gf_y_hi),
                                              TABLE_Z)

# ──────────────────────────────────────────────────────────────────────────────
# Hard constraints — rejection sampler loops until ALL are satisfied
# ──────────────────────────────────────────────────────────────────────────────

# Pairwise clearance: objects must not overlap
require (distance from bowl to plate)    > _min_clearance
require (distance from bowl to ramekin)  > _min_clearance
require (distance from plate to ramekin) > _min_clearance

if _has_goal_fixture:
    _gf_to_bowl = (((_gfw + 0.10) ** 2 + (_gfl + 0.10) ** 2) ** 0.5) / 2.0
    _gf_to_plate = (((_gfw + 0.20) ** 2 + (_gfl + 0.20) ** 2) ** 0.5) / 2.0
    _gf_to_ramekin = (((_gfw + 0.08) ** 2 + (_gfl + 0.08) ** 2) ** 0.5) / 2.0
    require (distance from goal_fixture to bowl) > _gf_to_bowl
    require (distance from goal_fixture to plate) > _gf_to_plate
    require (distance from goal_fixture to ramekin) > _gf_to_ramekin

# Workspace bounds are enforced by `in SAFE_REGION` / `in PLATE_SAFE_REGION`
# specifiers above — no explicit boundary requires needed.

# ──────────────────────────────────────────────────────────────────────────────
# Soft constraints — prefer OOD configurations
# ──────────────────────────────────────────────────────────────────────────────

bowl_train_pt  = new Point at Vector(_btx, _bty, TABLE_Z)
plate_train_pt = new Point at Vector(_ptx, _pty, TABLE_Z)

require[0.8] distance from bowl  to bowl_train_pt  > _ood_margin
require[0.7] distance from plate to plate_train_pt > _ood_margin

if _has_goal_fixture:
    goal_fixture_train_pt = new Point at Vector(_gfx, _gfy, TABLE_Z)
    require[0.7] distance from goal_fixture to goal_fixture_train_pt > _ood_margin
