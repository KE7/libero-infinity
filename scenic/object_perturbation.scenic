"""object_perturbation.scenic — Layer 3b: OOD asset distribution over object identities.

Samples object asset identities from a Scenic 3 distribution: at each scene sample,
Scenic draws uniformly from ASSET_VARIANTS for each perturbed object.

The chosen asset_class is then used by the Python harness (eval.py) to:
  1. Rewrite the BDDL object declaration via bddl_preprocessor.substitute_asset()
  2. Reload the LIBERO environment so MuJoCo loads the correct mesh/material

Objects whose class has no OOD variants (not in ASSET_VARIANTS) keep their
canonical class.  Their positions are placed at the BDDL canonical location
(no spatial perturbation — use combined.scenic for both).

Instantiation (Python)
──────────────────────
    import scenic
    from libero_infinity.bddl_preprocessor import patched_bddl

    scenario = scenic.scenarioFromFile(
        "scenic/object_perturbation.scenic",
        params={
            "perturb_class": "akita_black_bowl",
            "bddl_path": "...",
        },
    )
    scene, _ = scenario.generate()

    # The chosen asset is in scene.params["chosen_asset"]
    chosen = scene.params["chosen_asset"]
    with patched_bddl(bddl_path, {"akita_black_bowl": chosen}) as tmp_bddl:
        env = OffScreenRenderEnv(bddl_file_name=tmp_bddl, ...)

Per-scene outputs
──────────────────
    scene.params["chosen_asset"]   → e.g. "white_bowl" or "yellow_bowl"
    scene.objects[0].asset_class   → same value (on the LIBEROObject)

Design notes
────────────
* Scenic's `Uniform(*list)` samples one element uniformly at random.
* The chosen_asset is exposed as a global parameter so eval.py can
  read it from scene.params without inspecting scene.objects.
* Object position is kept at the BDDL canonical centre (no spatial noise).
  To combine spatial + object perturbation, see combined_perturbation.scenic.
* If you want to exclude the canonical (training) asset from the distribution,
  set `include_canonical = False` in params; the Scenic program slices [1:].
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# Global parameters
# ──────────────────────────────────────────────────────────────────────────────
param perturb_class     = "akita_black_bowl"   # canonical class being perturbed
param bddl_path         = ""
param include_canonical = True    # include canonical class in the draw?

# ──────────────────────────────────────────────────────────────────────────────
# Asset sampling
# ──────────────────────────────────────────────────────────────────────────────

# Retrieve the variant list from the world-model registry
_all_variants = ASSET_VARIANTS.get(globalParameters.perturb_class,
                                   [globalParameters.perturb_class])

# Optionally exclude the canonical (first) entry for purely OOD evaluation
_variants = _all_variants if globalParameters.include_canonical else _all_variants[1:]

# Sample one asset uniformly — Scenic propagates this through require constraints
chosen_asset = Uniform(*_variants)

# Expose as a global param so eval.py can read from scene.params
param chosen_asset = chosen_asset

# ──────────────────────────────────────────────────────────────────────────────
# Object declaration — uses BDDL canonical position (no spatial perturbation)
# The simulator will NOT override the position of this object; it keeps
# wherever the BDDL default placement sampler puts it.
# ──────────────────────────────────────────────────────────────────────────────
bowl = new LIBEROObject with libero_name "akita_black_bowl_1",
                         with asset_class chosen_asset,
                         with width 0.10, with length 0.10, with height 0.06,
                         with allowCollisions True,
                         at Vector(-0.09, 0.0, TABLE_Z)   # BDDL canonical centre for bowl

# Secondary objects: placed at their BDDL canonical positions (no spatial perturbation).
# allowCollisions=True prevents Scenic's FCL geometry check for these fixed placements.
plate = new LIBEROObject with libero_name "plate_1",
                          with asset_class "plate",
                          with width 0.20, with length 0.20, with height 0.02,
                          with allowCollisions True,
                          at Vector(0.05, -0.02, TABLE_Z)

# ──────────────────────────────────────────────────────────────────────────────
# Constraints
# ──────────────────────────────────────────────────────────────────────────────

# No spatial constraints needed for pure object perturbation.
# chosen_asset was already sampled from Uniform(*_variants) which ensures
# it is in ASSET_VARIANTS[perturb_class] — no further require needed.
# (Avoid referencing globalParameters inside require closures; use locals.)
