"""background_perturbation.scenic — Layer 3f: Background texture perturbation.

Varies wall and floor textures by modifying MuJoCo's writable mat_texid buffer,
sampling uniformly from the LIBERO PNG texture asset pool on disk.

The texture params are injected by LIBEROSimulation._apply_background_perturbation()
after env.reset(), using model.material_name2id() and model.texture_name2id()
to swap the textures assigned to the ``walls_mat`` and ``floorplane`` materials.
When a sampled texture name is not loaded in the current model, the simulator
falls back to a random loaded texture gracefully.

Instantiation (Python)
──────────────────────
    import scenic
    scenario = scenic.scenarioFromFile(
        "scenic/background_perturbation.scenic",
        params={"bddl_path": "..."},
    )
    scene, _ = scenario.generate()
    # scene.params["wall_texture"]  → sampled texture name, e.g. "gray_wall"
    # scene.params["floor_texture"] → sampled texture name, e.g. "marble_floor"
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# Global parameters
# ──────────────────────────────────────────────────────────────────────────────
param task = "background_perturbation"
param bddl_path = ""

# ──────────────────────────────────────────────────────────────────────────────
# Texture candidate pool (discovered from LIBERO assets at compile time)
# Falls back to LIBERO_BACKGROUND_TEXTURES from libero_model if the disk
# list is empty (should not happen in a normal installation).
# ──────────────────────────────────────────────────────────────────────────────
_bg_candidates = LIBERO_BACKGROUND_TEXTURES if LIBERO_BACKGROUND_TEXTURES else ["smooth_light_gray_plaster"]

# ──────────────────────────────────────────────────────────────────────────────
# Texture sampling — each generated scene gets a specific name
# ──────────────────────────────────────────────────────────────────────────────
param wall_texture = Uniform(*_bg_candidates)
param floor_texture = Uniform(*_bg_candidates)

# ──────────────────────────────────────────────────────────────────────────────
# Ego placeholder (required by Scenic; the simulator ignores its position)
# ──────────────────────────────────────────────────────────────────────────────
ego = new Object at Vector(0, 0, TABLE_Z), with allowCollisions True
