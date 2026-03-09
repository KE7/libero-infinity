"""lighting_perturbation.scenic — Layer 3e: Lighting perturbation.

Varies scene lighting by modifying light positions, intensity, and ambient
levels. Tests policy robustness to visual appearance changes that don't
affect physics.

The lighting params are injected by LIBEROSimulation._apply_lighting_perturbation()
after env.reset(), modifying the MuJoCo model's light arrays.

Instantiation (Python)
──────────────────────
    import scenic
    scenario = scenic.scenarioFromFile(
        "scenic/lighting_perturbation.scenic",
        params={
            "bddl_path": "...",
            "intensity_min": 0.5,
            "intensity_max": 2.0,
        },
    )
    scene, _ = scenario.generate()
    # scene.params["light_intensity"] → sampled multiplier
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# Global parameters
# ──────────────────────────────────────────────────────────────────────────────
param task = "lighting_perturbation"
param bddl_path = ""

# Light intensity range (multiplier: 1.0 = default)
param intensity_min = 0.4
param intensity_max = 2.0

# Light position offset ranges (metres)
param light_pos_range = 0.5

# Ambient level range (0.0 = dark, 1.0 = bright)
param ambient_min = 0.05
param ambient_max = 0.6

# ──────────────────────────────────────────────────────────────────────────────
# Extract params at compile time
# ──────────────────────────────────────────────────────────────────────────────
_imin = globalParameters.intensity_min
_imax = globalParameters.intensity_max
_lpr = globalParameters.light_pos_range
_amin = globalParameters.ambient_min
_amax = globalParameters.ambient_max

# ──────────────────────────────────────────────────────────────────────────────
# Lighting sampling
# ──────────────────────────────────────────────────────────────────────────────
_intensity = Range(_imin, _imax)
_lx = Range(-_lpr, _lpr)
_ly = Range(-_lpr, _lpr)
_lz = Range(-_lpr, _lpr)
_ambient = Range(_amin, _amax)

param light_intensity = _intensity
param light_x_offset = _lx
param light_y_offset = _ly
param light_z_offset = _lz
param ambient_level = _ambient

# ──────────────────────────────────────────────────────────────────────────────
# Ego placeholder
# ──────────────────────────────────────────────────────────────────────────────
ego = new Object at Vector(0, 0, TABLE_Z), with allowCollisions True
