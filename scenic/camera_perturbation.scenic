"""camera_perturbation.scenic — Layer 3d: Camera pose perturbation.

Perturbs the agentview camera position and tilt angle, which LIBERO-Plus
found to be the most damaging perturbation axis for policy performance.

The camera offsets are injected by LIBEROSimulation._apply_camera_perturbation()
after env.reset(), modifying the MuJoCo model's cam_pos and cam_quat arrays.

Objects are placed at their BDDL canonical positions (no spatial perturbation).

Instantiation (Python)
──────────────────────
    import scenic
    scenario = scenic.scenarioFromFile(
        "scenic/camera_perturbation.scenic",
        params={
            "bddl_path": "...",
            "camera_x_range": 0.10,     # max x offset (metres)
            "camera_y_range": 0.10,
            "camera_z_range": 0.08,
            "camera_tilt_range": 15.0,   # max tilt (degrees)
        },
    )
    scene, _ = scenario.generate()
    # scene.params["camera_x_offset"] → sampled offset value
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# Global parameters
# ──────────────────────────────────────────────────────────────────────────────
param task = "camera_perturbation"
param bddl_path = ""

# Camera perturbation ranges (metres for position, degrees for tilt)
param camera_x_range = 0.10
param camera_y_range = 0.10
param camera_z_range = 0.08
param camera_tilt_range = 15.0

# ──────────────────────────────────────────────────────────────────────────────
# Extract params at compile time
# ──────────────────────────────────────────────────────────────────────────────
_cx = globalParameters.camera_x_range
_cy = globalParameters.camera_y_range
_cz = globalParameters.camera_z_range
_ct = globalParameters.camera_tilt_range

# ──────────────────────────────────────────────────────────────────────────────
# Camera offset sampling
# ──────────────────────────────────────────────────────────────────────────────
_cam_x = Range(-_cx, _cx)
_cam_y = Range(-_cy, _cy)
_cam_z = Range(-_cz, _cz)
_cam_tilt = Range(-_ct, _ct)

param camera_x_offset = _cam_x
param camera_y_offset = _cam_y
param camera_z_offset = _cam_z
param camera_tilt = _cam_tilt

# ──────────────────────────────────────────────────────────────────────────────
# Ego placeholder (required by Scenic — not used for camera-only perturbation)
# ──────────────────────────────────────────────────────────────────────────────
ego = new Object at Vector(0, 0, TABLE_Z), with allowCollisions True
