"""Frozen settings and storage schemas from experiment contract sections 1–5."""

from types import MappingProxyType

import numpy as np
import pyarrow as pa

JOINT_ORDER = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)
PHYSICS_RATE_HZ = 30
PHYSICS_DT_S = 1 / PHYSICS_RATE_HZ
ACTION_HOLD_STEPS = 1
IMAGE_SHAPE = (240, 320, 3)
GRIPPER_FORMULA = "q_gripper_target = q_closed + opening * (q_open - q_closed)"

_INTRINSICS = {
    "width_px": 320, "height_px": 240, "vertical_fov_deg": 60,
    "intrinsic_model": "pinhole_zero_distortion",
    "fx_px": 207.85, "fy_px": 207.85, "cx_px": 160.0, "cy_px": 120.0,
    "distortion_coefficients": (0, 0, 0, 0, 0), "near_m": 0.01, "far_m": 5.0,
    "channels": "RGB",
}
INTRINSICS = MappingProxyType(_INTRINSICS)
CAMERAS = MappingProxyType({
    "external": MappingProxyType({
        **_INTRINSICS, "identity": "fixed", "parent_frame": "world",
        "extrinsics": "pending Task 2 probe",
    }),
    "wrist": MappingProxyType({
        **_INTRINSICS, "identity": "wrist", "parent_frame": "gripper_link",
        "mount_link": "wrist_camera_mount_link", "extrinsics": "pending Task 2 probe",
    }),
})

# Fields a future policy input loader may read (actions are training targets).
POLICY_INPUT_FIELDS = (
    "external_path", "wrist_path", "measured_joint_position_rad",
    "measured_joint_velocity_rad_s", "measured_gripper_opening", "measured_ee_pose",
)


def gripper_target(opening: float, q_closed: float, q_open: float) -> float:
    """Section 3: map normalized opening to the engine's gripper joint limits."""
    if not np.isfinite([opening, q_closed, q_open]).all() or q_closed == q_open:
        raise ValueError("gripper mapping requires finite, distinct limits")
    if not 0 <= opening <= 1:
        raise ValueError("gripper opening must be in [0, 1]")
    return q_closed + opening * (q_open - q_closed)


def gripper_opening(position: float, q_closed: float, q_open: float) -> float:
    """Section 3: invert the same mapping for measured gripper position."""
    gripper_target(0.0, q_closed, q_open)
    return (position - q_closed) / (q_open - q_closed)


def _schema(fields):
    return pa.schema([pa.field(name, dtype, nullable=False) for name, dtype in fields])


F32 = pa.float32()
V5, V6, V7 = (pa.list_(F32, n) for n in (5, 6, 7))
TRANSITION_SCHEMA = _schema([
    ("episode_id", pa.string()), ("step_index", pa.int64()), ("sim_time", pa.float64()),
    ("external_path", pa.string()), ("wrist_path", pa.string()),
    ("external_step_index", pa.int64()), ("wrist_step_index", pa.int64()),
    ("command_joint_position_target_rad", V5), ("command_gripper_opening", F32),
    ("measured_joint_position_rad", V6), ("measured_joint_velocity_rad_s", V6),
    ("measured_gripper_opening", F32), ("measured_ee_pose", V7),
    ("next_step_index", pa.int64()), ("next_sim_time", pa.float64()),
    ("next_measured_joint_position_rad", V6), ("next_measured_joint_velocity_rad_s", V6),
    ("next_measured_gripper_opening", F32),
])

# Contract leaves the companion's numeric types open. Use float32[7] pose
# and float32[3] net world-frame force on the single block, documented in manifest.
# Full snapshots and render stamps stay quarantined here, including the final state.
SIM_ONLY_SCHEMA = _schema([
    ("episode_id", pa.string()), ("step_index", pa.int64()), ("sim_time", pa.float64()),
    ("sim_only", pa.bool_()), ("object_pose_world", V7),
    ("contact_forces", pa.list_(F32, 3)),
    ("segmentation_paths", pa.struct([
        pa.field(camera, pa.string(), nullable=False) for camera in CAMERAS
    ])),
    ("measured_joint_position_rad", V6), ("measured_joint_velocity_rad_s", V6),
    ("measured_gripper_opening", F32), ("measured_ee_pose", V7),
    ("terminated", pa.bool_()), ("state_checksum", pa.string()),
    # No RGB render of the terminal next state: its per-render stamps are null.
])
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("external_state_checksum", pa.string()))
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("wrist_state_checksum", pa.string()))
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("external_rgb_sha256", pa.string()))
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("wrist_rgb_sha256", pa.string()))
SIM_ONLY_FIELDS = tuple(SIM_ONLY_SCHEMA.names)
