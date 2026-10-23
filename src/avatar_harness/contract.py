"""Frozen settings and storage schemas from experiment contract sections 1–5."""

from types import MappingProxyType

import numpy as np
import pyarrow as pa

CONTRACT_REVISION = "B"
ARMS = ("left", "right")
RAW_JOINT_ORDER = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)
JOINT_ORDER = tuple(f"{arm}_arm/{joint}" for arm in ARMS for joint in RAW_JOINT_ORDER)
ARM_JOINT_INDICES = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
GRIPPER_INDICES = (5, 11)
COMMAND_JOINT_ORDER = tuple(JOINT_ORDER[i] for i in ARM_JOINT_INDICES)
BASE_POSES = MappingProxyType({
    "left": ((-0.24, -0.12, 0.0), np.pi / 3),
    "right": ((0.24, -0.12, 0.0), 2 * np.pi / 3),
})
BLOCK_POS = (0.0, 0.08, 0.015)
EXTERNAL_POS, EXTERNAL_LOOKAT = (0.72, -0.70, 0.52), (0.16, 0.0, 0.15)
HEAD_POS, HEAD_LOOKAT = (0.0, -0.50, 0.65), (0.0, 0.05, 0.10)
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


def camera_transform(position, lookat):
    """Optical x-right/y-down/z-forward pose in world coordinates."""
    forward = np.asarray(lookat, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    transform[:3, 3] = position
    return tuple(tuple(float(v) for v in row) for row in transform)


CAMERAS = MappingProxyType({
    "external": MappingProxyType({
        **_INTRINSICS, "identity": "fixed", "parent_frame": "world",
        "extrinsics": camera_transform(EXTERNAL_POS, EXTERNAL_LOOKAT),
    }),
    "head": MappingProxyType({
        **_INTRINSICS, "identity": "head", "parent_frame": "world",
        "mount_link": "head_camera_mast",
        "extrinsics": camera_transform(HEAD_POS, HEAD_LOOKAT),
    }),
})

# Fields a future policy input loader may read (actions are training targets).
POLICY_INPUT_FIELDS = (
    "external_path", "head_path", "measured_joint_position_rad",
    "measured_joint_velocity_rad_s", "measured_gripper_opening", "left_ee_pose", "right_ee_pose",
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
V2, V7, V10, V12 = (pa.list_(F32, n) for n in (2, 7, 10, 12))
TRANSITION_SCHEMA = _schema([
    ("episode_id", pa.string()), ("step_index", pa.int64()), ("sim_time", pa.float64()),
    ("external_path", pa.string()), ("head_path", pa.string()),
    ("external_step_index", pa.int64()), ("head_step_index", pa.int64()),
    ("command_joint_position_target_rad", V10), ("command_gripper_opening", V2),
    ("measured_joint_position_rad", V12), ("measured_joint_velocity_rad_s", V12),
    ("measured_gripper_opening", V2), ("left_ee_pose", V7), ("right_ee_pose", V7),
    ("next_step_index", pa.int64()), ("next_sim_time", pa.float64()),
    ("next_measured_joint_position_rad", V12), ("next_measured_joint_velocity_rad_s", V12),
    ("next_measured_gripper_opening", V2),
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
    ("measured_joint_position_rad", V12), ("measured_joint_velocity_rad_s", V12),
    ("measured_gripper_opening", V2), ("left_ee_pose", V7), ("right_ee_pose", V7),
    ("terminated", pa.bool_()), ("state_checksum", pa.string()),
    # No RGB render of the terminal next state: its per-render stamps are null.
])
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("external_state_checksum", pa.string()))
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("head_state_checksum", pa.string()))
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("external_rgb_sha256", pa.string()))
SIM_ONLY_SCHEMA = SIM_ONLY_SCHEMA.append(pa.field("head_rgb_sha256", pa.string()))
SIM_ONLY_FIELDS = tuple(SIM_ONLY_SCHEMA.names)
