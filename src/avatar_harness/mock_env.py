"""Deterministic CPU fake for contract sections 1–4; not SO-101 physics.

Each arm uses a hand-written two-link FK chain in its base XY plane: the
first two joints rotate links of length 0.18 and 0.16 m. Other joints still have independent
first-order dynamics. RGB is a diagnostic pattern, not a geometric renderer.
Neither the URDF nor hardware calibration is loaded or claimed here.
"""

import hashlib
import math
import struct

import numpy as np

from .contract import (ARMS, ARM_JOINT_INDICES, BASE_POSES, CAMERAS, GRIPPER_INDICES,
                       IMAGE_SHAPE, JOINT_ORDER, RAW_JOINT_ORDER, PHYSICS_DT_S, PHYSICS_RATE_HZ)
from .contract import gripper_opening, gripper_target
from .env_api import Action, EpisodeInfo, SimState, canonical_state_bytes

ENGINE_NAME = "avatar_harness.mock"
ENGINE_VERSION = "2"
LINK_LENGTHS = (0.18, 0.16)
RESPONSE_GAIN = 6.0  # s^-1; exact first-order solution cannot overshoot its target.
_Y, _X = np.indices(IMAGE_SHAPE[:2], dtype=np.uint16)


def fake_ee_pose(joints: np.ndarray) -> np.ndarray:
    """Sections 1, 4: fake two-link XY FK; pose is base-frame position + xyzw."""
    a, b = float(joints[0]), float(joints[0] + joints[1])
    x = LINK_LENGTHS[0] * math.cos(a) + LINK_LENGTHS[1] * math.cos(b)
    y = LINK_LENGTHS[0] * math.sin(a) + LINK_LENGTHS[1] * math.sin(b)
    return np.asarray([x, y, 0.025, 0, 0, math.sin(b / 2), math.cos(b / 2)], dtype=np.float32)


def fake_ee_world(joints, arm):
    """Place a fake base-frame FK chain using the contract's fixed base pose."""
    position, yaw = BASE_POSES[arm]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(position) + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ fake_ee_pose(joints)[:3]


def mock_cameras() -> dict:
    """Section 2: contract camera poses; mock RGB remains a diagnostic pattern."""
    return {name: {**definition, "calibration_source": "contract pose; mock diagnostic pattern, not geometric rendering"}
            for name, definition in CAMERAS.items()}


def render_snapshot(state: SimState, camera_name: str) -> np.ndarray:
    """Sections 4–6: deterministic RGB; usable offline for stale-buffer checks.

    A prefix contains literal camera-name bytes, uint64 step, and SHA-256 of
    *only* nonprivileged measured state. Object/contact/segmentation data and
    their checksums never enter RGB. The rest is a smooth diagnostic pattern.
    """
    if camera_name not in CAMERAS:
        raise ValueError(f"unknown camera: {camera_name}")
    digest = hashlib.sha256(canonical_state_bytes(state, privileged=False)).digest()
    offset = 17 if camera_name == "external" else 93
    rgb = np.empty(IMAGE_SHAPE, dtype=np.uint8)
    rgb[..., 0] = (_X + state.step_index % 256 + offset) % 256
    rgb[..., 1] = (_Y + int(digest[0]) + offset) % 256
    rgb[..., 2] = ((_X // 4 + _Y // 4) + int(digest[1])) % 256
    header = camera_name.encode("ascii").ljust(8, b"\0") + struct.pack("<Q", state.step_index) + digest
    rgb.reshape(-1)[:len(header)] = np.frombuffer(header, dtype=np.uint8)
    return rgb


def segmentation_snapshot(object_pose: np.ndarray, joints: np.ndarray, camera_name: str) -> np.ndarray:
    """Section 5: fake int32 labels (0 background, 1 left arm, 2 right arm, 3 block), diagnostics only."""
    if camera_name not in CAMERAS:
        raise ValueError(f"unknown camera: {camera_name}")
    offset = 0 if camera_name == "external" else 25
    labels = np.zeros(IMAGE_SHAPE[:2], dtype=np.int32)
    for i, start in enumerate((0, 6)):
        arm_x = int(95 + 110 * i + 40 * math.sin(float(joints[start]))) + offset
        labels[80:180, max(0, arm_x - 6):min(320, arm_x + 6)] = i + 1
    x = int(160 + 220 * float(object_pose[0])) + offset
    y = int(120 - 220 * float(object_pose[1]))
    labels[max(0, y - 8):min(240, y + 8), max(0, x - 8):min(320, x + 8)] = 3
    return labels


class MockEnv:
    """Sections 1–4: twelve-joint fake with frozen rendering and simple block pushing."""

    def __init__(self, *, table_half_extents=(0.4, 0.35), object_xy=(0.16, -0.20)):
        """Section 1: fixed scene knobs for testing table-edge termination."""
        self._bounds = np.asarray(table_half_extents, dtype=np.float64)
        self._object_xy = np.asarray(object_xy, dtype=np.float64)
        if (self._bounds.shape != (2,) or self._object_xy.shape != (2,)
                or not np.isfinite(self._bounds).all() or not np.isfinite(self._object_xy).all()
                or np.any(self._bounds <= 0) or np.any(np.abs(self._object_xy) > self._bounds)):
            raise ValueError("initial block must be inside finite positive table bounds")
        self._frozen = None
        self._initialized = False

    def reset(self, seed: int) -> EpisodeInfo:
        """Sections 1–3: deterministic fixed scene; seed is retained without randomization."""
        self._step = 0
        self._q = np.asarray([0, 0, 0, 0, 0, 0.5] * 2, dtype=np.float64)
        self._dq = np.zeros(12)
        self._object = np.asarray([*self._object_xy, 0.025, 0, 0, 0, 1], dtype=np.float64)
        self._contact = np.zeros(3)
        self._terminated = False
        self._frozen = None
        self._last_command = None
        self._initialized = True
        scene = {
            "kind": "mock_planar_fixture", "physics_fidelity": "fake; no URDF loaded",
            "source": "TheRobotStudio/SO-ARM100",
            "urdf_path": "Simulation/SO101/so101_new_calib.urdf",
            "urdf_revision": "not-loaded:mock-planar-v2", "joint_names": list(JOINT_ORDER),
            "raw_urdf_joint_names": list(RAW_JOINT_ORDER),
            "base_poses": {arm: {"position_m": list(pos), "yaw_rad": yaw} for arm, (pos, yaw) in BASE_POSES.items()},
            "source_chain_endpoints": ["shoulder_link", "moving_jaw_so101_v1_link"],
            "geometry": {"arm_link_lengths_m": list(LINK_LENGTHS),
                         "table_half_extents_m": self._bounds.tolist(), "table_height_m": 0.0,
                         "block_size_m": [0.05, 0.05, 0.05]},
            "mass_kg": {"block": 0.1, "arm_links": [0.1, 0.1]},
            "friction": {"table": 0.5, "block": 0.5},
            "materials": "flat colors; no textures", "domain_randomization": False,
            "initial_state": {"joint_position_rad": self._q.tolist(),
                              "joint_velocity_rad_s": self._dq.tolist(),
                              "object_pose_world": self._object.tolist()},
            "dynamics": {"response_gain_per_s": RESPONSE_GAIN,
                         "contact_model": "fake XY displacement transfer within 0.055 m",
                         "unused_parameters": "mass and friction describe fixture only"},
        }
        return EpisodeInfo(ENGINE_NAME, ENGINE_VERSION, seed, scene["urdf_revision"],
                           scene, mock_cameras(), (0.0, 0.0), (1.0, 1.0))

    def get_state(self) -> SimState:
        """Section 4: freeze a deep snapshot; repeated reads preserve the same instant."""
        if not self._initialized:
            raise RuntimeError("reset before reading state")
        if self._frozen is None:
            q = self._q.astype(np.float32)
            obj = self._object.astype(np.float32)
            # Copies captured by this closure are never changed by physics.
            segmentation = lambda camera: segmentation_snapshot(obj, q, camera)
            self._frozen = SimState(
                self._step, self._step / PHYSICS_RATE_HZ, q, self._dq,
                [gripper_opening(float(q[i]), 0.0, 1.0) for i in GRIPPER_INDICES],
                fake_ee_pose(q[:6]), fake_ee_pose(q[6:]), obj,
                self._contact, segmentation, self._terminated,
            )
        return self._frozen

    def render(self, camera_name: str) -> np.ndarray:
        """Sections 2, 4: render only after get_state; never advance the simulation."""
        if self._frozen is None:
            raise RuntimeError("render requires a frozen state; call get_state first")
        return render_snapshot(self._frozen, camera_name)

    def step(self, action: Action, n_steps: int) -> None:
        """Sections 3–4: exact first-order response; block leaving the table terminates."""
        if not self._initialized or self._terminated:
            raise RuntimeError("reset required before stepping an inactive episode")
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        self._last_command = {
            "command_joint_position_target_rad": action.joint_position_target_rad.tolist(),
            "command_gripper_opening": action.gripper_opening.tolist(),
        }
        target = np.empty(12)
        target[list(ARM_JOINT_INDICES)] = action.joint_position_target_rad
        target[list(GRIPPER_INDICES)] = [gripper_target(v, 0.0, 1.0) for v in action.gripper_opening]
        self._frozen = None
        for _ in range(n_steps):
            old_q = self._q.copy()
            self._q += (1 - math.exp(-RESPONSE_GAIN * PHYSICS_DT_S)) * (target - self._q)
            self._dq = (self._q - old_q) / PHYSICS_DT_S
            self._contact[:] = 0
            for i, arm in enumerate(ARMS):
                old_ee = fake_ee_world(old_q[6*i:6*i+6], arm)
                new_ee = fake_ee_world(self._q[6*i:6*i+6], arm)
                if np.linalg.norm(self._object[:2] - old_ee[:2]) <= 0.055:
                    displacement = new_ee[:2] - old_ee[:2]
                    self._object[:2] += displacement
                    self._contact[:2] += displacement / PHYSICS_DT_S * 0.1
            self._step += 1
            if np.any(np.abs(self._object[:2]) > self._bounds):
                self._object[2] -= 0.5 * 9.81 * PHYSICS_DT_S**2
                self._terminated = True
                break

    def command_log_entry(self, action: Action) -> dict:
        """Section 3: return the cached controller input, never infer it from measured q."""
        if self._last_command is None:
            raise RuntimeError("no command has been sent")
        if (self._last_command["command_joint_position_target_rad"] != action.joint_position_target_rad.tolist()
                or self._last_command["command_gripper_opening"] != action.gripper_opening.tolist()):
            raise ValueError("requested command is not the most recently applied command")
        return {key: value.copy() if isinstance(value, list) else value
                for key, value in self._last_command.items()}


def scripted_actions(n_steps: int = 300):
    """Sections 3, 6: opposite-phase sweeps on both arms and two open/close cycles."""
    if n_steps < 2:
        raise ValueError("the sweep needs at least two steps")
    for step in range(n_steps):
        phase = 2 * math.pi * step / (n_steps - 1)
        left = np.asarray([0.65 * math.sin(phase), 0.45 * math.sin(2 * phase),
                           0.25 * math.cos(phase), 0.0, 0.0])
        yield Action(np.r_[left, -left], [0.5 - 0.5 * math.cos(phase), 0.5 + 0.5 * math.cos(phase)])
