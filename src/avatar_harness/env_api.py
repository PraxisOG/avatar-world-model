"""Engine boundary and canonical snapshots, experiment contract sections 1–5."""

from dataclasses import dataclass
import hashlib
from typing import Callable, Protocol

import numpy as np


def _array(value, shape):
    array = np.asarray(value, dtype="<f4")
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite float32 array with shape {shape}")
    # Bytes backing prevents callers from re-enabling writes on frozen snapshots.
    return np.frombuffer(array.tobytes(), dtype="<f4").reshape(shape)


@dataclass(frozen=True)
class Action:
    """Section 3: ten absolute arm targets and two normalized openings (left, right).

    Quantize at the controller boundary so the actual commands sent are exactly
    representable in the contract's float32 storage.
    """

    joint_position_target_rad: np.ndarray
    gripper_opening: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "joint_position_target_rad", _array(self.joint_position_target_rad, (10,)))
        value = _array(self.gripper_opening, (2,))
        if np.any((value < 0) | (value > 1)):
            raise ValueError("gripper opening must be finite and in [0, 1]")
        object.__setattr__(self, "gripper_opening", value)


@dataclass(frozen=True)
class EpisodeInfo:
    """Sections 1–3: resolved calibration, scene and engine provenance at reset."""

    engine_name: str
    engine_version: str
    seed: int
    source_revision: str
    scene: dict
    cameras: dict
    q_closed: tuple[float, float]
    q_open: tuple[float, float]
    device: str = "CPU"


@dataclass(frozen=True)
class SimState:
    """Sections 3–5: immutable numeric snapshot, already quantized for storage.

    Segmentation must close over this snapshot, never mutable engine state.
    It returns int32 [240,320] labels; the recorder checks lossless uint16 fit.
    contact_forces is the net contact force on the block in world-frame newtons.
    """

    step_index: int
    sim_time: float
    measured_joint_position_rad: np.ndarray
    measured_joint_velocity_rad_s: np.ndarray
    measured_gripper_opening: np.ndarray
    left_ee_pose: np.ndarray
    right_ee_pose: np.ndarray
    object_pose_world: np.ndarray
    contact_forces: np.ndarray
    segmentation: Callable[[str], np.ndarray]
    terminated: bool = False

    def __post_init__(self):
        if not isinstance(self.step_index, (int, np.integer)) or self.step_index < 0:
            raise ValueError("step_index must be a nonnegative integer")
        if not np.isfinite(self.sim_time) or self.sim_time < 0:
            raise ValueError("sim_time must be finite and nonnegative")
        for name, size in (
            ("measured_joint_position_rad", 12), ("measured_joint_velocity_rad_s", 12),
            ("measured_gripper_opening", 2), ("left_ee_pose", 7), ("right_ee_pose", 7),
            ("object_pose_world", 7), ("contact_forces", 3),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), (size,)))


def canonical_state_bytes(state: SimState, *, privileged: bool = True) -> bytes:
    """Sections 4–6: little-endian float64 encoding of stored numeric values.

    Order: step, time, measured q, measured dq, openings (left, right), left
    EE pose, right EE pose, then (when privileged) object pose, contact forces, terminated as 0/1. Float32 fields
    are promoted *after* quantization. Segmentation is a derived callable, not
    a numeric state field. Its PNGs are independently checked on reload.
    """
    values = [state.step_index, state.sim_time, *state.measured_joint_position_rad,
              *state.measured_joint_velocity_rad_s, *state.measured_gripper_opening,
              *state.left_ee_pose, *state.right_ee_pose]
    if privileged:
        values.extend([*state.object_pose_world, *state.contact_forces, int(state.terminated)])
    return np.asarray(values, dtype="<f8").tobytes()


def state_checksum(state: SimState) -> str:
    """Section 6: SHA-256 of all frozen numeric state, for privileged diagnostics."""
    return hashlib.sha256(canonical_state_bytes(state)).hexdigest()


class SimEnv(Protocol):
    """Sections 3–4: capture boundary; get_state freezes until the next step."""

    def reset(self, seed: int) -> EpisodeInfo:
        """Sections 1–3: reset to step zero and return resolved scene metadata."""
        ...

    def get_state(self) -> SimState:
        """Section 4: freeze and return measured state without advancing physics."""
        ...

    def render(self, camera_name: str) -> np.ndarray:
        """Sections 2, 4: uint8 [240,320,3] RGB from a frozen state; never step."""
        ...

    def step(self, action: Action, n_steps: int) -> None:
        """Sections 3–4: send absolute targets and hold for n_steps physics ticks."""
        ...

    def command_log_entry(self, action: Action) -> dict:
        """Section 3: return the engine's last actually sent commands, after step."""
        ...
