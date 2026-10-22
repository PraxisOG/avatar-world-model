"""Raw Parquet/PNG recorder implementing experiment contract sections 1–6."""

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import PIL
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__
from .contract import (ACTION_HOLD_STEPS, ARMS, CAMERAS, COMMAND_JOINT_ORDER,
                       CONTRACT_REVISION, GRIPPER_FORMULA, IMAGE_SHAPE,
                       JOINT_ORDER, RAW_JOINT_ORDER, PHYSICS_DT_S, PHYSICS_RATE_HZ, POLICY_INPUT_FIELDS,
                       SIM_ONLY_FIELDS, SIM_ONLY_SCHEMA, TRANSITION_SCHEMA)
from .env_api import Action, SimEnv, SimState, state_checksum


def _measured(state):
    return {
        "measured_joint_position_rad": state.measured_joint_position_rad.tolist(),
        "measured_joint_velocity_rad_s": state.measured_joint_velocity_rad_s.tolist(),
        "measured_gripper_opening": state.measured_gripper_opening.tolist(),
        "left_ee_pose": state.left_ee_pose.tolist(),
        "right_ee_pose": state.right_ee_pose.tolist(),
    }


def _git_revision():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=2, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class EpisodeRecorder:
    """Sections 3–5: freeze → render pair → read measured → step → next state.

    meta accepts seed (default 0) and episode_id (default episode-{seed}); all
    other user metadata is nested to prevent overriding engine provenance.
    record() requires an empty output directory and never deletes existing data.
    """

    def __init__(self, env: SimEnv, out_dir, meta: dict):
        """Sections 1–4: bind an engine, episode directory and capture metadata."""
        self.env = env
        self.out_dir = Path(out_dir)
        self.meta = dict(meta)

    def _privileged_row(self, state, episode_id):
        paths = {}
        for camera in CAMERAS:
            labels = state.segmentation(camera)
            if (not isinstance(labels, np.ndarray) or labels.shape != IMAGE_SHAPE[:2]
                    or labels.dtype != np.int32 or np.any(labels < 0) or np.any(labels > 65535)):
                raise ValueError("segmentation must be int32 [240,320] IDs in [0,65535]")
            relative = f"sim_only/segmentation/{camera}/{state.step_index}.png"
            Image.fromarray(labels.astype(np.uint16)).save(self.out_dir / relative)
            paths[camera] = relative
        return {
            "episode_id": episode_id, "step_index": state.step_index, "sim_time": state.sim_time,
            "sim_only": True, "object_pose_world": state.object_pose_world.tolist(),
            "contact_forces": state.contact_forces.tolist(), "segmentation_paths": paths,
            **_measured(state), "terminated": state.terminated, "state_checksum": state_checksum(state),
            **{f"{camera}_{suffix}": None for camera in CAMERAS
               for suffix in ("state_checksum", "rgb_sha256")},
        }

    def _assert_frozen(self, frozen):
        measured = self.env.get_state()
        if state_checksum(measured) != state_checksum(frozen):
            raise ValueError("render/segmentation changed the frozen simulation state")
        return measured

    def record(self, actions: Iterable[Action]) -> Path:
        """Sections 1–6: capture an episode, including the final next-state companion."""
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            raise FileExistsError(f"episode directory must be empty: {self.out_dir}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for camera in CAMERAS:
            (self.out_dir / "images" / camera).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "sim_only" / "segmentation" / camera).mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        seed = self.meta.get("seed", 0)
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        episode_id = self.meta.get("episode_id", f"episode-{seed}")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a nonempty string")
        info = self.env.reset(seed)
        if info.seed != seed:
            raise ValueError("engine returned a different seed")
        transitions, privileged = [], []
        for action in actions:
            frozen = self.env.get_state()
            if frozen.terminated:
                break
            if frozen.step_index != len(transitions) or frozen.sim_time != frozen.step_index / PHYSICS_RATE_HZ:
                raise ValueError("engine did not freeze at the expected step and simulation time")
            if privileged and privileged[-1]["state_checksum"] != state_checksum(frozen):
                raise ValueError("state changed between transitions without stepping")
            row = {"episode_id": episode_id, "step_index": frozen.step_index, "sim_time": frozen.sim_time}
            stamps = {}
            for camera in CAMERAS:
                stamps[f"{camera}_state_checksum"] = state_checksum(self._assert_frozen(frozen))
                # Copy immediately: a renderer may reuse its storage for the other camera.
                rendered = self.env.render(camera)
                if (not isinstance(rendered, np.ndarray) or rendered.dtype != np.uint8
                        or rendered.shape != IMAGE_SHAPE):
                    raise ValueError("render must return uint8 [240,320,3] RGB")
                pixels = rendered.copy()
                self._assert_frozen(frozen)
                relative = f"images/{camera}/{frozen.step_index}.png"
                Image.fromarray(pixels).save(self.out_dir / relative)
                row[f"{camera}_path"] = relative
                row[f"{camera}_step_index"] = frozen.step_index
                stamps[f"{camera}_rgb_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()

            measured = self._assert_frozen(frozen)
            row.update(_measured(measured))
            if not privileged:
                privileged.append(self._privileged_row(measured, episode_id))
            privileged[-1].update(stamps)
            self._assert_frozen(frozen)

            self.env.step(action, ACTION_HOLD_STEPS)
            sent = self.env.command_log_entry(action)
            # Read engine-side commands; never reconstruct them from measured state.
            actual = Action(sent["command_joint_position_target_rad"], sent["command_gripper_opening"])
            if (actual.joint_position_target_rad.tolist() != list(sent["command_joint_position_target_rad"])
                    or actual.gripper_opening.tolist() != list(sent["command_gripper_opening"])):
                raise ValueError("engine commands must be float32-representable at the controller boundary")
            row["command_joint_position_target_rad"] = actual.joint_position_target_rad.tolist()
            row["command_gripper_opening"] = actual.gripper_opening.tolist()
            next_state = self.env.get_state()
            if (next_state.step_index != frozen.step_index + ACTION_HOLD_STEPS
                    or next_state.sim_time != next_state.step_index / PHYSICS_RATE_HZ):
                raise ValueError("engine advanced the wrong number of physics steps")
            row.update({
                "next_step_index": next_state.step_index, "next_sim_time": next_state.sim_time,
                "next_measured_joint_position_rad": next_state.measured_joint_position_rad.tolist(),
                "next_measured_joint_velocity_rad_s": next_state.measured_joint_velocity_rad_s.tolist(),
                "next_measured_gripper_opening": next_state.measured_gripper_opening.tolist(),
            })
            transitions.append(row)
            privileged.append(self._privileged_row(next_state, episode_id))
            self._assert_frozen(next_state)
            if next_state.terminated:
                break

        if not transitions:
            raise ValueError("cannot record an empty episode")
        pq.write_table(pa.Table.from_pylist(transitions, schema=TRANSITION_SCHEMA),
                       self.out_dir / "transitions.parquet", compression="NONE")
        pq.write_table(pa.Table.from_pylist(privileged, schema=SIM_ONLY_SCHEMA),
                       self.out_dir / "sim_only.parquet", compression="NONE")
        manifest = {
            "episode_id": episode_id, "seed": seed, "scene": info.scene,
            "contract_revision": CONTRACT_REVISION,
            "raw_urdf_joint_order": list(RAW_JOINT_ORDER),
            "command_joint_order": list(COMMAND_JOINT_ORDER), "gripper_order": list(ARMS),
            "source_revision": info.source_revision, "joint_order": list(JOINT_ORDER),
            "gripper_mapping": {arm: {"formula": GRIPPER_FORMULA, "q_closed": info.q_closed[i],
                                      "q_open": info.q_open[i], "units": "rad"}
                                for i, arm in enumerate(ARMS)},
            "physics_rate_hz": PHYSICS_RATE_HZ, "physics_dt_s": PHYSICS_DT_S,
            "N": ACTION_HOLD_STEPS, "action_hold_steps": ACTION_HOLD_STEPS,
            "control_interval_s": ACTION_HOLD_STEPS * PHYSICS_DT_S, "cameras": info.cameras,
            "engine": f"{info.engine_name} {info.engine_version}", "engine_name": info.engine_name,
            "engine_version": info.engine_version, "device": info.device,
            "harness_version": __version__, "harness_git_revision": _git_revision(),
            "capture_metadata": self.meta, "transition_count": len(transitions),
            "state_count": len(privileged),
            "termination_reason": "object_off_table" if privileged[-1]["terminated"] else "actions_exhausted",
            "validation_scope": "offline capture integrity; excludes physical fidelity and ROCm training",
            "policy_input_fields": list(POLICY_INPUT_FIELDS),
            "sim_only_fields": list(SIM_ONLY_FIELDS),
            "sim_only": {"sim_only": True, "files": {
                "sim_only.parquet": {"sim_only": True},
                "sim_only/segmentation": {"sim_only": True},
            }, "fields": {name: {"sim_only": True} for name in SIM_ONLY_FIELDS},
                "object_pose_world": "float32[7]: position meters + quaternion xyzw, world frame",
                "contact_forces": "float32[3]: net contact force on block, world frame, newtons",
                "segmentation_encoding": "int32 IDs losslessly stored as uint16 grayscale PNG; range 0..65535",
                "state_checksum_encoding": (
                    "SHA-256; float64 little-endian; stored float32 promoted to float64; order: "
                    "step_index, sim_time, measured_joint_position_rad[12], measured_joint_velocity_rad_s[12], "
                    "measured_gripper_opening[2], left_ee_pose[7], right_ee_pose[7], object_pose_world[7], contact_forces[3], "
                    "terminated(0/1); segmentation callable excluded; per-render stamps null at final next state"
                )},
        }
        _write_json(self.out_dir / "manifest.json", manifest)
        elapsed = time.perf_counter() - started
        count = len(transitions)
        _write_json(self.out_dir / "benchmark.json", {
            "stage": "capture_and_write", "elapsed_s": elapsed,
            "scope": "reset through manifest write, including RGB, segmentation and Parquet; excludes validation",
            "counts": {"transitions": count, "frames": 2 * count,
                       "physics_steps": count * ACTION_HOLD_STEPS, "segmentation_masks": 2 * (count + 1)},
            "frames_per_sec": 2 * count / elapsed, "steps_per_sec": count * ACTION_HOLD_STEPS / elapsed,
            "peak_vram_bytes": 0 if info.device == "CPU" else None,
            "vram_measurement": "no GPU used" if info.device == "CPU" else "not measured by engine-agnostic recorder",
            "device": info.device, "software_versions": {
                "python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__,
                "pillow": PIL.__version__, "harness": __version__, "engine": manifest["engine"],
            },
        })
        return self.out_dir
