"""Offline reload checks for experiment contract sections 2–6.

Usage: python -m avatar_harness.validate EPISODE_DIR
PASS covers capture integrity only, not the live-engine/ROCm training gates.
The mock supports exact offline re-rendering. Other engines require their own
live replay check; here their stored state stamps and image hashes are checked.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .contract import (ACTION_HOLD_STEPS, CAMERAS, GRIPPER_FORMULA, IMAGE_SHAPE,
                       JOINT_ORDER, PHYSICS_DT_S, PHYSICS_RATE_HZ, POLICY_INPUT_FIELDS,
                       SIM_ONLY_FIELDS, SIM_ONLY_SCHEMA, TRANSITION_SCHEMA)
from .env_api import SimState, state_checksum


@dataclass
class ValidationReport:
    """Section 6: named offline checks with an aggregate success flag."""

    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Section 6: true only when every executed check passes."""
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)

    def __bool__(self):
        return self.passed

    def format(self) -> str:
        """Section 6: per-check PASS/FAIL lines followed by the aggregate result."""
        lines = [f"{'PASS' if ok else 'FAIL'} {name}: {detail}" for name, ok, detail in self.checks]
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bits(value):
    return np.asarray(value, dtype="<f4").tobytes()


def _check_array(array):
    if pa.types.is_floating(array.type):
        _require(np.isfinite(array.to_numpy(zero_copy_only=False)).all(), "nonfinite numeric value")
    elif pa.types.is_fixed_size_list(array.type):
        _require(array.values.null_count == 0, "null array element")
        _check_array(array.values)
    elif pa.types.is_struct(array.type):
        for index in range(array.type.num_fields):
            _require(array.field(index).null_count == 0, "null struct element")
            _check_array(array.field(index))


def _check_schema(table, expected):
    _require(len(table.column_names) == len(set(table.column_names)), "duplicate column names")
    for field in expected:
        _require(field.name in table.column_names, f"missing column {field.name}")
        column = table[field.name]
        _require(column.type == field.type, f"{field.name}: expected {field.type}, got {column.type}")
        if not field.nullable:
            _require(column.null_count == 0, f"null in {field.name}")
        for chunk in column.chunks:
            _check_array(chunk)


def _rigid(value):
    transform = np.asarray(value, dtype=np.float64)
    _require(transform.shape == (4, 4) and np.isfinite(transform).all(), "extrinsics must be finite 4x4")
    _require(np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0), "bad homogeneous transform")
    rotation = transform[:3, :3]
    _require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
             and abs(np.linalg.det(rotation) - 1) < 1e-6, "extrinsics must be a rigid rotation")
    return transform


def _snapshot(row):
    return SimState(
        step_index=row["step_index"], sim_time=row["sim_time"],
        measured_joint_position_rad=row["measured_joint_position_rad"],
        measured_joint_velocity_rad_s=row["measured_joint_velocity_rad_s"],
        measured_gripper_opening=row["measured_gripper_opening"],
        measured_ee_pose=row["measured_ee_pose"], object_pose_world=row["object_pose_world"],
        contact_forces=row["contact_forces"], segmentation=lambda camera: None,
        terminated=row["terminated"],
    )


def _read_png(root, relative, shape, *, rgb):
    path = root / relative
    _require(path.resolve().is_relative_to(root.resolve()), "image path escapes episode directory")
    with Image.open(path) as image:
        _require(image.format == "PNG", f"not a PNG: {relative}")
        _require(image.mode == "RGB" if rgb else image.mode in ("I;16", "I;16B", "I"),
                 f"wrong image mode: {relative}")
        pixels = np.asarray(image)
    _require(pixels.shape == shape, f"wrong PNG shape: {relative}")
    _require(pixels.dtype == np.uint8 if rgb else np.issubdtype(pixels.dtype, np.integer),
             f"wrong PNG dtype: {relative}")
    if not rgb:
        _require(np.all((pixels >= 0) & (pixels <= 65535)), "segmentation outside uint16 range")
    return pixels


def validate_episode(episode_dir) -> ValidationReport:
    """Sections 2–6: reload every artifact, check alignment, chain and quarantine."""
    report = ValidationReport()
    root = Path(episode_dir)

    def check(name, function):
        try:
            detail = function()
            report.checks.append((name, True, detail or "ok"))
        except Exception as error:
            report.checks.append((name, False, f"{type(error).__name__}: {error}"))

    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
        table = pq.read_table(root / "transitions.parquet")
        companion = pq.read_table(root / "sim_only.parquet")
        rows, states = table.to_pylist(), companion.to_pylist()
        report.checks.append(("reload", True, f"{len(rows)} transitions, {len(states)} companion states"))
    except Exception as error:
        report.checks.append(("reload", False, f"{type(error).__name__}: {error}"))
        return report

    def schema():
        _check_schema(table, TRANSITION_SCHEMA)
        _check_schema(companion, SIM_ONLY_SCHEMA)
        return "contract columns, shapes, dtypes and finite values"

    def calibration():
        _require(manifest["joint_order"] == list(JOINT_ORDER), "joint order changed")
        _require(manifest["physics_rate_hz"] == PHYSICS_RATE_HZ
                 and manifest["physics_dt_s"] == PHYSICS_DT_S
                 and manifest["action_hold_steps"] == manifest["N"] == ACTION_HOLD_STEPS
                 and manifest["control_interval_s"] == PHYSICS_DT_S, "wrong control timing")
        mapping = manifest["gripper_mapping"]
        _require(mapping["formula"] == GRIPPER_FORMULA and mapping["units"] == "rad", "wrong gripper mapping")
        _require(np.isfinite([mapping["q_closed"], mapping["q_open"]]).all()
                 and mapping["q_closed"] != mapping["q_open"], "invalid gripper limits")
        _require(set(manifest["cameras"]) == set(CAMERAS), "must have exactly external and wrist cameras")
        for name, expected in CAMERAS.items():
            camera = manifest["cameras"][name]
            for key, value in expected.items():
                if key != "extrinsics":
                    _require(camera[key] == (list(value) if isinstance(value, tuple) else value),
                             f"wrong {name} camera field {key}")
            transform = _rigid(camera["extrinsics"])
            if name == "wrist":
                mount = _rigid(camera["mount_transform"])
                optical = _rigid(camera["optical_frame_offset"])
                _require(np.allclose(mount @ optical, transform, rtol=0, atol=1e-6),
                         "wrist extrinsics disagree with mount and optical offset")
        for key in ("episode_id", "engine", "engine_name", "engine_version", "source_revision"):
            _require(isinstance(manifest[key], str) and bool(manifest[key]), f"missing provenance {key}")
        _require(isinstance(manifest["seed"], int), "seed must be integer")
        _require(manifest["engine"] == f'{manifest["engine_name"]} {manifest["engine_version"]}',
                 "engine name/version disagree")
        for key in ("geometry", "mass_kg", "friction", "initial_state", "urdf_revision"):
            _require(bool(manifest["scene"][key]), f"missing scene setting {key}")
        _require(manifest["source_revision"] == manifest["scene"]["urdf_revision"], "source revisions disagree")
        return "numeric rigid camera transforms, intrinsics, scene and provenance"

    def counts():
        _require(len(rows) > 0, "episode is empty")
        _require(len(states) == len(rows) + 1, "companion must include final next state")
        _require(manifest["transition_count"] == len(rows) and manifest["state_count"] == len(states),
                 "manifest counts disagree")
        expected_rgb, expected_seg = set(), set()
        for camera in CAMERAS:
            for row in rows:
                relative = f'images/{camera}/{row["step_index"]}.png'
                _require(row[f"{camera}_path"] == relative, "RGB path does not match step")
                expected_rgb.add(relative)
            for state in states:
                relative = f'sim_only/segmentation/{camera}/{state["step_index"]}.png'
                _require(state["segmentation_paths"][camera] == relative, "segmentation path does not match step")
                expected_seg.add(relative)
        actual_rgb = {p.relative_to(root).as_posix() for p in (root / "images").rglob("*.png")}
        actual_seg = {p.relative_to(root).as_posix() for p in (root / "sim_only").rglob("*.png")}
        _require(actual_rgb == expected_rgb and len(actual_rgb) == 2 * len(rows), "RGB PNG count/paths disagree")
        _require(actual_seg == expected_seg and len(actual_seg) == 2 * len(states), "segmentation PNG count/paths disagree")
        return f"{len(actual_rgb)} RGB PNGs, {len(actual_seg)} segmentation PNGs"

    def alignment():
        for row in rows:
            _require(row["external_step_index"] == row["wrist_step_index"] == row["step_index"],
                     f'camera step mismatch at {row["step_index"]}')
        return "external_step_index == wrist_step_index == step_index"

    def chain():
        names = ("measured_joint_position_rad", "measured_joint_velocity_rad_s", "measured_gripper_opening")
        for left, right in zip(rows, rows[1:]):
            _require(left["next_step_index"] == right["step_index"], "next step breaks transition chain")
            for name in names:
                _require(_bits(left[f"next_{name}"]) == _bits(right[name]), f"bitwise chain mismatch: {name}")
        _require(len(states) == len(rows) + 1, "final next state missing")
        for index, row in enumerate(rows):
            for name in (*names, "measured_ee_pose"):
                _require(_bits(row[name]) == _bits(states[index][name]), f"observation/companion mismatch: {name}")
            for name in names:
                _require(_bits(row[f"next_{name}"]) == _bits(states[index + 1][name]),
                         f"next-state/companion mismatch: {name}")
        return "bitwise measured-state chain, including final next state"

    def timing():
        for records in (rows, states):
            keys = [(row["episode_id"], row["step_index"]) for row in records]
            _require(len(keys) == len(set(keys)), "duplicate (episode_id, step_index)")
            previous = -1.0
            for index, row in enumerate(records):
                _require(row["episode_id"] == manifest["episode_id"], "episode ID disagrees with manifest")
                _require(row["step_index"] == index, "step sequence must start at zero and be contiguous")
                _require(row["sim_time"] == index / PHYSICS_RATE_HZ and row["sim_time"] > previous,
                         "simulation time is not strictly monotonic at 30 Hz")
                previous = row["sim_time"]
        for row in rows:
            _require(row["next_step_index"] == row["step_index"] + ACTION_HOLD_STEPS
                     and row["next_sim_time"] == row["next_step_index"] / PHYSICS_RATE_HZ,
                     "next step/time violates N=1")
        _require(not any(state["terminated"] for state in states[:-1]), "episode continued after termination")
        return "unique episode/step keys and t/30 timing"

    def commands():
        delta = max(np.max(np.abs(np.asarray(row["command_joint_position_target_rad"])
                                  - np.asarray(row["measured_joint_position_rad"][:5]))) for row in rows)
        _require(delta > 1e-6, "all commands equal measured state; suggests relabeling bug")
        targets = np.asarray([row["command_joint_position_target_rad"] for row in rows])
        _require(np.any(np.ptp(targets, axis=0) > 0), "scripted arm commands are constant")
        return f"max |command-measured| = {delta:.6g} rad; nonconstant commands"

    def motion():
        positions = np.asarray([row["measured_joint_position_rad"][:5] for row in rows])
        span = float(np.max(np.ptp(positions, axis=0)))
        _require(span > 0.1, f"arm motion range {span:.6g} rad does not exceed 0.1 rad")
        return f"max measured arm range = {span:.6g} rad"

    def quarantine():
        _require(set(table.column_names) == set(TRANSITION_SCHEMA.names),
                 "transition table contains columns outside the contract allowlist (possible privileged leak)")
        _require(all(row["sim_only"] is True for row in states), "companion rows must have sim_only==true")
        _require(manifest["policy_input_fields"] == list(POLICY_INPUT_FIELDS), "policy input allowlist changed")
        _require(manifest["sim_only_fields"] == list(SIM_ONLY_FIELDS), "sim_only field list changed")
        declaration = manifest["sim_only"]
        _require(declaration["sim_only"] is True, "missing manifest quarantine flag")
        for name in SIM_ONLY_FIELDS:
            _require(declaration["fields"][name]["sim_only"] is True, f"unflagged sim_only field {name}")
        for name in ("sim_only.parquet", "sim_only/segmentation"):
            _require(declaration["files"][name]["sim_only"] is True, f"unflagged sim_only file {name}")
        return "privileged fields/files isolated and flagged; policy input allowlist intact"

    def measured_values():
        mapping = manifest["gripper_mapping"]
        for state in states:
            opening = state["measured_gripper_opening"]
            q = state["measured_joint_position_rad"][5]
            expected = mapping["q_closed"] + opening * (mapping["q_open"] - mapping["q_closed"])
            _require(0 <= opening <= 1 and np.isclose(q, expected, atol=1e-6, rtol=0), "measured gripper mapping disagrees")
            for pose in (state["measured_ee_pose"], state["object_pose_world"]):
                _require(abs(np.linalg.norm(pose[3:]) - 1) < 1e-5, "non-unit pose quaternion")
        _require(all(0 <= row["command_gripper_opening"] <= 1 for row in rows), "command opening outside [0,1]")
        return "gripper mapping and xyzw quaternion normalization"

    def stamps():
        for index, row in enumerate(states):
            checksum = state_checksum(_snapshot(row))
            _require(row["state_checksum"] == checksum, f'state checksum mismatch at {row["step_index"]}')
            for camera in CAMERAS:
                if index < len(rows):
                    _require(row[f"{camera}_state_checksum"] == checksum, f"{camera} render state stamp mismatch")
                else:
                    _require(row[f"{camera}_state_checksum"] is None and row[f"{camera}_rgb_sha256"] is None,
                             "final next state must not claim an RGB render")
        return "canonical float64 little-endian state stamps match both render calls"

    def pixels():
        # Importing this CPU fake never loads a simulator or GPU runtime.
        from .mock_env import ENGINE_NAME, ENGINE_VERSION, render_snapshot, segmentation_snapshot
        is_mock = manifest["engine_name"] == ENGINE_NAME
        if is_mock:
            _require(manifest["engine_version"] == ENGINE_VERSION, "unsupported mock replay version")
        for index, state in enumerate(states):
            snapshot = _snapshot(state)
            for camera in CAMERAS:
                labels = _read_png(root, state["segmentation_paths"][camera], IMAGE_SHAPE[:2], rgb=False)
                if is_mock:
                    expected = segmentation_snapshot(snapshot.object_pose_world, snapshot.measured_joint_position_rad, camera)
                    _require(np.array_equal(labels, expected), f"{camera} segmentation replay differs at {index}")
                if index < len(rows):
                    rgb = _read_png(root, rows[index][f"{camera}_path"], IMAGE_SHAPE, rgb=True)
                    _require(hashlib.sha256(rgb.tobytes()).hexdigest() == state[f"{camera}_rgb_sha256"],
                             f"{camera} RGB digest differs at {index}")
                    if is_mock:
                        _require(np.array_equal(rgb, render_snapshot(snapshot, camera)),
                                 f"{camera} replay differs at {index}; stale or duplicated render buffer")
        if manifest["engine_name"] == "avatar_harness.mujoco":
            # Optional engine imported only for its episodes; CPU mock tests stay independent.
            from .mujoco_env import verify_replay
            return verify_replay(root, manifest, rows, states)
        return ("all PNGs decoded; mock RGB/segmentation re-render exactly matches reloaded state" if is_mock else
                "all PNGs decoded and RGB digests match; live-engine re-render is outside this offline check")

    def performance():
        elapsed = benchmark["elapsed_s"]
        _require(np.isfinite(elapsed) and elapsed > 0, "invalid benchmark elapsed seconds")
        counts = benchmark["counts"]
        _require(counts == {"transitions": len(rows), "frames": 2 * len(rows),
                            "physics_steps": len(rows) * ACTION_HOLD_STEPS,
                            "segmentation_masks": 2 * len(states)}, "benchmark counts disagree")
        for rate, count in (("frames_per_sec", "frames"), ("steps_per_sec", "physics_steps")):
            _require(np.isclose(benchmark[rate], counts[count] / elapsed, rtol=1e-12, atol=0), "benchmark rate/count mismatch")
        _require(benchmark["stage"] == "capture_and_write" and bool(benchmark["scope"]), "missing measurement scope")
        _require(benchmark["device"] == manifest["device"], "device mismatch")
        vram = benchmark["peak_vram_bytes"]
        _require(vram is None or (isinstance(vram, int) and vram >= 0), "invalid VRAM bytes")
        if benchmark["device"] == "CPU":
            _require(vram == 0, "CPU mock must report zero GPU memory")
        for name in ("python", "numpy", "pyarrow", "pillow", "harness", "engine"):
            _require(bool(benchmark["software_versions"][name]), f"missing software version {name}")
        return "capture counts, elapsed seconds, rates, device and software versions"

    for name, function in (
        ("schema", schema), ("manifest", calibration), ("file_counts", counts),
        ("step_alignment", alignment), ("state_chain", chain), ("time_and_keys", timing),
        ("command_vs_measured", commands), ("motion", motion), ("quarantine", quarantine),
        ("measured_values", measured_values), ("state_checksums", stamps),
        ("image_content", pixels), ("benchmark", performance),
    ):
        check(name, function)
    return report


def main(argv=None) -> int:
    """Section 6: validate EPISODE_DIR and print per-check results; exit 0/1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args(argv)
    report = validate_episode(args.episode_dir)
    print(report.format())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
