"""Offline unittest checks for experiment contract sections 3–6 (no GPU)."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from avatar_harness.contract import ARMS, ARM_JOINT_INDICES, CAMERAS, JOINT_ORDER, TRANSITION_SCHEMA, gripper_target
from avatar_harness.env_api import Action, canonical_state_bytes, state_checksum
from avatar_harness.mock_env import MockEnv, fake_ee_pose, fake_ee_world, render_snapshot, scripted_actions
from avatar_harness.recorder import EpisodeRecorder
from avatar_harness.validate import validate_episode


class HarnessTests(unittest.TestCase):
    """Section 6: end-to-end reload, intentional corruptions and deterministic replay."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(prefix="harness-fixture-")
        cls.original = Path(cls.fixture.name) / "episode"
        cls.count = 48
        EpisodeRecorder(MockEnv(), cls.original, {"seed": 42, "episode_id": "test-42"}).record(
            scripted_actions(cls.count)
        )

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="harness-test-")
        self.addCleanup(self.work.cleanup)
        self.episode = Path(self.work.name) / "episode"
        shutil.copytree(self.original, self.episode)

    def _rewrite(self, mutate, filename="transitions.parquet"):
        path = self.episode / filename
        table = pq.read_table(path)
        rows = table.to_pylist()
        mutate(rows)
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path, compression="NONE")

    def _fails(self, check):
        report = validate_episode(self.episode)
        self.assertFalse(report.passed, report.format())
        self.assertTrue(any(name == check and not passed for name, passed, _ in report.checks), report.format())

    def test_full_mock_capture_validate_pass(self):
        report = validate_episode(self.episode)
        self.assertTrue(report.passed, report.format())
        table = pq.read_table(self.episode / "transitions.parquet")
        self.assertEqual(table.schema, TRANSITION_SCHEMA)
        self.assertEqual(table.num_rows, self.count)
        states = pq.read_table(self.episode / "sim_only.parquet").to_pylist()
        self.assertEqual(len(states), self.count + 1)
        self.assertEqual(states[-1]["step_index"], self.count)
        for camera in CAMERAS:
            self.assertIsNone(states[-1][f"{camera}_state_checksum"])
            self.assertTrue((self.episode / states[-1]["segmentation_paths"][camera]).is_file())

    def test_deleted_png_fails(self):
        (self.episode / "images/external/3.png").unlink()
        self._fails("file_counts")

    def test_extra_png_fails(self):
        shutil.copyfile(self.episode / "images/head/0.png", self.episode / "images/head/999.png")
        self._fails("file_counts")

    def test_wrong_external_step_index_fails(self):
        self._rewrite(lambda rows: rows[4].update(external_step_index=3))
        self._fails("step_alignment")

    def test_wrong_row_next_measured_fails(self):
        self._rewrite(lambda rows: rows[4].update(
            next_measured_joint_position_rad=rows[3]["measured_joint_position_rad"]
        ))
        self._fails("state_chain")

    def test_relabel_commands_as_measured_fails(self):
        def corrupt(rows):
            for row in rows:
                row["command_joint_position_target_rad"] = [row["measured_joint_position_rad"][i] for i in ARM_JOINT_INDICES]
                row["command_gripper_opening"] = row["measured_gripper_opening"]
        self._rewrite(corrupt)
        self._fails("command_vs_measured")

    def test_state_checksum_and_parquet_bytes_deterministic(self):
        second = Path(self.work.name) / "repeat"
        EpisodeRecorder(MockEnv(), second, {"seed": 42, "episode_id": "test-42"}).record(scripted_actions(self.count))
        for filename in ("transitions.parquet", "sim_only.parquet"):
            self.assertEqual((self.original / filename).read_bytes(), (second / filename).read_bytes())
        self.assertTrue(validate_episode(second).passed)

    def test_stale_rgb_with_matching_file_digest_still_fails(self):
        # Model a renderer that gave the recorder stale pixels: even its saved
        # RGB digest is consistent, but replay from frozen state must reject it.
        stale = self.episode / "images/external/2.png"
        shutil.copyfile(stale, self.episode / "images/external/3.png")
        with Image.open(stale) as image:
            digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        self._rewrite(lambda rows: rows[3].update(external_rgb_sha256=digest), "sim_only.parquet")
        self._fails("image_content")

    def test_stale_render_implementation_fails(self):
        class StaleEnv(MockEnv):
            def render(self, camera_name):
                if not hasattr(self, "buffers"):
                    self.buffers = {}
                if camera_name not in self.buffers:
                    self.buffers[camera_name] = super().render(camera_name)
                return self.buffers[camera_name]
        out = Path(self.work.name) / "stale"
        EpisodeRecorder(StaleEnv(), out, {}).record(scripted_actions(16))
        report = validate_episode(out)
        self.assertFalse(report.passed)
        self.assertIn("stale or duplicated render buffer", report.format())

    def test_wrong_schema_dtype_fails(self):
        path = self.episode / "transitions.parquet"
        table = pq.read_table(path)
        index = table.schema.get_field_index("command_joint_position_target_rad")
        column = pa.array(table.column(index).to_pylist(), type=pa.list_(pa.float64(), 10))
        pq.write_table(table.set_column(index, "command_joint_position_target_rad", column), path)
        self._fails("schema")

    def test_wrong_schema_shape_fails(self):
        path = self.episode / "transitions.parquet"
        table = pq.read_table(path)
        index = table.schema.get_field_index("left_ee_pose")
        column = pa.array([value[:6] for value in table.column(index).to_pylist()], type=pa.list_(pa.float32(), 6))
        pq.write_table(table.set_column(index, "left_ee_pose", column), path)
        self._fails("schema")

    def test_privileged_transition_column_fails(self):
        path = self.episode / "transitions.parquet"
        table = pq.read_table(path)
        table = table.append_column("object_pose_world", pa.array([[0.0] * 7] * self.count))
        pq.write_table(table, path)
        self._fails("quarantine")

    def test_false_sim_only_flag_fails(self):
        self._rewrite(lambda rows: rows[4].update(sim_only=False), "sim_only.parquet")
        self._fails("quarantine")

    def test_final_next_state_corruption_fails(self):
        self._rewrite(lambda rows: rows[-1].update(next_measured_gripper_opening=[0.125, 0.5]))
        self._fails("state_chain")

    def test_bitwise_chain_rejects_signed_zero_change(self):
        def corrupt(rows):
            self.assertEqual(rows[4]["next_measured_joint_position_rad"][4], 0.0)
            rows[4]["next_measured_joint_position_rad"][4] = -0.0
        self._rewrite(corrupt)
        self._fails("state_chain")

    def test_nonmonotonic_time_fails(self):
        self._rewrite(lambda rows: rows[4].update(sim_time=rows[3]["sim_time"]))
        self._fails("time_and_keys")

    def test_duplicate_key_fails(self):
        self._rewrite(lambda rows: rows[4].update(step_index=3))
        self._fails("time_and_keys")

    def test_missing_segmentation_fails(self):
        (self.episode / f"sim_only/segmentation/head/{self.count}.png").unlink()
        self._fails("file_counts")

    def test_corrupt_state_checksum_fails(self):
        self._rewrite(lambda rows: rows[4].update(head_state_checksum="0" * 64), "sim_only.parquet")
        self._fails("state_checksums")

    def test_placeholder_extrinsics_fails(self):
        path = self.episode / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cameras"]["external"]["extrinsics"] = "pending Task 2 probe"
        path.write_text(json.dumps(manifest))
        self._fails("manifest")

    def test_stationary_controller_fails_motion(self):
        class DisconnectedEnv(MockEnv):
            def step(self, action, n_steps):
                before = self._q.copy()
                super().step(action, n_steps)
                self._q = before
                self._dq[:] = 0
        out = Path(self.work.name) / "disconnected"
        EpisodeRecorder(DisconnectedEnv(), out, {}).record(scripted_actions(16))
        report = validate_episode(out)
        self.assertTrue(any(name == "motion" and not passed for name, passed, _ in report.checks))

    def test_swapped_manifest_arm_order_fails(self):
        path = self.episode / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["joint_order"] = manifest["joint_order"][6:] + manifest["joint_order"][:6]
        path.write_text(json.dumps(manifest))
        self._fails("manifest")

    def test_swapped_measured_arm_order_fails(self):
        def corrupt(rows):
            for row in rows:
                q = row["measured_joint_position_rad"]
                row["measured_joint_position_rad"] = q[6:] + q[:6]
        self._rewrite(corrupt)
        self._fails("state_chain")

    def test_swapped_command_arm_order_fails(self):
        # Correct schema and labels cannot rescue commands routed to the wrong arm.
        def corrupt(rows):
            for row in rows:
                q = row["command_joint_position_target_rad"]
                row["command_joint_position_target_rad"] = q[5:] + q[:5]
        self._rewrite(corrupt)
        self._fails("image_content")

    def test_single_arm_only_motion_fails(self):
        for moving in range(2):
            with self.subTest(moving=ARMS[moving]):
                actions = []
                for action in scripted_actions(32):
                    targets = np.zeros(10)
                    targets[5*moving:5*moving+5] = action.joint_position_target_rad[5*moving:5*moving+5]
                    actions.append(Action(targets, action.gripper_opening))
                out = Path(self.work.name) / f"only-{moving}"
                EpisodeRecorder(MockEnv(), out, {}).record(actions)
                report = validate_episode(out)
                self.assertFalse(report.passed)
                self.assertTrue(any(name == "motion" and not ok for name, ok, _ in report.checks), report.format())
                self.assertTrue(any(name == "image_content" and ok for name, ok, _ in report.checks), report.format())

    def test_swapped_gripper_commands_fail(self):
        self._rewrite(lambda rows: [row.update(command_gripper_opening=row["command_gripper_opening"][::-1])
                                    for row in rows])
        self._fails("image_content")

    def test_gripper_only_motion_fails(self):
        out = Path(self.work.name) / "grippers-only"
        EpisodeRecorder(MockEnv(), out, {}).record(
            Action(np.zeros(10), action.gripper_opening) for action in scripted_actions(32))
        report = validate_episode(out)
        self.assertTrue(any(name == "motion" and not ok for name, ok, _ in report.checks), report.format())

    def test_swapped_base_identities_fail(self):
        path = self.episode / "manifest.json"
        manifest = json.loads(path.read_text())
        bases = manifest["scene"]["base_poses"]
        bases["left"], bases["right"] = bases["right"], bases["left"]
        path.write_text(json.dumps(manifest))
        self._fails("manifest")

    def test_wrong_head_step_index_fails(self):
        self._rewrite(lambda rows: rows[4].update(head_step_index=3))
        self._fails("step_alignment")

    def test_wrong_right_ee_pose_fails(self):
        self._rewrite(lambda rows: rows[4].update(right_ee_pose=rows[4]["left_ee_pose"]))
        self._fails("state_chain")

    def test_wrong_right_gripper_mapping_fails(self):
        path = self.episode / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["gripper_mapping"]["right"]["q_open"] = 2.0
        path.write_text(json.dumps(manifest))
        self._fails("measured_values")

    def test_extra_camera_fails(self):
        path = self.episode / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cameras"]["third"] = manifest["cameras"]["head"]
        path.write_text(json.dumps(manifest))
        self._fails("manifest")

    def test_training_loader_dimensions_and_action_order(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_smoke", Path(__file__).resolve().parents[1] / "scripts/train_smoke.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ext, head, proprio, action = module.load_episode(self.episode)
        self.assertEqual(ext.shape, (self.count, 240, 320, 3))
        self.assertEqual(head.shape, ext.shape)
        self.assertEqual(proprio.shape, (self.count, 14))
        self.assertEqual(action.shape, (self.count, 12))
        rows = pq.read_table(self.episode / "transitions.parquet").to_pylist()
        np.testing.assert_array_equal(proprio[4], rows[4]["measured_joint_position_rad"] + rows[4]["measured_gripper_opening"])
        np.testing.assert_array_equal(action[4], rows[4]["command_joint_position_target_rad"] + rows[4]["command_gripper_opening"])
        # Policy loader must not read even an unreadable privileged companion.
        (self.episode / "sim_only.parquet").write_bytes(b"unreadable privileged data")
        module.load_episode(self.episode)

    def test_cli_exit_codes(self):
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        command = [sys.executable, "-m", "avatar_harness.validate", str(self.episode)]
        good = subprocess.run(command, env=environment, text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertTrue(good.stdout.endswith("PASS\n"))
        (self.episode / "images/head/0.png").unlink()
        bad = subprocess.run(command, env=environment, text=True, capture_output=True)
        self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)
        self.assertTrue(bad.stdout.endswith("FAIL\n"))

    def test_recorder_refuses_overwrite(self):
        before = (self.episode / "transitions.parquet").read_bytes()
        with self.assertRaises(FileExistsError):
            EpisodeRecorder(MockEnv(), self.episode, {}).record(scripted_actions(16))
        self.assertEqual(before, (self.episode / "transitions.parquet").read_bytes())


class EngineBoundaryTests(unittest.TestCase):
    """Sections 3–5: temporal and quarantine behavior at the engine boundary."""

    def test_frozen_state_and_render_timing(self):
        env = MockEnv()
        env.reset(0)
        with self.assertRaises(RuntimeError):
            env.render("external")
        state = env.get_state()
        checksum = state_checksum(state)
        before = state.segmentation("external")
        with self.assertRaises(FrozenInstanceError):
            state.step_index = 9
        with self.assertRaises(ValueError):
            state.measured_joint_position_rad[0] = 1
        with self.assertRaises(ValueError):
            state.measured_joint_position_rad.setflags(write=True)
        external = env.render("external")
        head = env.render("head")
        self.assertFalse(np.array_equal(external, head))
        self.assertEqual(state_checksum(env.get_state()), checksum)
        action = Action(np.ones(10), [1.0, 1.0])
        env.step(action, 1)
        with self.assertRaises(RuntimeError):
            env.render("external")
        next_state = env.get_state()
        self.assertEqual(next_state.step_index, 1)
        self.assertEqual(next_state.sim_time, 1 / 30)
        self.assertTrue(np.all(next_state.measured_joint_position_rad[:5] > 0))
        self.assertTrue(np.all(next_state.measured_joint_position_rad[:5] < 1))
        self.assertEqual(state_checksum(state), checksum)
        np.testing.assert_array_equal(state.segmentation("external"), before)
        self.assertEqual(env.command_log_entry(action)["command_joint_position_target_rad"], [1.0] * 10)

    def test_dual_arm_fk_and_independent_control(self):
        env = MockEnv()
        env.reset(0)
        state = env.get_state()
        env.step(Action([.4, .3, .2, 0, 0] + [0] * 5, [.1, .9]), 3)
        moved = env.get_state()
        self.assertFalse(np.array_equal(moved.left_ee_pose, state.left_ee_pose))
        np.testing.assert_array_equal(moved.right_ee_pose, state.right_ee_pose)
        np.testing.assert_array_equal(moved.left_ee_pose, fake_ee_pose(moved.measured_joint_position_rad[:6]))
        np.testing.assert_array_equal(moved.right_ee_pose, fake_ee_pose(moved.measured_joint_position_rad[6:]))
        self.assertLess(moved.measured_gripper_opening[0], .5)
        self.assertGreater(moved.measured_gripper_opening[1], .5)

    def test_single_arm_actions_rejected(self):
        with self.assertRaises(ValueError):
            Action(np.ones(5), [1, 1])
        with self.assertRaises(ValueError):
            Action(np.ones(10), 1)
        with self.assertRaises(ValueError):
            Action(np.ones(10), [0, 1.1])

    def test_privileged_state_never_encoded_in_rgb(self):
        env = MockEnv()
        env.reset(0)
        state = env.get_state()
        other = replace(state, object_pose_world=[0.1, 0.2, 0.3, 0, 0, 0, 1], contact_forces=[1, 2, 3])
        self.assertNotEqual(state_checksum(state), state_checksum(other))
        for camera in CAMERAS:
            np.testing.assert_array_equal(render_snapshot(state, camera), render_snapshot(other, camera))

    def test_checksum_canonicalizes_stored_float32(self):
        env = MockEnv()
        env.reset(0)
        state = replace(env.get_state(), measured_gripper_opening=[0.1, 0.2])
        values = np.frombuffer(canonical_state_bytes(state), dtype="<f8")
        self.assertEqual(values[26], float(np.float32(0.1)))
        self.assertEqual(state_checksum(state), hashlib.sha256(values.tobytes()).hexdigest())

    def test_object_falls_and_recorder_stops_with_final_state(self):
        left_tip = fake_ee_world(np.zeros(6), "left")
        env = MockEnv(table_half_extents=(0.5, float(left_tip[1]) + 0.001), object_xy=left_tip[:2])
        action = Action([0.8, 0, 0, 0, 0] + [0] * 5, [0.5, 0.5])
        with tempfile.TemporaryDirectory() as directory:
            EpisodeRecorder(env, directory, {}).record([action] * 10)
            rows = pq.read_table(Path(directory) / "transitions.parquet").to_pylist()
            states = pq.read_table(Path(directory) / "sim_only.parquet").to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(states), 2)
            self.assertTrue(states[-1]["terminated"])
            self.assertLess(states[-1]["object_pose_world"][2], states[0]["object_pose_world"][2])
            with self.assertRaises(RuntimeError):
                env.step(action, 1)

    def test_render_that_advances_physics_is_rejected(self):
        class AdvancingEnv(MockEnv):
            def render(self, camera_name):
                pixels = super().render(camera_name)
                self.step(Action(np.ones(10), [1, 1]), 1)
                return pixels
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "changed the frozen"):
                EpisodeRecorder(AdvancingEnv(), directory, {}).record(scripted_actions(16))

    def test_commands_are_read_from_engine_log(self):
        class OffsetController(MockEnv):
            def step(self, action, n_steps):
                super().step(Action(action.joint_position_target_rad + np.float32(0.125), action.gripper_opening), n_steps)

            def command_log_entry(self, action):
                return dict(self._last_command)
        actions = list(scripted_actions(16))
        with tempfile.TemporaryDirectory() as directory:
            EpisodeRecorder(OffsetController(), directory, {}).record(actions)
            rows = pq.read_table(Path(directory) / "transitions.parquet").to_pylist()
            np.testing.assert_array_equal(rows[0]["command_joint_position_target_rad"],
                                          actions[0].joint_position_target_rad + np.float32(0.125))
            self.assertTrue(validate_episode(directory).passed)

    def test_mapping_and_frozen_contract_constants(self):
        self.assertEqual(len(JOINT_ORDER), 12)
        self.assertEqual(gripper_target(0.25, -0.2, 0.6), 0.0)
        with self.assertRaises(ValueError):
            gripper_target(1.1, 0, 1)
        with self.assertRaises(TypeError):
            CAMERAS["external"]["width_px"] = 100


if __name__ == "__main__":
    unittest.main()
