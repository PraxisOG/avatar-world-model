"""Regression checks for model inheritance and playback time semantics."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from arm_assembly import ARM_JOINTS, REACHY, SHADOW, build_scene, trajectory
from probe_dashboard import Dashboard, Timeline


class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = json.loads((ROOT / 'config/arm_probe.json').read_text())
        cls.model, _ = build_scene(cls.config, cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_scoped_defaults_do_not_leak_between_embodiments(self):
        originals = [mujoco.MjModel.from_xml_path(str(REACHY / 'scenes/base_scene.xml')),
                     mujoco.MjModel.from_xml_path(str(SHADOW / 'right_hand.xml'))]
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            original = originals[0] if name in ARM_JOINTS else originals[1]
            source_id = original.joint(name).id
            for field in ('dof_damping', 'dof_armature', 'dof_frictionloss'):
                np.testing.assert_allclose(getattr(self.model, field)[self.model.jnt_dofadr[i]],
                                           getattr(original, field)[original.jnt_dofadr[source_id]])
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            original = originals[1] if name.startswith('rh_') else originals[0]
            source_id = original.actuator(name).id
            for field in ('actuator_ctrlrange', 'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm'):
                np.testing.assert_allclose(getattr(self.model, field)[i], getattr(original, field)[source_id])

    def test_only_one_wrist_and_no_free_base(self):
        names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.assertEqual(len(names), 29)
        self.assertFalse(any('WRJ' in name or 'hand_finger' in name for name in names))
        self.assertTrue(np.all(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
        self.assertEqual(self.model.ntendon, 4)
        self.assertEqual(self.model.neq, 0)

    def test_script_is_deterministic_and_within_original_limits(self):
        a, stages = trajectory(self.model, self.config)
        b, _ = trajectory(self.model, self.config)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a >= self.model.jnt_range[:, 0]))
        self.assertTrue(np.all(a <= self.model.jnt_range[:, 1]))
        self.assertIn('finger curl', stages)
        self.assertIn('pinch posture (no object/contact task)', stages)


class PlaybackTests(unittest.TestCase):
    def test_speed_never_changes_sample_index_and_step_pauses(self):
        timeline = Timeline(625)
        timeline.advance()
        timeline.apply({'action': 'speed', 'value': 2})
        self.assertEqual(timeline.step, 1)
        timeline.apply({'action': 'pause'})
        self.assertEqual(timeline.step, 1)
        timeline.apply({'action': 'step'})
        self.assertTrue(timeline.paused)
        self.assertEqual(timeline.step, 2)
        timeline.apply({'action': 'restart'})
        self.assertEqual((timeline.step, timeline.cycle), (0, 0))
        self.assertTrue(timeline.paused)
        with self.assertRaises(ValueError):
            timeline.apply({'action': 'speed', 'value': float('nan')})

    def test_http_validation_and_bounded_commands(self):
        dashboard = Dashboard(0)
        try:
            def post(payload, origin=None):
                headers = {'Content-Type': 'application/json'}
                if origin:
                    headers['Origin'] = origin
                return urlopen(Request(dashboard.url + '/api/control', json.dumps(payload).encode(), headers), timeout=2)
            for payload, code, origin in [({'action': 'unknown'}, 400, None),
                                          ({'action': 'play'}, 403, 'https://example.com'),
                                          ({'action': 'speed', 'value': True}, 400, None)]:
                with self.assertRaises(HTTPError) as error:
                    post(payload, origin)
                self.assertEqual(error.exception.code, code)
            for _ in range(32):
                with post({'action': 'pause'}) as response:
                    self.assertEqual(response.status, 202)
            with self.assertRaises(HTTPError) as error:
                post({'action': 'pause'})
            self.assertEqual(error.exception.code, 429)
        finally:
            dashboard.close()


if __name__ == '__main__':
    unittest.main()
