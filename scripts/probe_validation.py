"""Measured checks for the synthetic arm assembly, separate from human replay."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image

from arm_assembly import ARM_JOINTS, ROOT, TIPS, diagnostics, set_state


def make_camera(config):
    camera = mujoco.MjvCamera()
    for key, value in config['camera'].items():
        if key == 'lookat':
            camera.lookat[:] = value
        else:
            setattr(camera, key, value)
    return camera


def graphics_info():
    return {key: GL.glGetString(value).decode() for key, value in
            [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}


def attachment_clearance(model, data):
    """Check cross-embodiment collider distances, including filtered pairs."""
    arm, hand = [], []
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        name = model.body(model.geom_bodyid[i]).name
        if name.startswith('r_'):
            arm.append(i)
        elif name.startswith('rh_'):
            hand.append(i)
    best = .1
    pair = None
    for a in arm:
        for b in hand:
            distance = mujoco.mj_geomDistance(model, data, a, b, .1, None)
            if distance < best:
                best = float(distance)
                pair = [model.body(model.geom_bodyid[i]).name for i in (a, b)]
    return best, pair


def joint_response(model, baseline, rate):
    data = mujoco.MjData(model)
    set_state(model, data, baseline, 0, rate)
    reference = data.site_xpos.copy()
    reference_rotation = data.site_xmat.copy()
    results = {}
    for i in range(model.njnt):
        name = model.joint(i).name
        q = baseline.copy()
        q[i] += .03 if q[i] + .03 <= model.jnt_range[i, 1] else -.03
        set_state(model, data, q, 0, rate)
        movements = np.linalg.norm(data.site_xpos - reference, axis=1)
        rotation_change = float(np.max(np.abs(data.site_xmat - reference_rotation)))
        results[name] = {'max_site_position_change_m': float(movements.max()),
                         'max_site_rotation_matrix_change': rotation_change}
        # A straight thumb can rotate about its long axis without moving its tip.
        if movements.max() < 1e-7 and rotation_change < 1e-7:
            raise AssertionError(f'{name} did not move any evaluation site')
    return results


def validate(model, config, frames, stages, output):
    output = Path(output)
    data = mujoco.MjData(model)
    camera = make_camera(config)
    n = len(frames)
    actual = np.empty_like(frames)
    tips = np.empty((n, 5, 3))
    tendons = np.empty((n, 4))
    hashes, preview, samples = [], [], []
    checks = {0, n // 2, n - 1}
    saved_images = {}
    keyframes = {'open': 0,
                 'curl': stages.index('finger curl') + round(config['segment_seconds'] * config['sample_rate_hz']) - 1,
                 'pinch': stages.index('pinch posture (no object/contact task)') + round(config['segment_seconds'] * config['sample_rate_hz']) - 1}
    started = time.perf_counter()
    with mujoco.Renderer(model, config['render_height'], config['render_width']) as renderer:
        graphics = graphics_info()
        if config['expected_renderer'] not in graphics['renderer']:
            raise RuntimeError(f"Expected {config['expected_renderer']}; got {graphics['renderer']}")
        for step, q in enumerate(frames):
            set_state(model, data, q, step, config['sample_rate_hz'])
            actual[step] = data.qpos.copy()
            tips[step] = [data.site(f'tip_{name}').xpos for name in TIPS]
            tendons[step] = data.ten_length.copy()
            row = diagnostics(model, data)
            distance, pair = attachment_clearance(model, data)
            row.update(step=step, time=step / config['sample_rate_hz'], stage=stages[step],
                       arm_hand_clearance_m=distance, closest_arm_hand_bodies=pair)
            samples.append(row)
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render().copy()
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            if step in checks:
                saved_images[step] = pixels.copy()
            for name, index in keyframes.items():
                if step == index:
                    Image.fromarray(pixels).save(output / f'{name}.png')
            if step % 6 == 0:
                preview.append(Image.fromarray(pixels).resize((720, 480)))
        elapsed = time.perf_counter() - started
        # Reload the generated model and saved numeric trajectory, not just IDs.
        np.savez_compressed(output / 'trajectory.npz', targets=frames, actual=actual,
                            timestamps=np.arange(n) / config['sample_rate_hz'],
                            fingertips_m=tips, tendon_lengths_rad=tendons,
                            joint_names=np.array([model.joint(i).name for i in range(model.njnt)]))
        reloaded = np.load(output / 'trajectory.npz')
        replay_model = mujoco.MjModel.from_xml_path(str(output / 'scene.xml'))
        replay_data = mujoco.MjData(replay_model)
        replay_errors = []
        with mujoco.Renderer(replay_model, config['render_height'], config['render_width']) as replay_renderer:
            for step in sorted(checks):
                set_state(replay_model, replay_data, reloaded['actual'][step], step, config['sample_rate_hz'])
                replay_renderer.update_scene(replay_data, camera=camera)
                pixels = replay_renderer.render()
                replay_errors.append(int(np.abs(pixels.astype(int) - saved_images[step].astype(int)).max()))
        reloaded.close()
    preview[0].save(output / 'assembly-preview.gif', save_all=True, append_images=preview[1:],
                    duration=200, loop=0)
    with (output / 'samples.jsonl').open('w') as stream:
        for row in samples:
            stream.write(json.dumps(row) + '\n')
    position_errors = [np.linalg.norm(np.array(row['palm_mount_position_m']) - config['palm_mount_position_m']) for row in samples]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.array(config['palm_mount_quaternion_wxyz']))
    rotation_errors = [np.max(np.abs(np.array(row['palm_mount_rotation']) - rotation.reshape(3, 3))) for row in samples]
    tendon_errors = []
    for i, finger in enumerate(('FF', 'MF', 'RF', 'LF')):
        joint_ids = [model.joint(f'rh_{finger}J{suffix}').qposadr[0] for suffix in (2, 1)]
        tendon_errors.append(float(np.max(np.abs(tendons[:, i] - actual[:, joint_ids].sum(axis=1)))))
    response = joint_response(model, frames[0], config['sample_rate_hz'])
    report = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Synthetic one-arm assembly probe only. No human replay, motor tracking, or physical grasping validated.',
        'replay_mode': 'kinematic', 'mujoco': mujoco.__version__, 'graphics': graphics,
        'config': config, 'frames': n, 'duration_source_seconds': None,
        'duration_scripted_seconds': (n - 1) / config['sample_rate_hz'],
        'validation_elapsed_seconds': elapsed, 'validation_frames_per_second': n / elapsed,
        'timing_scope': 'Forward kinematics, collision/distance diagnostics, rendering, hashing and sparse preview PNG saves; not a production capture benchmark.',
        'joint_count': model.njnt, 'arm_joints': list(ARM_JOINTS), 'finger_joint_count': 22,
        'actuator_count': model.nu, 'tendon_count': model.ntendon,
        'joint_limit_violations': sum(row['limit_violations'] for row in samples),
        'max_native_contact_penetration_m': max(row['max_penetration_m'] for row in samples),
        'minimum_arm_hand_collider_clearance_m': min(row['arm_hand_clearance_m'] for row in samples),
        'mount_position_residual_max_m': float(max(position_errors)),
        'mount_rotation_matrix_residual_max': float(max(rotation_errors)),
        'tendon_sum_residual_max_rad': max(tendon_errors),
        'thumb_index_gap_open_m': samples[0]['thumb_index_gap_m'],
        'thumb_index_gap_pinch_m': samples[keyframes['pinch']]['thumb_index_gap_m'],
        'per_joint_site_response_for_0_03_rad': response,
        'fingertip_motion_span_m': {name: np.ptp(tips[:, i], axis=0).tolist() for i, name in enumerate(TIPS)},
        'unique_rendered_frames': len(set(hashes)), 'reload_max_pixel_errors': replay_errors,
        'source_tracking_metrics': None,
        'acceptance_tolerances': {'penetration_m': config['penetration_tolerance_m'],
                                  'mount_position_m': 1e-10, 'mount_rotation_matrix': 1e-10,
                                  'tendon_sum_rad': 1e-10, 'replay_pixel_error': 0},
        'tolerance_rationale': 'Assembly numerical checks use floating point tolerances; penetration budget is a conservative 1 mm probe criterion, not a source replay or physical safety threshold.',
        'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                        for p in ('scripts/arm_assembly.py', 'scripts/probe_validation.py', 'config/arm_probe.json', 'assets.lock.json', 'requirements.lock.txt')},
    }
    report['passed'] = bool(
        report['joint_limit_violations'] == 0 and
        report['max_native_contact_penetration_m'] <= config['penetration_tolerance_m'] and
        report['minimum_arm_hand_collider_clearance_m'] >= -config['penetration_tolerance_m'] and
        max(position_errors) < 1e-10 and max(rotation_errors) < 1e-10 and
        max(tendon_errors) < 1e-10 and max(replay_errors) == 0 and len(set(hashes)) > 10)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError('Assembly probe failed; see validation.json for measured failures')
    return report
