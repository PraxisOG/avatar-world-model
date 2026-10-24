"""Prepare and verify a right-arm replay from one original EgoDex recording."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from arm_assembly import ROOT, build_scene, diagnostics, set_state
from egodex_source import FINGERS, load_source
from probe_validation import attachment_clearance, graphics_info, make_camera
from retarget_egodex import chain_positions, retarget

CODE_FILES = ('scripts/source_replay.py', 'scripts/egodex_source.py', 'scripts/retarget_egodex.py',
              'scripts/arm_assembly.py', 'config/arm_probe.json', 'config/source_replay.json',
              'assets.lock.json', 'requirements.lock.txt')


def fingerprints():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}


def prepare(selection, config, output):
    output = Path(output)
    model, manifest = build_scene(config, output)
    source, confidence, times, images, inspection = load_source(ROOT, selection, output)
    q, palm_targets, rotation_targets, raw_local, normalized, metadata = retarget(model, source, confidence, selection, output)
    data = mujoco.MjData(model)
    n = len(q)
    camera = make_camera(config)
    actual, actual_palms, actual_tips = [], [], []
    raw_targets, normalized_targets = [], []
    rows, hashes, previews, frame_images = [], [], [], {}
    (output / 'external').mkdir(exist_ok=True)
    (output / 'source-reference').mkdir(exist_ok=True)
    with mujoco.Renderer(model, config['render_height'], config['render_width']) as renderer:
        graphics = graphics_info()
        if config['expected_renderer'] not in graphics['renderer']:
            raise RuntimeError(f"Wrong renderer: {graphics['renderer']}")
        for frame in range(n):
            set_state(model, data, q[frame], frame, config['sample_rate_hz'])
            palm = data.body('rh_palm')
            actual.append(data.qpos.copy())
            actual_palms.append(palm.xpos.copy())
            tips = np.array([chain_positions(data, finger)[-1] for finger in FINGERS])
            raw_world = np.array([raw_local[finger][frame, -1] @ rotation_targets[frame].T + palm_targets[frame] for finger in FINGERS])
            fit_world = np.array([normalized[finger][frame, -1] @ rotation_targets[frame].T + palm_targets[frame] for finger in FINGERS])
            actual_tips.append(tips)
            raw_targets.append(raw_world)
            normalized_targets.append(fit_world)
            row = diagnostics(model, data)
            row['source_frame_id'] = selection['source_start_frame'] + frame
            row['source_timestamp'] = float(times[frame])
            row['simulation_timestamp'] = float(times[frame] - times[0])
            row['palm_position_error_m'] = float(np.linalg.norm(palm.xpos - palm_targets[frame]))
            row['palm_orientation_error_deg'] = float(np.linalg.norm(Rotation.from_matrix(rotation_targets[frame].T @ palm.xmat.reshape(3, 3)).as_rotvec()) * 180 / np.pi)
            row['normalized_fingertip_errors_m'] = np.linalg.norm(tips - fit_world, axis=1).tolist()
            row['raw_human_fingertip_errors_m'] = np.linalg.norm(tips - raw_world, axis=1).tolist()
            row['arm_hand_clearance_m'], row['closest_arm_hand_bodies'] = attachment_clearance(model, data)
            row['solver_converged'] = metadata['solver_frames'][frame]['converged']
            row['tendon_lengths_rad'] = data.ten_length.tolist()
            tendon_actuators = [model.actuator(f'rh_A_{code}J0').id for code in ('FF', 'MF', 'RF', 'LF')]
            row['tendon_control_range_violations'] = int(sum(
                not model.actuator_ctrlrange[a, 0] - 1e-8 <= data.ten_length[j] <= model.actuator_ctrlrange[a, 1] + 1e-8
                for j, a in enumerate(tendon_actuators)))
            rows.append(row)
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render().copy()
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            Image.fromarray(pixels).save(output / 'external' / f'{frame:05d}.png')
            images[frame].save(output / 'source-reference' / f'{frame:05d}.jpg', quality=90)
            if frame in (0, n // 2, n - 1):
                frame_images[frame] = pixels
            canvas = Image.new('RGB', (1440, 520), '#101820')
            external = Image.fromarray(pixels).resize((720, 480))
            reference = images[frame].copy()
            reference.thumbnail((720, 480))
            canvas.paste(external, (0, 40))
            canvas.paste(reference, (720, 40 + (480 - reference.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 12), 'Robot replay | kinematic | 0.25x preview', fill='white')
            draw.text((736, 12), f'Original source POV | frame {row["source_frame_id"]} | {times[frame]:.3f}s', fill='white')
            previews.append(canvas)
    actual = np.array(actual)
    actual_palms = np.array(actual_palms)
    actual_tips = np.array(actual_tips)
    raw_targets = np.array(raw_targets)
    normalized_targets = np.array(normalized_targets)
    np.savez_compressed(output / 'replay-trajectory.npz', solved_configurations=q, actual_joint_positions=actual,
                        source_frame_ids=np.arange(selection['source_start_frame'], selection['source_stop_frame_exclusive']),
                        source_timestamps=times, simulation_timestamps=times - times[0],
                        palm_targets_m=palm_targets, palm_rotation_targets=rotation_targets,
                        actual_palms_m=actual_palms, actual_fingertips_m=actual_tips,
                        original_human_fingertips_m=raw_targets, normalized_fingertip_targets_m=normalized_targets,
                        joint_names=np.array([model.joint(i).name for i in range(model.njnt)]))
    previews[0].save(output / 'replay-preview-quarter-speed.gif', save_all=True,
                     append_images=previews[1:], duration=133, loop=0)
    previews[n // 2].save(output / 'comparison.png')
    # Fresh model + serialized simulator-read states, with image comparison.
    replay_model = mujoco.MjModel.from_xml_path(str(output / 'scene.xml'))
    replay_data = mujoco.MjData(replay_model)
    replay_errors = []
    with np.load(output / 'replay-trajectory.npz') as saved:
        with mujoco.Renderer(replay_model, config['render_height'], config['render_width']) as renderer:
            for frame, expected in frame_images.items():
                set_state(replay_model, replay_data, saved['actual_joint_positions'][frame], frame, config['sample_rate_hz'])
                renderer.update_scene(replay_data, camera=camera)
                replay_errors.append(int(np.abs(renderer.render().astype(int) - expected.astype(int)).max()))
    normalized_errors = np.linalg.norm(actual_tips - normalized_targets, axis=2)
    raw_errors = np.linalg.norm(actual_tips - raw_targets, axis=2)
    accept = selection['acceptance']
    joint_steps = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    valid = []
    for i, row in enumerate(rows):
        failures = []
        for key, threshold in [('palm_position_error_m', accept['palm_position_max_m']),
                               ('palm_orientation_error_deg', accept['palm_orientation_max_deg']),
                               ('max_penetration_m', accept['penetration_max_m'])]:
            if row[key] > threshold:
                failures.append(key)
        if normalized_errors[i].mean() > accept['normalized_fingertip_mean_max_m'] or normalized_errors[i].max() > accept['normalized_fingertip_worst_max_m']:
            failures.append('normalized_fingertip_error')
        if i and joint_steps[i - 1] > accept['joint_step_max_rad']:
            failures.append('joint_discontinuity')
        if row['arm_hand_clearance_m'] < -accept['penetration_max_m']:
            failures.append('arm_hand_collision')
        if row['limit_violations'] or row['tendon_control_range_violations']:
            failures.append('joint_or_tendon_limit')
        if not row['solver_converged']:
            failures.append('solver_not_converged')
        row['rejection_reasons'] = failures
        row['accepted'] = not failures
        valid.append(not failures)
    report = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'One right-arm/hand kinematic replay of a short real-source segment. Full Phase 1, left/head replay, motor control, and training remain incomplete.',
        'recording': inspection['recording'], 'session_id': inspection['session_id'],
        'source_hashes': selection['expected_source_sha256'], 'official_split': 'train',
        'source_start_frame': selection['source_start_frame'], 'source_stop_frame_exclusive': selection['source_stop_frame_exclusive'],
        'frames': n, 'duration_seconds': float(times[-1] - times[0]), 'source_pts_verified': True,
        'minimum_consumed_source_confidence': inspection['required_confidence_minimum'],
        'palm_position_error_max_m': max(row['palm_position_error_m'] for row in rows),
        'palm_orientation_error_max_deg': max(row['palm_orientation_error_deg'] for row in rows),
        'normalized_fingertip_mean_error_m': float(normalized_errors.mean()),
        'normalized_fingertip_max_error_m': float(normalized_errors.max()),
        'raw_human_fingertip_mean_error_m': float(raw_errors.mean()),
        'raw_human_fingertip_max_error_m': float(raw_errors.max()),
        'per_finger_errors': {finger: {'normalized_mean_m': float(normalized_errors[:, j].mean()),
                                       'raw_mean_m': float(raw_errors[:, j].mean())} for j, finger in enumerate(FINGERS)},
        'joint_limit_violations': sum(row['limit_violations'] for row in rows),
        'tendon_range_violations': sum(row['tendon_control_range_violations'] for row in rows),
        'max_penetration_m': max(row['max_penetration_m'] for row in rows),
        'minimum_arm_hand_clearance_m': min(row['arm_hand_clearance_m'] for row in rows),
        'max_joint_step_rad': float(joint_steps.max()),
        'per_joint_motion_span_rad': dict(zip([model.joint(i).name for i in range(model.njnt)], np.ptp(q, axis=0).tolist())),
        'source_head_translation_from_initial_max_m': float(np.linalg.norm(source['camera'][:, :3, 3] - source['camera'][0, :3, 3], axis=1).max()),
        'head_and_left_replay': 'Not implemented; source channels retained unchanged',
        'physics_steps': 0, 'replay_mode': 'kinematic', 'graphics': graphics,
        'unique_external_frames': len(set(hashes)), 'reload_max_pixel_errors': replay_errors,
        'accepted_frames': sum(valid), 'rejected_frames': n - sum(valid),
        'acceptance': accept, 'acceptance_scope': selection['acceptance_scope'],
        'passed': bool(all(valid) and max(replay_errors) == 0 and len(set(hashes)) == n),
        'full_phase1_gate_passed': False, 'code_sha256': fingerprints(),
    }
    manifest.update(source_recording=inspection['recording'], source_session=inspection['session_id'],
                    source_provenance=inspection['provenance'], source_start_frame=selection['source_start_frame'],
                    source_stop_frame_exclusive=selection['source_stop_frame_exclusive'],
                    retargeting=metadata, camera_intrinsic_source=inspection['intrinsic'],
                    source_timing=inspection['timestamp_source'], source_finger_data='original-tracking.npz',
                    finger_allocation=metadata['tendon_allocation'], full_phase1_gate_passed=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'replay-metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    artifact_names = ['scene.xml', 'manifest.json', 'replay-trajectory.npz', 'replay-metrics.json',
                      'original-tracking.npz', 'source-inspection.json']
    artifact_names.extend(f'source-reference/{i:05d}.jpg' for i in range(n))
    report['artifact_sha256'] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifact_names}
    (output / 'replay-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    for axis, label in enumerate('xyz'):
        axes[0].plot(times, palm_targets[:, axis], label=f'target {label}')
        axes[0].plot(times, actual_palms[:, axis], '--', label=f'robot {label}')
    axes[0].set(ylabel='Palm position (m)', title='Recorded-source target and robot palm (fixed world alignment)')
    axes[0].legend(ncol=3)
    axes[1].plot(times, normalized_errors.mean(axis=1) * 1000, label='Robot-length target error')
    axes[1].plot(times, raw_errors.mean(axis=1) * 1000, label='Original human fingertip error')
    axes[1].set(ylabel='Mean fingertip error (mm)', title='Morphology-adjusted fit and original-source discrepancy')
    axes[1].legend()
    for finger, code in zip(FINGERS, ('FF', 'MF', 'RF', 'LF', 'TH')):
        name = f'rh_{code}J3' if finger != 'thumb' else 'rh_THJ4'
        axes[2].plot(times, q[:, model.joint(name).qposadr[0]], label=finger)
    axes[2].set(xlabel='Original video timestamp (s)', ylabel='Flexion joint (rad)', title='Articulated finger motion')
    axes[2].legend(ncol=5)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(output / 'tracking-errors.png', dpi=150)
    plt.close(figure)
    print(json.dumps({key: report[key] for key in ('passed', 'frames', 'duration_seconds',
        'palm_position_error_max_m', 'palm_orientation_error_max_deg', 'normalized_fingertip_mean_error_m',
        'normalized_fingertip_max_error_m', 'raw_human_fingertip_mean_error_m', 'max_penetration_m',
        'max_joint_step_rad', 'accepted_frames', 'rejected_frames')}, indent=2))
    if not report['passed']:
        raise RuntimeError(f'Replay did not pass the configured probe checks; inspect {output}/replay-metrics.json')
    return report


def load_replay(output):
    output = Path(output)
    report = json.loads((output / 'replay-validation.json').read_text())
    if not report['passed']:
        raise ValueError('Replay has failed validation; refusing to label it accepted')
    if report['code_sha256'] != fingerprints():
        raise ValueError('Code/configuration changed; rerun ./run-replay --prepare')
    for name, expected in report['artifact_sha256'].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Replay artifact checksum mismatch: {name}')
    manifest = json.loads((output / 'manifest.json').read_text())
    with np.load(output / 'replay-trajectory.npz') as saved:
        frames = saved['actual_joint_positions'].copy()
        times = saved['source_timestamps'].copy()
        ids = saved['source_frame_ids'].copy()
    metrics = json.loads((output / 'replay-metrics.json').read_text())
    if len(frames) != len(metrics) or len(frames) != len(times):
        raise ValueError('Replay arrays and metrics disagree')
    import base64
    images = [base64.b64encode((output / 'source-reference' / f'{i:05d}.jpg').read_bytes()).decode() for i in range(len(frames))]
    replay = {'recording': report['recording'], 'source_timestamps': times, 'source_frame_ids': ids,
              'reference_images': images, 'metrics': metrics, 'report': report}
    model = mujoco.MjModel.from_xml_path(str(output / 'scene.xml'))
    return model, manifest['config'], frames, replay
