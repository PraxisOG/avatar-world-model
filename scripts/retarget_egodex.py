"""Constrained geometric right-arm/hand retargeting; never infers motor forces."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from arm_assembly import ARM_JOINTS, set_state
from egodex_source import FINGERS, finger_channels


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError('Degenerate source geometry')
    return vector / length


def chain_positions(data, finger):
    return np.array([data.site(f'landmark_{finger}_{name}').xpos for name in ('knuckle', 'middle', 'distal')]
                    + [data.site(f'tip_{finger}').xpos])


def retarget(model, source, confidences, selection, output):
    count = len(source['camera'])
    data = mujoco.MjData(model)
    base = model.qpos0.copy()
    base[model.joint('r_elbow_pitch').qposadr[0]] = -1.0
    base[model.joint('r_shoulder_roll').qposadr[0]] = -.15
    set_state(model, data, base, 0, 30)
    robot_shoulder = data.body('r_shoulder_base_link').xpos.copy()
    source_shoulder = np.median(source['rightArm'][:, :3, 3], axis=0)
    lateral = np.median(source['rightArm'][:, :3, 3] - source['leftArm'][:, :3, 3], axis=0)
    lateral[1] = 0
    lateral = unit(lateral)
    up = np.array([0., 1., 0.])
    alignment = np.stack([unit(np.cross(up, lateral)), -lateral, up])
    assert np.linalg.det(alignment) > .999999
    world_transform = np.eye(4)
    world_transform[:3, :3] = alignment
    world_transform[:3, 3] = robot_shoulder - alignment @ source_shoulder

    # Calibrate one constant wrist-to-palm basis from tracked knuckles. Avoid
    # interpreting ARKit's wrist axes as Shadow's axes or rotating per frame to fit.
    offsets = []
    for frame in range(count):
        wrist = source['rightHand'][frame]
        z = unit(source['rightMiddleFingerKnuckle'][frame, :3, 3] - wrist[:3, 3])
        x = source['rightIndexFingerKnuckle'][frame, :3, 3] - source['rightLittleFingerKnuckle'][frame, :3, 3]
        x = unit(x - z * np.dot(x, z))
        basis = np.stack([x, np.cross(z, x), z], axis=1)
        offsets.append(wrist[:3, :3].T @ basis)
    wrist_to_palm = Rotation.from_matrix(np.array(offsets)).mean().as_matrix()
    palms = np.array([alignment @ pose[:3, :3] @ wrist_to_palm for pose in source['rightHand']])
    palm_targets = (source['rightHand'][:, :3, 3] - source_shoulder) @ alignment.T * selection['arm_scale'] + robot_shoulder
    palm_rotation = data.body('rh_palm').xmat.reshape(3, 3).copy()
    palm_position = data.body('rh_palm').xpos.copy()
    neutral_chains = {finger: (chain_positions(data, finger) - palm_position) @ palm_rotation for finger in FINGERS}
    raw_local, normalized = {}, {}
    for finger in FINGERS:
        channels = finger_channels('right', finger)
        raw_world = np.stack([source[name][:, :3, 3] for name in channels], axis=1)
        source_palms = np.array([pose[:3, :3] @ wrist_to_palm for pose in source['rightHand']])
        local = np.einsum('npi,nij->npj', raw_world - source['rightHand'][:, None, :3, 3], source_palms)
        raw_local[finger] = local
        target = np.zeros_like(local)
        # Fixed anatomical base offsets, then preserve source bone directions
        # at the robot's segment lengths. Keep raw metrics alongside these targets.
        target[:, 0] = neutral_chains[finger][0]
        lengths = np.linalg.norm(np.diff(neutral_chains[finger], axis=0), axis=1)
        directions = np.diff(local, axis=1)
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
        if np.any(norms < 1e-5):
            raise ValueError(f'Degenerate {finger} tracking')
        for j in range(3):
            target[:, j + 1] = target[:, j] + directions[:, j] / norms[:, j] * lengths[j]
        normalized[finger] = target

    arm_indices = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
    finger_indices = {}
    for finger, code in zip(FINGERS, ('FF', 'MF', 'RF', 'LF', 'TH')):
        # LFJ5 has no trustworthy metacarpal observation here. Hold it explicitly.
        suffixes = (5, 4, 3, 2, 1) if finger == 'thumb' else (4, 3, 2, 1)
        finger_indices[finger] = np.array([model.joint(f'rh_{code}J{j}').qposadr[0] for j in suffixes])
    q_history, solver_rows = [], []
    previous = base.copy()
    lower, upper = model.jnt_range.T
    started = time.perf_counter()
    for frame in range(count):
        q = previous.copy()
        def arm_residual(values):
            q[arm_indices] = values
            set_state(model, data, q, frame, 30)
            position = (data.body('rh_palm').xpos - palm_targets[frame]) / .01
            orientation = Rotation.from_matrix(palms[frame].T @ data.body('rh_palm').xmat.reshape(3, 3)).as_rotvec() / .1
            return np.r_[position, orientation, .005 * (values - previous[arm_indices])]
        seeds = [previous[arm_indices]]
        if frame == 0:
            seeds.extend([np.array([pitch, -.3, 0, -1.3, 0, 0, 0]) for pitch in (-1.0, .7)])
        results = [least_squares(arm_residual, seed, bounds=(lower[arm_indices], upper[arm_indices]),
                                max_nfev=200, ftol=1e-8, xtol=1e-8, gtol=1e-8) for seed in seeds]
        arm_fit = min(results, key=lambda result: np.linalg.norm(result.fun))
        q[arm_indices] = arm_fit.x
        evaluations = int(arm_fit.nfev)
        successes = [bool(arm_fit.success)]
        for finger in FINGERS:
            indices = finger_indices[finger]
            weights = np.sqrt([confidences[name][frame] for name in finger_channels('right', finger)])[:, None]
            def finger_residual(values):
                q[indices] = values
                set_state(model, data, q, frame, 30)
                local = (chain_positions(data, finger) - data.body('rh_palm').xpos) @ data.body('rh_palm').xmat.reshape(3, 3)
                return np.r_[((local - normalized[finger][frame]) * weights / .01).ravel(),
                             .02 * (values - previous[indices])]
            seed = previous[indices].copy()
            if frame == 0:
                # Interior initialization avoids a near-zero trust region when
                # all flexion variables begin exactly on their lower bounds.
                seed = np.maximum(seed, lower[indices] + .15 * (upper[indices] - lower[indices]))
                seed = np.minimum(seed, upper[indices] - .15 * (upper[indices] - lower[indices]))
            fit = least_squares(finger_residual, seed, bounds=(lower[indices], upper[indices]),
                                max_nfev=150, ftol=None, xtol=1e-8, gtol=1e-8)
            q[indices] = fit.x
            evaluations += int(fit.nfev)
            successes.append(bool(fit.success))
        # The human and robot have different finger widths. Refine the whole
        # hand together when independently fitted fingers overlap, retaining
        # all source-fit residuals rather than adding collision exclusions.
        hand_indices = np.concatenate(list(finger_indices.values()))
        pairs = set()
        for _ in range(3):
            set_state(model, data, q, frame, 30)
            for contact in data.contact:
                if contact.dist < 0:
                    pairs.add(tuple(sorted((int(contact.geom1), int(contact.geom2)))))
            if not pairs:
                break
            checked_pairs = sorted(pairs)
            def hand_residual(values):
                q[hand_indices] = values
                set_state(model, data, q, frame, 30)
                p = data.body('rh_palm')
                residuals = []
                for finger in FINGERS:
                    local = (chain_positions(data, finger) - p.xpos) @ p.xmat.reshape(3, 3)
                    weights = np.sqrt([confidences[name][frame] for name in finger_channels('right', finger)])[:, None]
                    residuals.extend(((local - normalized[finger][frame]) * weights / .01).ravel())
                residuals.extend(.02 * (values - previous[hand_indices]))
                residuals.extend(max(0., .0005 - mujoco.mj_geomDistance(model, data, a, b, .02, None)) / .001
                                 for a, b in checked_pairs)
                return np.array(residuals)
            refined = least_squares(hand_residual, q[hand_indices].copy(),
                                    bounds=(lower[hand_indices], upper[hand_indices]),
                                    max_nfev=150, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q[hand_indices] = refined.x
            evaluations += int(refined.nfev)
            successes.append(bool(refined.success))
            set_state(model, data, q, frame, 30)
            if not any(contact.dist < -1e-6 for contact in data.contact):
                break
        q_history.append(q.copy())
        solver_rows.append({'frame': frame, 'converged': all(successes), 'residual_evaluations': evaluations})
        previous = q.copy()
    metadata = {
        'source_to_robot_rigid_transform': world_transform.tolist(), 'arm_scale': selection['arm_scale'],
        'source_anchor': 'Median rightArm position (glenohumeral shoulder), not the medial rightShoulder/clavicle label',
        'robot_anchor': 'fixed r_shoulder_base_link', 'wrist_to_palm_rotation': wrist_to_palm.tolist(),
        'hand_root_mapping': 'Human rightHand origin maps to robot rh_palm origin; Reachy wrist output lies one mount offset proximal.',
        'finger_mapping': 'Robot-length phalanges follow source local bone directions, with fixed per-finger robot knuckle anchors. Raw unnormalized human landmarks retained for separate errors.',
        'little_metacarpal': 'LFJ5 held at zero because source metacarpal confidence is not usable; raw metacarpal data preserved.',
        'tendon_allocation': 'Middle/distal joint allocation is solved from separate source bone directions; original sum-tendon definitions and joint limits retained.',
        'temporal_method': 'Previous-frame initialization plus weak continuity penalty; no source smoothing/resampling or output clipping.',
        'elbow_target': 'Unused: source forearm confidence drops out. Redundant robot arm configurations are resolved by initialization and continuity, not human elbow labels.',
        'collision_refinement': 'If fitted fingers overlap, jointly refine finger configurations with source residuals, limits, continuity and a 0.5 mm clearance objective for colliding geom pairs. No collision exclusions added.',
        'elapsed_seconds': time.perf_counter() - started, 'solver_frames': solver_rows,
    }
    output = Path(output)
    (output / 'retargeting.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return np.array(q_history), palm_targets, palms, raw_local, normalized, metadata
