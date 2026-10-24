"""Inspect an original EgoDex recording and retain video-relative source timing."""

import hashlib
import json
from pathlib import Path

import av
import h5py
import numpy as np

FINGERS = ('index', 'middle', 'ring', 'little', 'thumb')


def finger_channels(side, finger):
    if finger == 'thumb':
        return [side + 'Thumb' + x for x in ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip')]
    return [side + finger.title() + 'Finger' + x for x in ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip')]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def load_source(root, selection, output):
    hdf5 = Path(root) / selection['source_hdf5']
    video = Path(root) / selection['source_mp4']
    start, stop = selection['source_start_frame'], selection['source_stop_frame_exclusive']
    provenance = {}
    for path in (hdf5, video):
        record = json.loads(path.with_suffix(path.suffix + '.provenance.json').read_text())
        if record['official_split'] != 'train':
            raise ValueError('This replay probe requires a training-eligible source')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Source checksum mismatch: {path}')
        if record['sha256'] != selection['expected_source_sha256'][path.suffix[1:]]:
            raise ValueError(f'Source does not match the pinned selection: {path}')
        provenance[path.suffix] = record
    if Path(provenance['.hdf5']['member']).with_suffix('.mp4').as_posix() != provenance['.mp4']['member']:
        raise ValueError('Source pose/video provenance does not refer to the same recording')
    with h5py.File(hdf5) as handle:
        transforms = {name: value[()].astype(np.float64) for name, value in handle['transforms'].items()}
        confidences = {name: value[()].astype(np.float64) for name, value in handle['confidences'].items()}
        intrinsic = handle['camera/intrinsic'][()]
        attributes = {key: json_value(value) for key, value in handle.attrs.items()}
    n = len(transforms['camera'])
    if not 0 <= start < stop <= n or stop - start < 2:
        raise ValueError('Invalid source segment')
    if not attributes.get('session_name'):
        raise ValueError('Missing session identity for split grouping')
    for name, array in confidences.items():
        if array.shape != (n,) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(f'Invalid source confidence array {name}')
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or np.any(np.diag(intrinsic) <= 0):
        raise ValueError('Invalid source camera intrinsic matrix')
    pts, timestamps, images = [], [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        video_info = {'width': stream.width, 'height': stream.height,
                      'nominal_rate': str(stream.average_rate), 'time_base': str(stream.time_base)}
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                raise ValueError('Source video has no presentation timestamp')
            pts.append(frame.pts)
            timestamps.append(float(frame.time))
            if start <= index < stop:
                image = frame.to_image()
                image.thumbnail((960, 540))
                images.append(image)
    if len(timestamps) != n:
        raise ValueError('Video and pose frame counts disagree')
    timestamps = np.array(timestamps)
    if not np.allclose(np.diff(timestamps), 1 / 30, atol=1e-7):
        raise ValueError('Unexpected frame timing; refusing assumed 30 Hz resampling')
    required = ['rightArm', 'leftArm', 'rightHand', 'leftHand']
    for side in ('right', 'left'):
        required.extend(name for finger in FINGERS for name in finger_channels(side, finger))
    for name in required:
        if name not in transforms or name not in confidences:
            raise ValueError(f'Missing required source channel {name}')
        if np.any(confidences[name][start:stop] < selection['source_confidence_min']):
            raise ValueError(f'Low source confidence in {name}; no interpolation or substitution is allowed')
    rotation_residuals = []
    for name, array in transforms.items():
        if array.shape != (n, 4, 4) or not np.isfinite(array).all():
            raise ValueError(f'Invalid source transform array {name}')
        chosen = array[start:stop]
        if not np.allclose(chosen[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(f'Invalid homogeneous transform row: {name}')
        rotations = chosen[:, :3, :3]
        residual = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
        rotation_residuals.append(residual)
        if residual > 1e-4 or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
            raise ValueError(f'Non-rigid rotation data: {name}')
    selected = {name: array[start:stop].copy() for name, array in transforms.items()}
    confidence = {name: array[start:stop].copy() for name, array in confidences.items()}
    report = {
        'dataset': 'EgoDex', 'official_split': 'train', 'recording': provenance['.hdf5']['member'],
        'session_id': attributes['session_name'], 'split_group': attributes['session_name'],
        'selection': selection, 'provenance': provenance, 'attributes': attributes,
        'full_recording_frames': n, 'selected_frames': stop - start,
        'video': video_info, 'intrinsic': intrinsic.tolist(),
        'source_start_seconds': float(timestamps[start]), 'source_last_seconds': float(timestamps[stop - 1]),
        'timestamp_source': 'Decoded MP4 PTS × time_base, matched by frame index to HDF5 under publisher convention',
        'session_absolute_frame_offset': None,
        'coordinate_frame': 'ARKit stationary session origin, right handed, Y up; transforms stored as 4x4 matrices',
        'translation_units': 'meters (ARKit convention; consistent with observed ~0.305 m upper arm and ~0.265 m forearm)',
        'pose_estimation': 'ARKit estimates, not measured human motor commands',
        'source_rotation_orthogonality_max': max(rotation_residuals),
        'source_channels': sorted(transforms),
        'confidence_summary': {name: {'minimum': float(array.min()), 'mean': float(array.mean())}
                               for name, array in confidence.items()},
        'required_confidence_minimum': min(float(confidence[name].min()) for name in required),
        'camera_confidence': None,
        'channel_motion_span_m': {name: np.ptp(array[:, :3, 3], axis=0).tolist()
                                  for name, array in selected.items()},
        'unsupported_or_unused': [
            'Head-camera and left-hand poses are preserved, but not replayed by the one-arm model.',
            'Metacarpal channels have very low constant confidences in these files; preserved but excluded from fitting.',
            'Right forearm/elbow confidence drops to zero in this segment; preserved but not used as an IK target.',
            'No source force, torque, physical contact labels, or human motor commands are supplied.',
            'Source camera confidence and absolute clip offset within its session are not supplied.',
            'No depth or lens-distortion parameters are supplied in this HDF5; source RGB is a reference, not reconstructed scene texture.'
        ],
        'missing_frame_policy': 'Reject; no interpolation, smoothing, or timestamp resampling.',
        'full_phase1_gate_passed': False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'source-inspection.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / 'original-tracking.npz',
                        **{f'transform_{key}': value for key, value in selected.items()},
                        **{f'confidence_{key}': value for key, value in confidence.items()},
                        source_frame_ids=np.arange(start, stop), source_pts=np.array(pts[start:stop]),
                        source_timestamps=timestamps[start:stop], camera_intrinsic=intrinsic)
    return selected, confidence, timestamps[start:stop], images, report
