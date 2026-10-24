"""Build a kinematic one-arm/hand probe from unmodified, pinned upstream assets."""

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REACHY = ROOT / '.cache/upstream/reachy2_mujoco_assets'
SHADOW = ROOT / '.cache/upstream/mujoco_menagerie/shadow_hand'
ARM_JOINTS = ('r_shoulder_pitch', 'r_shoulder_roll', 'r_elbow_yaw',
              'r_elbow_pitch', 'r_wrist_roll', 'r_wrist_pitch', 'r_wrist_yaw')
TIPS = ('index', 'middle', 'ring', 'little', 'thumb')


def numbers(values):
    return ' '.join(format(float(x), '.12g') for x in values)


def build_scene(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = ET.parse(REACHY / 'objects/reachy2_body.xml').getroot()
    reachy_assets = ET.parse(REACHY / 'assets/reachy2_asset.xml').getroot()
    shadow = ET.parse(SHADOW / 'right_hand.xml').getroot()
    arm = deepcopy(source.find(".//body[@name='r_shoulder_base_link']"))
    arm.set('pos', numbers(config['shoulder_position_m']))
    arm.set('childclass', 'reachy')
    outlet = arm.find(".//body[@name='r_wrist_out_link']")
    outlet.remove(outlet.find("body[@name='r_hand_palm_link']"))
    palm = deepcopy(shadow.find(".//body[@name='rh_palm']"))
    palm.remove(palm.find("joint[@name='rh_WRJ1']"))
    palm.set('childclass', 'right_hand')
    palm.set('pos', numbers(config['palm_mount_position_m']))
    palm.set('quat', numbers(config['palm_mount_quaternion_wxyz']))
    outlet.append(palm)
    # The connector is a visual marker, not a designed physical adapter.
    ET.SubElement(outlet, 'geom', name='mount_marker', type='capsule',
                  fromto='0 0 0 ' + numbers(config['palm_mount_position_m']),
                  size='.016', rgba='.22 .65 .75 1', contype='0', conaffinity='0', mass='0')
    for name, body, z in [('index', 'rh_ffdistal', .026), ('middle', 'rh_mfdistal', .026),
                          ('ring', 'rh_rfdistal', .026), ('little', 'rh_lfdistal', .026),
                          ('thumb', 'rh_thdistal', .027)]:
        ET.SubElement(palm.find(f".//body[@name='{body}']"), 'site',
                      name=f'tip_{name}', pos=f'0 0 {z}', size='.003',
                      rgba='1 .55 .15 1', group='4')
    for finger, prefix in [('index', 'ff'), ('middle', 'mf'), ('ring', 'rf'), ('little', 'lf')]:
        for label, suffix in [('knuckle', 'knuckle'), ('middle', 'middle'), ('distal', 'distal')]:
            ET.SubElement(palm.find(f".//body[@name='rh_{prefix}{suffix}']"), 'site',
                          name=f'landmark_{finger}_{label}', size='.002', group='4')
    for label, body in [('knuckle', 'rh_thbase'), ('middle', 'rh_thhub'), ('distal', 'rh_thdistal')]:
        ET.SubElement(palm.find(f".//body[@name='{body}']"), 'site',
                      name=f'landmark_thumb_{label}', size='.002', group='4')
    ET.SubElement(palm, 'site', name='palm_frame', size='.004', group='4')
    ET.SubElement(outlet, 'site', name='wrist_output', size='.004', group='4')

    root = ET.Element('mujoco', model='Reachy right arm with Shadow palm — kinematic probe')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', cone='elliptic', impratio='10')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(config['render_width']), offheight=str(config['render_height']))
    ET.SubElement(visual, 'headlight', ambient='.3 .3 .3', diffuse='.7 .7 .7')
    defaults = ET.SubElement(root, 'default')
    scoped_reachy = deepcopy(reachy_assets.find('default'))
    scoped_reachy.set('class', 'reachy')
    defaults.append(scoped_reachy)
    defaults.append(deepcopy(shadow.find("default/default[@class='right_hand']")))
    assets = ET.SubElement(root, 'asset')
    ET.SubElement(assets, 'texture', name='probe_sky', type='skybox', builtin='gradient',
                  rgb1='.16 .23 .3', rgb2='.03 .05 .08', width='256', height='1536')
    for element in reachy_assets.find('asset'):
        element = deepcopy(element)
        if 'file' in element.attrib:
            element.set('file', str((REACHY / 'scenes' / element.get('file')).resolve()))
        assets.append(element)
    for element in shadow.find('asset'):
        element = deepcopy(element)
        if 'file' in element.attrib:
            element.set('file', str(SHADOW / 'assets' / element.get('file')))
        assets.append(element)
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -1 3', dir='0 0 -1')
    ET.SubElement(world, 'light', pos='-1 -2 1.8', dir='1 2 -1', diffuse='.6 .6 .6')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .1', rgba='.13 .17 .21 1')
    ET.SubElement(world, 'geom', name='mount_stand', type='cylinder', pos='0 .10 .575',
                  size='.025 .575', rgba='.3 .36 .4 1', contype='0', conaffinity='0')
    world.append(arm)
    names = {x.get('name') for x in arm.iter('body')}
    joints = {x.get('name') for x in arm.iter('joint')}
    contact = ET.SubElement(root, 'contact')
    for upstream in [reachy_assets, shadow]:
        for element in upstream.find('contact'):
            if element.get('body1') in names and element.get('body2') in names:
                contact.append(deepcopy(element))
    root.append(deepcopy(shadow.find('tendon')))
    actuators = ET.SubElement(root, 'actuator')
    for element in reachy_assets.find('actuator'):
        if element.get('joint') in joints:
            element = deepcopy(element)
            element.set('class', 'reachy')
            actuators.append(element)
    for element in shadow.find('actuator'):
        if element.get('joint') in joints or element.get('tendon'):
            actuators.append(deepcopy(element))
    # No original Reachy equality references remain: all concern the removed
    # grippers or mobile base. Shadow has tendon transmissions, no equalities.
    ET.indent(root)
    scene_path = output / 'scene.xml'
    ET.ElementTree(root).write(scene_path, encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    original_arm = mujoco.MjModel.from_xml_path(str(REACHY / 'scenes/base_scene.xml'))
    original_hand = mujoco.MjModel.from_xml_path(str(SHADOW / 'right_hand.xml'))
    for i in range(model.njnt):
        name = model.joint(i).name
        upstream = original_arm if name in ARM_JOINTS else original_hand
        np.testing.assert_allclose(model.joint(i).range, upstream.joint(name).range, atol=1e-12)
        np.testing.assert_allclose(model.joint(i).axis, upstream.joint(name).axis, atol=1e-12)
    assert model.nq == model.njnt == 29 and model.nu == 25 and model.ntendon == 4
    assert set(ARM_JOINTS).issubset(joints)
    assert not any('WRJ' in name or 'hand_finger' in name for name in joints)
    manifest = {
        'replay_mode': 'kinematic', 'source_recording': None,
        'upstream_assets': json.loads((ROOT / 'assets.lock.json').read_text()),
        'config': config, 'joints': [model.joint(i).name for i in range(model.njnt)],
        'joint_units': 'radians', 'quaternion_order': 'wxyz',
        'wrist_owner': 'Reachy r_wrist_roll/r_wrist_pitch/r_wrist_yaw',
        'mount_parent': 'r_wrist_out_link', 'mount_child': 'rh_palm',
        'palm_axes': '+Z along extended fingers; +X toward thumb; -Y toward palm face',
        'modifications': [
            'Right shoulder subtree fixed at configured world position; source orientation retained.',
            'Original Reachy gripper subtree and its actuators/equalities removed.',
            'Shadow forearm, wrist body, WRJ2 and WRJ1 joints/actuators removed.',
            'Shadow palm mounted rigidly with configured transform; finger limits/axes/tendons retained.',
            'Reachy defaults scoped under reachy; Shadow defaults remain a sibling class.',
            'Only contact exclusions with retained bodies copied; no new exclusions added.',
            'Teal spacer and stand are visual-only simulation markers; not physical adapter designs.',
            'Fingertip evaluation sites added at approximate distal-pad points; not measured human landmarks.'
        ],
        'finger_allocation': 'Curl sweeps split distal tendon sums equally. Pinch uses upstream two finger pinch keyframe allocation (not an equal-angle constraint).',
        'control': 'No physics stepping or motor tracking; state is assigned then mj_forward is called.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return model, manifest


def trajectory(model, config):
    """Deterministic scripted assembly checks, not a source recording."""
    baseline = model.qpos0.copy()
    def put(q, name, value):
        q[model.joint(name).qposadr[0]] = value
    put(baseline, 'r_elbow_pitch', -1.0)
    put(baseline, 'r_shoulder_roll', -.15)
    targets = [('open hand', baseline.copy())]
    for name, amplitude in zip(ARM_JOINTS, [.3, .1, .3, .25, .25, .25, .4]):
        for sign in (1, -1):
            q = baseline.copy()
            put(q, name, q[model.joint(name).qposadr[0]] + amplitude * sign)
            targets.append((f'arm sweep: {name}', q))
        targets.append(('return to neutral arm', baseline.copy()))
    curled = baseline.copy()
    for finger in ('FF', 'MF', 'RF', 'LF'):
        for suffix, value in [('J3', .8), ('J2', .6), ('J1', .6)]:
            put(curled, f'rh_{finger}{suffix}', value)
    for name, value in [('rh_THJ5', .17), ('rh_THJ4', .65), ('rh_THJ2', .45), ('rh_THJ1', .4)]:
        put(curled, name, value)
    targets.extend([('finger curl', curled), ('open hand', baseline.copy())])
    original_hand = mujoco.MjModel.from_xml_path(str(SHADOW / 'right_hand.xml'))
    pinch_key = ET.parse(SHADOW / 'keyframes.xml').find(".//key[@name='two finger pinch']")
    pinch_values = np.fromstring(pinch_key.get('qpos'), sep=' ')
    pinch = baseline.copy()
    for i in range(original_hand.njnt):
        name = original_hand.joint(i).name
        if 'WRJ' not in name:
            put(pinch, name, pinch_values[original_hand.jnt_qposadr[i]])
    targets.extend([('pinch posture (no object/contact task)', pinch), ('open hand', baseline.copy())])
    frames, stages = [], []
    previous = baseline
    samples = max(2, round(config['segment_seconds'] * config['sample_rate_hz']))
    for label, target in targets:
        for t in np.linspace(0, 1, samples, endpoint=False):
            alpha = t * t * (3 - 2 * t)
            frames.append(previous + alpha * (target - previous))
            stages.append(label)
        previous = target
    frames.append(previous.copy())
    stages.append('open hand')
    frames = np.array(frames)
    lower, upper = model.jnt_range.T
    if np.any(frames < lower - 1e-10) or np.any(frames > upper + 1e-10):
        raise ValueError('Scripted trajectory exceeds joint limits; refusing to clip silently')
    return frames, stages


def set_state(model, data, q, step, rate):
    data.qpos[:] = q
    data.qvel[:] = 0
    data.time = step / rate
    mujoco.mj_forward(model, data)


def diagnostics(model, data):
    contacts = []
    for c in data.contact:
        if c.dist < 0:
            contacts.append({
                'body1': model.body(model.geom_bodyid[c.geom1]).name,
                'body2': model.body(model.geom_bodyid[c.geom2]).name,
                'penetration_m': float(-c.dist),
            })
    q = data.qpos
    limit_error = np.maximum(model.jnt_range[:, 0] - q, q - model.jnt_range[:, 1])
    tips = {name: data.site(f'tip_{name}').xpos.copy() for name in TIPS}
    outlet = data.body('r_wrist_out_link')
    palm = data.body('rh_palm')
    rotation = outlet.xmat.reshape(3, 3)
    return {
        'limit_violations': int(np.sum(limit_error > 1e-9)),
        'max_penetration_m': max((c['penetration_m'] for c in contacts), default=0.0),
        'contacts': contacts,
        'thumb_index_gap_m': float(np.linalg.norm(tips['thumb'] - tips['index'])),
        'palm_mount_position_m': (rotation.T @ (palm.xpos - outlet.xpos)).tolist(),
        'palm_mount_rotation': (rotation.T @ palm.xmat.reshape(3, 3)).tolist(),
        'tip_positions_m': {name: pos.tolist() for name, pos in tips.items()},
    }
