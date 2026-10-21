"""SO-101 CPU physics and headless software EGL, contract sections 1–6.

MuJoCo is optional and imported only on construction. Linux Landlock denies
device access before loading EGL: Mesa can otherwise probe DRM cards even with
LIBGL_ALWAYS_SOFTWARE. This restriction affects only the calling process.
EE poses use gripper_link relative to base_link, with xyzw quaternions.
"""

import atexit
import ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .contract import (CAMERAS, IMAGE_SHAPE, JOINT_ORDER, PHYSICS_DT_S,
                       PHYSICS_RATE_HZ, gripper_opening, gripper_target)
from .env_api import Action, EpisodeInfo, SimState, state_checksum

ENGINE_NAME = "avatar_harness.mujoco"
ASSETS = Path(__file__).resolve().parents[2] / "assets"
EE_BODY = "gripper_link"
EXTERNAL_POS = (0.72, -0.70, 0.52)
EXTERNAL_LOOKAT = (0.16, 0.0, 0.15)
WRIST_POS = (0.03, 0.0, -0.105)
WRIST_EULER = (0.0, 1.35, 0.0)  # MuJoCo xyz, radians, camera looks down -Z.
_CPU_READY = False


def _deny_devices():
    """Fail closed without Linux Landlock; allow only ordinary CPU /dev nodes.

    Rules handle file reads/writes, leaving directory traversal unrestricted.
    No global permissions, cgroups, GPU tools, or GPU runtimes are touched.
    """
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("CPU-only EGL requires Linux Landlock on x86_64/aarch64")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    libc = ctypes.CDLL(None, use_errno=True)

    def checked(result):
        if result < 0:
            code = ctypes.get_errno()
            raise RuntimeError(f"CPU device isolation failed: {os.strerror(code)}")
        return result

    access = (1 << 1) | (1 << 2)  # LANDLOCK_ACCESS_FS_WRITE_FILE | READ_FILE
    ruleset = Ruleset(access)
    fd = checked(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))
    try:
        paths = [p.path for p in os.scandir("/") if p.name != "dev" and not p.is_symlink()]
        paths += [f"/dev/{name}" for name in ("null", "zero", "random", "urandom", "tty", "pts", "shm")
                  if Path(f"/dev/{name}").exists()]
        for path in paths:
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                checked(libc.syscall(445, fd, 1, ctypes.byref(Rule(access, parent)), 0))
            finally:
                os.close(parent)
        checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        checked(libc.syscall(446, fd, 0))
    finally:
        os.close(fd)


def _load_mujoco():
    global _CPU_READY
    if not _CPU_READY:
        if "mujoco" in sys.modules or "OpenGL.EGL" in sys.modules:
            raise RuntimeError("construct MuJoCoEnv before importing MuJoCo/OpenGL to enforce CPU-only EGL")
        mesa = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        if not mesa.is_file():
            raise RuntimeError(f"Mesa EGL vendor configuration missing: {mesa}")
        _deny_devices()
        os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                          LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe",
                          EGL_PLATFORM="surfaceless", EGL_LOG_LEVEL="fatal",
                          __EGL_VENDOR_LIBRARY_FILENAMES=str(mesa), LP_NUM_THREADS="2")
        import mujoco
        import mujoco.egl as egl

        # Use a surfaceless display, bypassing MuJoCo's GPU device enumeration.
        display = egl.EGL.eglGetPlatformDisplayEXT(0x31DD, None, None)
        if not egl.EGL.eglInitialize(display, None, None):
            raise RuntimeError("cannot initialize software surfaceless EGL")
        egl.EGL_DISPLAY = display
        atexit.register(egl.EGL.eglTerminate, display)
        _CPU_READY = True
    import mujoco
    return mujoco


def _numbers(values):
    return " ".join(format(float(x), ".17g") for x in values)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_model(mj):
    provenance = json.loads((ASSETS / "provenance.json").read_text())
    for relative, digest in provenance["sha256"].items():
        if _sha(ASSETS / relative) != digest:
            raise ValueError(f"source asset checksum differs: {relative}")
    urdf = ET.parse(ASSETS / "so101_new_calib.urdf").getroot()
    ET.SubElement(ET.SubElement(urdf, "mujoco"), "compiler",
                  fusestatic="false", discardvisual="true")
    for mesh in urdf.iter("mesh"):
        mesh.set("filename", str((ASSETS / mesh.get("filename")).resolve()))
    source = mj.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    if tuple(source.joint(i).name for i in range(source.njnt)) != JOINT_ORDER:
        raise ValueError("compiled URDF joint order disagrees with the contract")
    # MjModel has no to_xml_string in 3.16: mj_saveLastXML is its equivalent.
    with tempfile.TemporaryDirectory(prefix="so101-mjcf-") as directory:
        path = Path(directory) / "compiled.xml"
        mj.mj_saveLastXML(str(path), source)
        root = ET.parse(path).getroot()
    ET.SubElement(root, "option", timestep=str(PHYSICS_DT_S), integrator="implicitfast",
                  gravity="0 0 -9.81", iterations="100", tolerance="1e-10")
    # Fix extent so near/far are in meters rather than inferred scene units.
    ET.SubElement(root, "statistic", extent="1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="320", offheight="240")
    ET.SubElement(visual, "map", znear="0.01", zfar="5")
    ET.SubElement(visual, "quality", offsamples="0", shadowsize="1024")
    asset = root.find("asset")
    for name, rgba in (("arm_yellow", "1 .82 .12 1"), ("motor_dark", ".1 .1 .1 1"),
                       ("table_gray", ".35 .38 .42 1"), ("block_red", ".85 .15 .1 1")):
        ET.SubElement(asset, "material", name=name, rgba=rgba)
    for index, geom in enumerate(root.findall(".//geom")):
        geom.set("name", f"arm_geom_{index}")
        color = np.fromstring(geom.attrib.pop("rgba", "1 .82 .12 1"), sep=" ")
        geom.set("material", "motor_dark" if color[0] < .2 else "arm_yellow")
        geom.set("friction", ".8 .005 .0001")
    for joint in root.findall(".//joint"):
        joint.set("damping", "0.1")
        joint.set("armature", "0.01")
    # MuJoCo does not automatically filter the fixed base's adjacent joint.
    # Its convex hull encloses the shoulder bearing, causing false contacts.
    ET.SubElement(ET.SubElement(root, "contact"), "exclude",
                  body1="base_link", body2="shoulder_link")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="plane", size=".5 .4 .01",
                  material="table_gray", friction=".8 .005 .0001")
    block = ET.SubElement(world, "body", name="block", pos=".43 .04 .015")
    ET.SubElement(block, "freejoint", name="block_free")
    ET.SubElement(block, "geom", name="block_geom", type="box", size=".015 .015 .015",
                  mass=".05", material="block_red", friction=".8 .005 .0001")
    ET.SubElement(world, "light", name="workspace_light", pos="0 -.4 1.2", dir=".1 .2 -1",
                  diffuse=".8 .8 .8", ambient=".3 .3 .3", directional="true")
    z = np.asarray(EXTERNAL_POS) - EXTERNAL_LOOKAT
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, "camera", name="external", pos=_numbers(EXTERNAL_POS),
                  xyaxes=_numbers([*x, *y]), fovy="60")
    gripper = world.find(f".//body[@name='{EE_BODY}']")
    if gripper is None:
        raise ValueError("compiled model is missing gripper_link")
    mount = ET.SubElement(gripper, "body", name="wrist_camera_mount_link", pos=_numbers(WRIST_POS))
    ET.SubElement(mount, "camera", name="wrist", euler=_numbers(WRIST_EULER), fovy="60")
    actuators = ET.SubElement(root, "actuator")
    for name in JOINT_ORDER:
        ET.SubElement(actuators, "position", name=f"{name}_position", joint=name,
                      kp="20", kv="1", ctrllimited="true",
                      ctrlrange=_numbers(source.joint(name).range))
    xml = ET.tostring(root, encoding="unicode")
    return mj.MjModel.from_xml_string(xml), xml, provenance


@dataclass(frozen=True)
class MuJoCoState(SimState):
    """Extra contact diagnostics stay in the engine's sim_only replay sidecar."""

    ncon: int = 0
    first_contact_dist: float | None = None
    first_contact_normal: tuple | None = None


class MuJoCoEnv:
    """A six-actuator SO-101 and free rigid block; no episode termination."""

    def __init__(self):
        self.mj = _load_mujoco()
        self.model, self.model_xml, self.provenance = _build_model(self.mj)
        self._bind_model()
        self._initialized = False
        self._frozen = None
        self.renderer_name = None

    def _bind_model(self):
        mj, model = self.mj, self.model
        self.data = mj.MjData(model)
        self.joint_ids = np.asarray([model.joint(name).id for name in JOINT_ORDER])
        self.qpos_indices = model.jnt_qposadr[self.joint_ids].copy()
        self.qvel_indices = model.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids = np.asarray([model.actuator(f"{name}_position").id for name in JOINT_ORDER])
        if (tuple(model.joint(int(i)).name for i in self.joint_ids) != JOINT_ORDER
                or not np.array_equal(model.actuator_trnid[self.actuator_ids, 0], self.joint_ids)
                or len(set(self.qpos_indices)) != 6 or len(set(self.qvel_indices)) != 6):
            raise ValueError("joint/actuator address mapping disagrees with contract")
        self.joint_ranges = model.jnt_range[self.joint_ids].copy()
        self.q_closed, self.q_open = self.joint_ranges[-1].tolist()
        self.ee_id = model.body(EE_BODY).id
        self.base_id = model.body("base_link").id
        self.block_id = model.body("block").id
        self.block_geom_id = model.geom("block_geom").id
        self._state_spec = mj.mjtState.mjSTATE_INTEGRATION
        self._state_size = mj.mj_stateSize(model, self._state_spec)

    def _vector(self, data=None):
        vector = np.empty(self._state_size)
        self.mj.mj_getState(self.model, self.data if data is None else data, vector, self._state_spec)
        return vector

    def _forward(self):
        # Preserve the solver input, including warm-start, for exact disk replay.
        self._forward_input = self._vector()
        self.mj.mj_forward(self.model, self.data)
        self._frozen = None

    def _cameras(self):
        result = {name: dict(camera) for name, camera in CAMERAS.items()}
        flip = np.diag([1., -1., -1.])  # MuJoCo x right/y up/-z forward -> contract axes.
        for name in result:
            cam = self.model.camera(name)
            rotation = np.empty(9)
            self.mj.mju_quat2Mat(rotation, cam.quat)
            optical = np.eye(4)
            optical[:3, :3] = rotation.reshape(3, 3) @ flip
            optical[:3, 3] = cam.pos
            mount = np.eye(4)
            if name == "wrist":
                mount[:3, 3] = self.model.body("wrist_camera_mount_link").pos
                result[name].update(mount_transform=mount.tolist(), optical_frame_offset=optical.tolist())
            result[name]["extrinsics"] = (mount @ optical).tolist()
            result[name]["calibration_source"] = "compiled MJCF; camera axes x right, y down, z forward"
        return result

    def reset(self, seed: int) -> EpisodeInfo:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        mj, model, data = self.mj, self.model, self.data
        mj.mj_resetData(model, data)
        self._step = 0
        self._last_command = None
        self._replay = []
        self._contacts = []
        self._initialized = True
        data.qpos[self.qpos_indices] = [0, 0, 0, 0, 0, gripper_target(.5, self.q_closed, self.q_open)]
        # The seed-zero acceptance scene has no jitter or domain randomization.
        block_adr = model.joint("block_free").qposadr[0]
        jitter = np.zeros(2) if seed == 0 else np.random.default_rng(seed).uniform(-.005, .005, 2)
        data.qpos[block_adr:block_adr + 2] += jitter
        data.ctrl[self.actuator_ids] = data.qpos[self.qpos_indices]
        self._forward()
        initial = self.get_state()
        # Establish and verify the renderer before promising CPU provenance.
        self.render("external")
        scene = {
            "kind": "so101_mujoco", "source": self.provenance["repository"],
            "urdf_path": self.provenance["source_path"], "urdf_revision": self.provenance["revision"],
            "urdf_sha256": self.provenance["sha256"]["so101_new_calib.urdf"],
            "asset_sha256": self.provenance["sha256"], "joint_names": list(JOINT_ORDER),
            "source_chain_endpoints": ["shoulder_link", "moving_jaw_so101_v1_link"],
            "ee_body": EE_BODY, "ee_reference_frame": "base_link", "quaternion_order": "xyzw",
            "geometry": {"arm": "source collision meshes, MuJoCo convex hull contacts",
                         "table": "plane at z=0; visual half extents .5 x .4 m",
                         "block_size_m": [.03, .03, .03]},
            "mass_kg": {model.body(i).name: float(model.body_mass[i]) for i in range(1, model.nbody)},
            "friction": {model.geom(i).name: model.geom_friction[i].tolist() for i in range(model.ngeom)},
            "materials": "flat RGBA; no textures", "domain_randomization": seed != 0,
            "block_xy_jitter_m": jitter.tolist(),
            "initial_state": {"joint_position_rad": initial.measured_joint_position_rad.tolist(),
                              "joint_velocity_rad_s": initial.measured_joint_velocity_rad_s.tolist(),
                              "object_pose_world": initial.object_pose_world.tolist()},
            "dynamics": {"integrator": "implicitfast", "position_kp": 20., "position_kv": 1.,
                         "joint_damping": .1, "joint_armature": .01, "gravity_m_s2": [0, 0, -9.81],
                         "joint_ranges_rad": self.joint_ranges.tolist(),
                         "qpos_indices": self.qpos_indices.tolist(), "qvel_indices": self.qvel_indices.tolist(),
                         "collision_exclusions": [["base_link", "shoulder_link"]],
                         "note": "integration fixture; actuator parameters are not hardware calibration"},
            "renderer": self.renderer_name, "graphics_backend": "EGL surfaceless / Mesa llvmpipe",
            "device_isolation": "Landlock read/write deny for GPU device nodes",
            "segmentation_encoding": "background=0; geom_id+1; geometry-only scene",
            "segmentation_geoms": {str(i + 1): model.geom(i).name for i in range(model.ngeom)},
        }
        self.info = EpisodeInfo(ENGINE_NAME, mj.__version__, seed, scene["urdf_revision"],
                                scene, self._cameras(), self.q_closed, self.q_open)
        return self.info

    def _pose(self, data, body):
        return np.r_[data.xpos[body], data.xquat[body][[1, 2, 3, 0]]]

    def _snapshot(self, data, step):
        mj, model = self.mj, self.model
        force = np.zeros(3)
        wrench = np.empty(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            if self.block_geom_id not in (contact.geom1, contact.geom2):
                continue
            mj.mj_contactForce(model, data, index, wrench)
            sign = 1 if contact.geom2 == self.block_geom_id else -1
            force += sign * contact.frame.reshape(3, 3).T @ wrench[:3]
        base_rot = data.xmat[self.base_id].reshape(3, 3)
        ee_rot = base_rot.T @ data.xmat[self.ee_id].reshape(3, 3)
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, ee_rot.reshape(-1))
        ee = np.r_[base_rot.T @ (data.xpos[self.ee_id] - data.xpos[self.base_id]), quat[[1, 2, 3, 0]]]
        # Own a data copy: even an old state's lazy segmentation survives stepping/reset.
        frozen_data = mj.MjData(model)
        mj.mj_copyData(frozen_data, model, data)
        masks = {}

        def segmentation(camera):
            if camera not in masks:
                labels = self._render(frozen_data, camera, segmentation=True)
                masks[camera] = labels.tobytes()
            return np.frombuffer(masks[camera], dtype=np.int32).reshape(IMAGE_SHAPE[:2])

        return MuJoCoState(
            step_index=step, sim_time=step / PHYSICS_RATE_HZ,
            measured_joint_position_rad=data.qpos[self.qpos_indices],
            measured_joint_velocity_rad_s=data.qvel[self.qvel_indices],
            measured_gripper_opening=gripper_opening(data.qpos[self.qpos_indices[-1]], self.q_closed, self.q_open),
            measured_ee_pose=ee, object_pose_world=self._pose(data, self.block_id),
            contact_forces=force, segmentation=segmentation, ncon=int(data.ncon),
            first_contact_dist=float(data.contact[0].dist) if data.ncon else None,
            first_contact_normal=tuple(data.contact[0].frame[:3]) if data.ncon else None,
        )

    def get_state(self) -> MuJoCoState:
        if not self._initialized:
            raise RuntimeError("reset before reading state")
        if self._frozen is None:
            self._frozen = self._snapshot(self.data, self._step)
            self._frozen_vector = self._vector()
            self._replay.append(self._forward_input.copy())
            self._contacts.append({"step_index": self._step, "ncon": self._frozen.ncon,
                                   "first_contact_dist_m": self._frozen.first_contact_dist,
                                   "first_contact_normal_world": self._frozen.first_contact_normal})
        if not np.array_equal(self._vector(), self._frozen_vector):
            raise RuntimeError("engine data changed outside step/reset")
        return self._frozen

    def _render(self, data, camera, *, segmentation=False):
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera: {camera}")
        before_time, before = data.time, self._vector(data)
        # Fresh Renderer/context and scene for every call; never reuse stale RGB.
        with self.mj.Renderer(self.model, height=240, width=320) as renderer:
            from OpenGL import GL
            name = GL.glGetString(GL.GL_RENDERER).decode()
            if "llvmpipe" not in name.lower():
                raise RuntimeError(f"refusing non-CPU renderer: {name}")
            self.renderer_name = name
            if segmentation:
                # 3.16 exposes this method, not Renderer(render_mode=...).
                renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render().copy()
        if data.time != before_time or not np.array_equal(before, self._vector(data)):
            raise RuntimeError("render changed the frozen simulation state/time")
        if segmentation:
            geom = pixels[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)
            return np.where(geom, pixels[..., 0] + 1, 0).astype(np.int32)
        return pixels

    def render(self, camera_name: str) -> np.ndarray:
        if self._frozen is None:
            raise RuntimeError("render requires a frozen state; call get_state first")
        self.get_state()
        return self._render(self.data, camera_name)

    def step(self, action: Action, n_steps: int) -> None:
        if not self._initialized:
            raise RuntimeError("reset before stepping")
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        targets = np.r_[action.joint_position_target_rad,
                        gripper_target(action.gripper_opening, self.q_closed, self.q_open)]
        if np.any(targets < self.joint_ranges[:, 0]) or np.any(targets > self.joint_ranges[:, 1]):
            raise ValueError("command outside actuator range; commands are never silently clipped")
        self.data.ctrl[self.actuator_ids] = targets
        self._last_command = {"command_joint_position_target_rad": action.joint_position_target_rad.tolist(),
                              "command_gripper_opening": action.gripper_opening}
        self._frozen = None
        for _ in range(n_steps):
            self.mj.mj_step(self.model, self.data)
            self._step += 1
        if not math.isclose(self.data.time, self._step / PHYSICS_RATE_HZ, abs_tol=1e-12, rel_tol=0):
            raise RuntimeError("MuJoCo time drift/reset detected")
        if np.any(self.data.warning.number):
            raise RuntimeError(f"MuJoCo physics warning: {self.data.warning.number.tolist()}")
        # mj_step integrates qpos after its last FK; refresh derived poses before rendering.
        self._forward()

    def command_log_entry(self, action: Action) -> dict:
        expected = {"command_joint_position_target_rad": action.joint_position_target_rad.tolist(),
                    "command_gripper_opening": action.gripper_opening}
        if self._last_command != expected:
            raise ValueError("requested command is not the most recently applied command")
        return expected

    def save_replay(self, out_dir):
        """Persist full float64 engine inputs for exact re-render, quarantined from policy data."""
        root = Path(out_dir)
        folder = root / "sim_only" / "mujoco"
        folder.mkdir()
        self.mj.mj_saveModel(self.model, str(folder / "model.mjb"), None)
        np.savez(folder / "states.npz", integration=np.asarray(self._replay),
                 state_spec=np.int64(self._state_spec))
        (folder / "contacts.json").write_text(json.dumps(self._contacts, indent=2) + "\n")
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        files = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(folder.iterdir())}
        manifest["mujoco_replay"] = {"sim_only": True, "files": files,
                                     "state_encoding": "float64 mjSTATE_INTEGRATION before mj_forward",
                                     "segmentation_encoding": self.info.scene["segmentation_encoding"]}
        for relative in files:
            manifest["sim_only"]["files"][relative] = {"sim_only": True}
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        benchmark_path = root / "benchmark.json"
        benchmark = json.loads(benchmark_path.read_text())
        benchmark["scope"] = ("reset through initial manifest write, including RGB, segmentation and Parquet; "
                              "excludes replay sidecar export and validation")
        benchmark_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True, allow_nan=False) + "\n")


def scripted_actions(joint_ranges, n_steps=150):
    """Three arm sine sweeps and one gripper opening cycle, inside compiled limits.

    Keep 10% gripper margin: MuJoCo joint limits are soft constraints, so driving
    exactly onto a limit can put measured normalized opening outside [0, 1].
    """
    if n_steps < 2:
        raise ValueError("the sweep needs at least two steps")
    ranges = np.asarray(joint_ranges)
    center = np.zeros(5)
    amplitude = np.minimum([.55, .30, .25, 0., 0.], .25 * np.diff(ranges[:5], axis=1)[:, 0])
    for index in range(n_steps):
        phase = 2 * math.pi * index / (n_steps - 1)
        yield Action(center + amplitude * np.sin([phase, 2 * phase, phase, 0, 0]),
                     .5 - .4 * math.cos(phase))


def verify_replay(root, manifest, rows, states):
    """Reload the compiled model and every solver input; compare state, masks and RGB."""
    from PIL import Image
    mj = _load_mujoco()
    if manifest["engine_version"] != mj.__version__:
        raise ValueError("MuJoCo replay requires the recorded engine version")
    declaration = manifest["mujoco_replay"]
    if declaration["sim_only"] is not True:
        raise ValueError("MuJoCo replay must be quarantined")
    for relative, digest in declaration["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to((root / "sim_only").resolve()) or _sha(path) != digest:
            raise ValueError(f"MuJoCo replay file mismatch: {relative}")
        if manifest["sim_only"]["files"][relative]["sim_only"] is not True:
            raise ValueError("unflagged replay file")
    folder = root / "sim_only" / "mujoco"
    env = MuJoCoEnv.__new__(MuJoCoEnv)
    env.mj = mj
    env.model = mj.MjModel.from_binary_path(str(folder / "model.mjb"))
    env._bind_model()
    if env._cameras() != manifest["cameras"]:
        # JSON converts tuples to lists in the fixed intrinsics.
        if json.loads(json.dumps(env._cameras())) != manifest["cameras"]:
            raise ValueError("replay model camera calibration differs from manifest")
    with np.load(folder / "states.npz", allow_pickle=False) as archive:
        vectors = archive["integration"]
        if (int(archive["state_spec"]) != int(env._state_spec)
                or vectors.shape != (len(states), env._state_size) or not np.isfinite(vectors).all()):
            raise ValueError("invalid MuJoCo integration state archive")
    contacts = json.loads((folder / "contacts.json").read_text())
    if len(contacts) != len(states):
        raise ValueError("contact diagnostic count differs")
    for index, (vector, recorded) in enumerate(zip(vectors, states)):
        mj.mj_resetData(env.model, env.data)
        mj.mj_setState(env.model, env.data, vector, env._state_spec)
        if not math.isclose(env.data.time, index / PHYSICS_RATE_HZ, abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"MuJoCo replay time differs at {index}")
        mj.mj_forward(env.model, env.data)
        snapshot = env._snapshot(env.data, index)
        if state_checksum(snapshot) != recorded["state_checksum"]:
            raise ValueError(f"MuJoCo state replay differs at {index}")
        diagnostic = {"step_index": index, "ncon": snapshot.ncon,
                      "first_contact_dist_m": snapshot.first_contact_dist,
                      "first_contact_normal_world": list(snapshot.first_contact_normal) if snapshot.ncon else None}
        if diagnostic != contacts[index]:
            raise ValueError(f"MuJoCo contact replay differs at {index}")
        for camera in CAMERAS:
            with Image.open(root / recorded["segmentation_paths"][camera]) as image:
                if not np.array_equal(np.asarray(image), snapshot.segmentation(camera)):
                    raise ValueError(f"{camera} segmentation replay differs at {index}")
            if index < len(rows):
                rgb = env._render(env.data, camera)
                if hashlib.sha256(rgb.tobytes()).hexdigest() != recorded[f"{camera}_rgb_sha256"]:
                    raise ValueError(f"{camera} replay differs at {index}; stale or duplicated render buffer")
    return "all PNGs decoded; MuJoCo state/contact/RGB/segmentation re-render exactly matches reloaded state"
