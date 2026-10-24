"""Verify local MuJoCo rendering; this is not human motion replay."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image


SCENE = """
<mujoco model="render_environment_probe">
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -2 3"/>
    <geom type="plane" size="2 2 .1" rgba=".18 .22 .27 1"/>
    <body pos="0 0 .55">
      <joint name="probe_hinge" axis="0 1 0" range="-60 60"/>
      <geom type="capsule" fromto="0 0 0 0 0 .55" size=".06"
            rgba=".12 .7 .85 1"/>
      <geom type="sphere" pos="0 0 .55" size=".09" rgba="1 .5 .12 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Load an upstream XML model unchanged")
    parser.add_argument("--output", type=Path, default=Path("artifacts/render-probe"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--viewer-seconds", type=float, default=0)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.frames < 2 or not np.isfinite(args.viewer_seconds) or args.viewer_seconds < 0:
        parser.error("frames must be >= 2 and viewer-seconds must be finite and >= 0")
    if not np.isfinite(args.distance_scale) or args.distance_scale <= 0:
        parser.error("distance-scale must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    model = (mujoco.MjModel.from_xml_path(str(args.model.resolve())) if args.model
             else mujoco.MjModel.from_xml_string(SCENE))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 135
    camera.elevation = -20
    if not args.model:
        camera.lookat[:] = [0, 0, .65]
        camera.distance = 2.5
    camera.distance *= args.distance_scale
    mujoco.mj_forward(model, data)
    frame_hashes = []
    first = None
    initial_state = data.qpos.copy()
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        renderer.render()  # Warm up before timing.
        graphics = {
            name: GL.glGetString(constant).decode()
            for name, constant in [
                ("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            ]
        }
        started = time.perf_counter()
        for step in range(args.frames):
            if not args.model:
                data.qpos[0] = .7 * np.sin(2 * np.pi * step / args.frames)
            data.time = step / 30
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render().copy()
            if first is None:
                first = pixels.copy()
                initial_state = data.qpos.copy()
            frame_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
        elapsed = time.perf_counter() - started
        Image.fromarray(first).save(args.output / "first.png")
        Image.fromarray(pixels).save(args.output / "last.png")
        data.qpos[:] = initial_state
        data.time = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        replay_pixels = renderer.render()
        repeat_error = int(np.abs(replay_pixels.astype(int) - first.astype(int)).max())
    if first.max() == first.min():
        raise RuntimeError("Renderer produced a uniform image")
    if not args.model and len(set(frame_hashes)) < 2:
        raise RuntimeError("Scripted motion did not change rendered pixels")
    if repeat_error:
        raise RuntimeError(f"Repeated initial state changed pixels: max error {repeat_error}")
    viewer_frames = 0
    if args.viewer_seconds:
        from mujoco import viewer as mj_viewer

        existing_threads = set(threading.enumerate())
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer_threads = set(threading.enumerate()) - existing_threads
            with viewer.lock():
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
            started_viewer = time.perf_counter()
            while viewer.is_running() and time.perf_counter() - started_viewer < args.viewer_seconds:
                with viewer.lock():
                    if not args.model:
                        data.qpos[0] = .7 * np.sin(2 * np.pi * viewer_frames / 120)
                    data.time = viewer_frames / 30
                    mujoco.mj_forward(model, data)
                viewer.sync()
                viewer_frames += 1
                time.sleep(1 / 30)
        # close() requests shutdown asynchronously. Let the viewer finish before
        # GLFW's process-exit hook destroys its window/context on another thread.
        for thread in viewer_threads:
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("Native viewer did not finish shutdown within 10 seconds")
    report = {
        "stage": "environment rendering probe",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "replay_mode": "kinematic",
        "human_source": None,
        "robot_assembly_validated": False,
        "model": str(args.model) if args.model else "built-in scripted hinge",
        "entry_xml_sha256": hashlib.sha256(args.model.read_bytes() if args.model else SCENE.encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "requirements.lock.txt").read_bytes()).hexdigest(),
        "camera": {"lookat": camera.lookat.tolist(), "distance": camera.distance,
                   "azimuth": camera.azimuth, "elevation": camera.elevation},
        "python": platform.python_version(), "mujoco": mujoco.__version__,
        "platform": platform.platform(), "graphics": graphics,
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "resolution": [640, 480], "frames": args.frames,
        "render_loop_seconds": elapsed, "render_loop_fps": args.frames / elapsed,
        "timing_scope": "forward kinematics, rendering, readback and image hashing; excludes setup and PNG writes",
        "unique_frame_hashes": len(set(frame_hashes)),
        "repeat_initial_state_max_pixel_error": repeat_error,
        "viewer_frames": viewer_frames,
        "physics_steps": 0,
        "joints": [
            {"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
             "limited": bool(model.jnt_limited[i]), "range": model.jnt_range[i].tolist()}
            for i in range(model.njnt)
        ],
        "actuators": model.nu, "tendons": model.ntendon, "equalities": model.neq,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "model", "graphics", "frames", "render_loop_fps", "unique_frame_hashes",
        "repeat_initial_state_max_pixel_error", "viewer_frames")}, indent=2))
    print(f"Full report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
