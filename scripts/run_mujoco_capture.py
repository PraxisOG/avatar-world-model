#!/usr/bin/env python3
"""Capture and reload one dual-arm SO-101 episode using CPU physics and software EGL.

    MUJOCO_GL=egl .venv-sim/bin/python scripts/run_mujoco_capture.py /tmp/mujoco_ep1

The output directory must be empty. Linux Landlock device isolation is required;
it applies only to this process. Neither ROCm nor a GPU context is initialized.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avatar_harness.contract import ARMS, BASE_POSES, BLOCK_POS, HEAD_POS, HEAD_LOOKAT
from avatar_harness.mujoco_env import (EE_BODY, EXTERNAL_LOOKAT, EXTERNAL_POS,
                                       MERGE_METHOD, MuJoCoEnv, scripted_actions)
from avatar_harness.recorder import EpisodeRecorder
from avatar_harness.validate import validate_episode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.steps < 2 or args.seed < 0:
        parser.error("--steps must be at least 2 and --seed must be nonnegative")
    try:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise FileExistsError(f"episode directory must be empty: {args.out_dir}")
        env = MuJoCoEnv()
        EpisodeRecorder(env, args.out_dir, {"seed": args.seed, "episode_id": f"mujoco-{args.seed}"}).record(
            scripted_actions(env.joint_ranges, args.steps)
        )
        env.save_replay(args.out_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"FAIL capture: {error}", flush=True)
        return 1
    print(f"Episode: {args.out_dir.resolve()}", flush=True)
    print(f"Engine: MuJoCo {env.mj.__version__}; physics=CPU; EGL renderer={env.renderer_name}", flush=True)
    print("Device isolation: Landlock; GPU device reads/writes denied; peak VRAM=0 bytes", flush=True)
    print("Scope: paired capture and exact disk replay; ROCm training is not exercised.", flush=True)
    report = validate_episode(args.out_dir)
    print(report.format(), flush=True)
    benchmark = json.loads((args.out_dir / "benchmark.json").read_text())
    print(f"Benchmark: {benchmark['counts']['transitions']} transitions, "
          f"{benchmark['counts']['frames']} RGB frames, "
          f"{benchmark['counts']['segmentation_masks']} masks; "
          f"elapsed={benchmark['elapsed_s']:.3f} s; "
          f"frames/sec={benchmark['frames_per_sec']:.3f}; steps/sec={benchmark['steps_per_sec']:.3f}", flush=True)
    if report.passed:
        print("BEGIN DUAL-ARM AMENDMENT")
        print("Rev B, 2026-10-21; units: meters/radians; optical axes: x-right/y-down/z-forward.")
        for i, arm in enumerate(ARMS):
            position, yaw = BASE_POSES[arm]
            print(f"{arm} base_link world position={list(position)} yaw_rad={yaw}")
            print(f"{arm} gripper range_rad=[{env.q_closed[i]}, {env.q_open[i]}]; "
                  f"q_closed={env.q_closed[i]}; q_open={env.q_open[i]}")
        print(f"shared block center_world_m={list(BLOCK_POS)}")
        print(f"head identity=head parent=world pos={list(HEAD_POS)} lookat={list(HEAD_LOOKAT)}")
        for camera, definition in env.info.cameras.items():
            print(f"{camera} camera_to_world={json.dumps(definition['extrinsics'])}")
        print(f"EE body per arm={EE_BODY}; reference_frame=that arm's base_link; quaternion=xyzw")
        print(f"merge_method={MERGE_METHOD}")
        print(f"FK reach probe={json.dumps(env.info.scene['reachability_probe'])}")
        print(f"URDF sha256 (both arms)={env.info.scene['urdf_sha256']}")
        print("END DUAL-ARM AMENDMENT", flush=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
