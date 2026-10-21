# Build spec: MuJoCo engine adapter + first real capture (v0.1 Tasks 3-4 prep)

Extend the capture harness with a MuJoCo SimEnv implementation and capture a
real episode. Read docs/v0.1-experiment-contract.md (the law) and the existing
src/avatar_harness/*.py first. MuJoCo 3.16 is in .venv-sim. Working probes on
this host: MUJOCO_GL=egl, mujoco.Renderer(model, height, width),
r.update_scene(data, camera=...), r.render(); physics via mj_step/mj_forward.

## Scene (contract section 1)

- Download the SO-101 URDF:
  https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf
  (save under assets/, record its sha256 in the manifest)
- Compile it with mujoco.MjModel.from_xml_path or from_xml_string (URDF OK),
  via an MJCF post-compile edit (to_xml_string + ET) to add:
  - a plane geom (floor/table) and one 3cm block box geom on the table,
    flat colors via material rgba
  - light source
  - camera 'external': fixed world body, 320x240, fov 60, position chosen so
    the whole arm+table is in frame; choose concrete numbers, lookat center
    of workspace. Record the numeric transform -> this resolves contract
    'pending Task 2 probe' for the external camera
  - camera 'wrist': child body of gripper_link (verify the exact body name in
    the compiled model; contract says wrist_camera_mount_link attaches to
    gripper_link in the camera URDF variant, so mounting on gripper_link is
    correct), offset in front of the gripper facing forward, 320x240 fov 60.
    Record its numeric transform relative to gripper_link
- Actuators: position actuators for the 5 arm joints (ky=1 or similar,
  ctrllimited to joint range) + gripper. Action per contract: first 5 entries
  absolute radian targets, entry 6 = gripper opening 0..1 mapped via the
  contract gripper mapping formula (read joint ranges from the compiled model
  for q_closed/q_open).
- Joint ordering: resolve names shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll, gripper to qpos/adr indices by name lookup from the
  compiled model. Assert order matches contract at init.

## SimEnv implementation (src/avatar_harness/mujoco_env.py, .venv-sim only)

- Implements env_api.SimEnv exactly like mock_env does. get_state returns
  step_index, sim_time=step*dt, measured pos/vel (qpos/qvel in contract
  order), gripper opening (inverse-mapped), ee_pose = xpos+xquat of the
  gripper body (document the body name as the EE frame), object pose of the
  block, contact info (ncon + first contact dist/normal if any),
  segmentation via Renderer(render_mode=mjrm_SEGMENTATION).
- render(camera): fresh mujoco.Renderer call against CURRENT frozen state;
  never steps. Use the same data buffer; assert data.time unchanged before
  and after both renders.
- step(action, n): set ctrl from action, mj_step n times.
- Termination: none needed for smoke; scripted ~150 steps.
- Determinism: reset(seed) reseeds block position jitter deterministically
  (seed=0 -> exact fixed layout so checksum chain is reproducible).

## Scripts

- scripts/run_mujoco_capture.py OUT_DIR [--steps 150]: scripted nonconstant
  sweep (same style as run_mock_capture: sine sweeps 2+ joints, one gripper
  cycle, amplitudes within ctrlrange), record via EpisodeRecorder, then run
  the validator (avatar_harness.validate) and exit nonzero on FAIL.
- scripts/resume_backend_check.py — NO, skip this, do not create; backend
  handling stays with the agent, not the repo.

## Contract amendment artifact

After a PASSING capture, print (stdout, clearly delimited) the two numeric
camera transforms you chose (external: pos+lookat or euler in world; wrist:
pos+euler relative to gripper_link) plus the EE body name and gripper joint
range numbers, so the contract doc can be amended with real values.

## Rules

- Only .venv-sim python for running; library code imports mujoco lazily
  inside mujoco_env.py so .venv-harness tests are unaffected.
- Run with MUJOCO_GL=egl. CPU physics only; do not initialize any GPU/hip
  context (vLLM is live on those cards; do not touch GPU at all).
- After writing, run the full chain and paste outputs:
  1) MUJOCO_GL=egl .venv-sim/bin/python scripts/run_mujoco_capture.py /tmp/mujoco_ep1
  2) .venv-harness/bin/python -m unittest discover -s tests  (must stay 30 OK)
- Do not git commit.
