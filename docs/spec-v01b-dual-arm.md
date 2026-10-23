# Spec: v0.1-b dual-arm amendment ("two hands")

Operator decision 2026-10-21: target embodiment is BIMANUAL — two SO-101
arms at one table — to match the bimanual composition of the ego-manipulation
dataset corpus (Open-AoE hand=left/right/both labels, EgoDex two 25-joint
hands). Amend contract + harness + engine + training path coherently. Do not
keep single-arm compat shims; one schema, dual-arm.

## Contract amendment (docs/v0.1-experiment-contract.md)

- Edit in place; add a revision line: "Rev B, 2026-10-21: dual-arm embodiment
  per operator decision (dataset alignment). Rev A single-arm capture remains
  valid as harness evidence (tag it in text)."
- Scene: two identical SO-101 instances (same sha256-pinned URDF), base bodies
  side by side on the table facing it, separated so workspaces overlap the
  central block region (pick concrete base poses, e.g. x=-0.28 and x=+0.28,
  yawed inward ~30 deg; verify both arms reach the block with a quick
  fk-based probe before fixing numbers). One shared block, one table.
- Joint ordering (contract law): left arm first in Rev A order
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper],
  then right arm in the same order. 12 entries total.
- Action: 10 absolute radian targets + 2 gripper openings (left then right).
  Measured: pos/vel 12 each + 2 gripper openings. EE pose: TWO fields,
  left_ee_pose and right_ee_pose (7 each, xyzw, base_link frame of that arm).
- Cameras: EXACTLY two streams (contract invariant), but ego view changes:
  - external: unchanged (already-verified matrix stays).
  - wrist -> head: NEW camera 'head', mounted on a fixed mast/body above and
    behind both arm bases, looking down at the central workspace (like a
    head-mounted ego cam over the bench). NOT attached to either arm.
    Choose concrete pos/lookat so both grippers and the block are usually in
    frame at rest AND mid-sweep; record numeric matrix; identity: head;
    parent_frame: world. The wrist-camera concept moves to out-of-scope note
    (still fine for later robot-wrist work, not this contract).
- Field renames in schema table: wrist_path->head_path,
  wrist_step_index->head_step_index, images/wrist/->images/head/,
  command/measured arrays widened to 12-dim, plus left/right EE fields.
- Pass criteria unchanged except motion check: range>0.1 rad required on at
  least one joint OF EACH ARM.

## Code changes

- src/avatar_harness/contract.py: JOINT_ORDER 12 entries (prefix 'L/' 'R/'
  style is fine internally; record both raw URDF names and disambiguated
  names), CAMERAS external+head with matrices, gripper mapping per arm,
  PHYSICS settings unchanged.
- mock_env.py: 12-joint mock, two 2-link FK chains (left/right EE poses),
  head camera pattern deterministic from state. Update its unit tests.
- mujoco_env.py: instantiate two copies of the pinned URDF (mjcf merge:
  compile each arm once, splice bodies under namespaces left_arm/ right_arm
  — MuJoCo 3.16 has mujoco.mj_appendModel or do XML-level merging with
  element name prefixes; pick the approach that keeps qpos slices contiguous
  per arm and document it). Add head camera body. All existing validator
  behaviors must still hold (checksums, segmentation for both cameras,
  command!=measured, quarantine).
- run_mujoco_capture.py: scripted sweep must move BOTH arms (nonconstant,
  opposite-phase sines so they don't collide; gripper cycle both arms).
- validate.py: update column names/dims for 12-dim schema, head_* fields,
  two EE poses, per-arm motion check. Keep every check it already had.
- scripts/train_smoke.py: proprio 14-dim (12 pos + 2 openings), action 12;
  image streams head+external. Keep CPU path working.
- tests: full mock capture PASS; all corruption cases still FAIL; add:
  swapped-arm-order corruption -> FAIL; single-arm-only motion -> FAIL.

## Verification chain (run all, paste outputs; EGL headless, CPU physics,
## do NOT touch GPU — vLLM is live)

1. .venv-harness/bin/python -m unittest discover -s tests   (all green)
2. MUJOCO_GL=egl .venv-sim/bin/python scripts/run_mujoco_capture.py /tmp/dual_ep1  (PASS)
3. CUDA_VISIBLE_DEVICES="" .venv-train/bin/python scripts/train_smoke.py /tmp/dual_ep1 --device cpu --warm-steps 30  (TRAIN_SMOKE_OK)
4. Print delimited block: BEGIN DUAL-ARM AMENDMENT ... final base poses, head
   camera matrix, per-arm gripper ranges, merge method chosen, END

Do not git commit. Iterate until all three commands pass.
