# Build spec: v0.1 capture harness (engine-agnostic)

Build the capture harness for the v0.1 experiment contract. Read
`docs/v0.1-experiment-contract.md` FIRST — it is the law. Every field name,
type, path, and rule in this spec comes from that contract; where this spec
and the contract disagree, the contract wins.

## Deliverables (all under this repo)

1. `src/avatar_harness/__init__.py`
2. `src/avatar_harness/contract.py` — frozen constants: joint order, camera
   definitions (both, incl. intrinsics dict from contract section 2),
   physics rate, action_hold_steps, gripper mapping formula as a function.
3. `src/avatar_harness/env_api.py` — the minimal `SimEnv` Protocol the
   recorder programs against:
   - `reset(seed) -> EpisodeInfo`
   - `get_state() -> SimState` (frozen snapshot: step_index, sim_time,
     measured joint pos/vel, measured gripper opening, measured ee_pose,
     object pose, contact forces, segmentation callable)
   - `render(camera_name) -> np.ndarray uint8 [240,320,3]` — MUST be called
     only against a frozen state; implementations must not advance physics
   - `step(action, n_steps) -> None` — apply action, advance physics n_steps
   - `command_log_entry(action) -> dict` — engine-side record of what was
     actually sent (for the command!=measured rule)
4. `src/avatar_harness/mock_env.py` — a deterministic fake implementing
   SimEnv with NO GPU, NO sim dependency:
   - 6-joint arm-like state; step integrates a first-order response toward
     absolute targets (dt=1/30, gain so motion is smooth and bounded)
   - ee_pose = hand-computed 2-link planar FK (documented as fake)
   - render() returns deterministic per-(state, camera) patterns: include
     the step_index and camera name in pixel content, and a checksum-
     visible state encoding, so the validation tests can catch stale buffers
   - object falls off table edge if pushed past bounds -> episode terminates
5. `src/avatar_harness/recorder.py` — `EpisodeRecorder(env, out_dir, meta)`
   - implements the contract's timing sequence exactly (freeze -> render
     both -> read measured -> step -> record next state)
   - stamps state_checksum: sha256 of canonical bytes of the frozen state
     (float64 little-endian of all state fields) recorded per render call
   - writes EXACTLY the contract section-4 file layout: manifest.json,
     transitions.parquet, images/external/{step}.png, images/wrist/{step}.png,
     sim_only.parquet, sim_only/segmentation/{camera}/{step}.png
   - segmentation PNGs are int32-as-uint16 or labeled palette; document choice
   - manifest includes: episode_id, scene settings, joint order, gripper
     mapping, physics rate, N, full camera defs, engine name+version string,
     seed, harness git revision if available, sim_only field list
6. `src/avatar_harness/validate.py` — CLI `python -m avatar_harness.validate EPISODE_DIR`
   enforcing contract section 6 checks that don't need the live engine:
   - reload all files; row count matches PNG counts on disk
   - step alignment: external_step_index == wrist_step_index == step_index
   - chain: next_step_index(t) == step_index(t+1) AND next_measured_state(t)
     == measured_state(t+1) bitwise for float arrays
   - monotonic sim_time; unique (episode_id, step_index)
   - command fields differ from measured fields somewhere (rule: file must
     contain at least one row where |command-measured| > 1e-6 for at least
     one joint; else fail: suggests relabeling bug)
   - motion check: max-min measured range of any joint > 0.1 rad
   - schema check: all contract-named columns present, correct shapes/dtypes
   - privileged quarantine: transitions.parquet has NO object/contact/seg
     columns; sim_only.parquet has sim_only==true on every row
   - prints PASS or FAIL with per-check lines; exit code 0/1
7. `scripts/run_mock_capture.py` — scripted nonconstant command sequence
   (sine-ish sweep on 2+ joints, one gripper open/close cycle), ~300 steps,
   writes to a temp/dir given by argv, then runs validate and exits nonzero
   on FAIL.
8. `tests/test_harness.py` — stdlib unittest, no pytest dependency:
   - full mock capture -> validate PASS
   - corruption: delete a PNG -> validate FAIL
   - corruption: rewrite one external_step_index -> FAIL
   - corruption: copy measured into next_measured of the WRONG row -> FAIL
   - corruption: relabel command==measured everywhere -> FAIL
   - state_checksum determinism: same seed -> identical parquet bytes
     (except manifest git-revision field if present; compare parquet only)

## Rules

- Python 3.12 compatible. Deps: numpy, pyarrow, pillow ONLY (training/engine
  code comes later in separate envs; do not import torch/mujoco/genesis).
- No network access in library code. Tests must run offline in <60s.
- Float policy: store what the contract says (float32 for state/action
  arrays, float64 for sim_time); checksum uses float64 canonicalization of
  the values as stored.
- Keep modules small and boring; docstring each public function with the
  contract section number it implements.
- After writing everything, RUN: `./.venv-harness/bin/python scripts/run_mock_capture.py /tmp/mock_ep && ./.venv-harness/bin/python -m unittest discover -s tests -v`
  Fix until green. Report the final test output.
