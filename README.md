# Avatar World Model

A research pipeline for turning external human video into estimated motor
kinematics and first-person video, ultimately to generate training datasets.

Start by replaying existing tracked egocentric human recordings on a simulated
upper body. Render paired external/head-camera video with joint labels, then
train a translator and test unseen movements in the same scene.

## Current status

The initial one-arm integration probe is implemented: a fixed Reachy right arm
with a Shadow palm and 22 finger joints, scripted motion, and a live local viewer.
Its 625-frame check passed on the RX 6800 with zero joint-limit violations,
zero detected contact penetration, and exact image replay at three checked states.
A first recorded-source right-arm/hand replay now passes its provisional probe
checks: 20 EgoDex frames (0.633 s), preserved video timestamps, articulated fingers,
zero joint/tendon-limit violations and zero detected penetration. Mean fingertip
error is 2.03 mm against targets adjusted for robot finger lengths, versus 23.91 mm
against original human fingertips. Both measurements are retained.
Both arms/head, simulated head POV, dataset capture, and training remain pending.
The current plan was aligned on 2026-10-22; source replay was checked on 2026-10-23.

**Phase 1:** inspect one source recording and produce one faithful head-and-hand
replay including articulated fingers, with synchronized external video,
head-mounted POV, and joint tracks. Begin with one Reachy 2 arm plus one Shadow
Hand in MuJoCo; expand to both sides and head after the integration probe.
Kinematic replay comes first; the one-arm assembly is checked only for the
documented scripted motions. Physical control is unvalidated.
**Phase 2:** build a pilot dataset and train an external-video-to-joints baseline;
render predicted POV from its motion.

## Read in this order

1. [Design](DESIGN.md): goal, scope, limitations, and source of truth.
2. [Roadmap to first training](docs/first-training-roadmap.md): gates and outputs.
3. [Phase 1 contract](docs/v0.1-experiment-contract.md): data and replay acceptance.
4. [Codex TXT handoff](docs/Avatar_Codex_Handoff.txt): implementation sequence and checks.
5. [Report index](docs/README.md): historical PDFs and research notes.

## Review questions

- Which source sample supplies synchronized head, both-hand, and finger poses with
  usable frames, calibration, tracking quality, and appropriate permissions?
- Can MuJoCo reproduce the selected recording and capture both cameras on the
  local RX 6800 development machine?
- What motion is unreachable with a fixed base, and how should it be rejected?
- How accurately can unseen movements be reconstructed from external RGB alone?
- Does rendered POV agree with the reconstructed head/hand motion?
- What additional evidence is needed before transferring to real external video?

Development target: the local Linux Mint desktop with an RX 6800 (16 GB).
Moe's four AMD V620 GPUs currently serve local LLMs and are reserved for possible
later training; begin any later training verification on one GPU.
No hardware purchase, block-pushing policy, or large-model training is required
for Phase 1. Prior dataset and software claims in references need verification.

Contributions should distinguish proposals from measured results and preserve
source provenance. See [AGENTS.md](AGENTS.md).

## Local development

Prepare and launch the recorded-motion replay:

```bash
./run-replay --prepare
./run-replay
```

Open **http://127.0.0.1:8765**. The robot replay and original video reference share
source frame indices and timestamps. Default playback is 0.25× to inspect this
short reach. The video reference is labeled separately from simulated head POV,
which is not implemented. Controls change playback only; dataset recording is off.
Stop the server with Ctrl-C. Use `--port` if another viewer is running.

Preparation fetches only the selected small training-recording pair, using bounded
HTTP ranges into the official archive; it refuses a full-archive response. It
verifies pinned source hashes, tracking quality and frame alignment, fits the
robot, then renders and validates every selected state on the RX 6800. Playback
checks code/configuration and artifact hashes; rerun preparation after changes.
See `config/source_replay.json` for selection and provisional probe thresholds.

The segment is EgoDex `part2/basic_pick_place/9`, frames 12–31 inclusive, from
0.400 to 1.033 seconds. Every selected video frame was inspected; the right hand
approaches before object contact. Original transforms, confidence, camera
intrinsics, video PTS and session identity are retained separately from robot
joint solutions. The fixed world alignment uses ARKit Y-up coordinates with no
arm scaling; finger targets preserve source phalange directions at robot lengths.
Joint limits and tendon transmissions remain intact; the four distal tendon
sums do not impose equal-angle constraints. The fit allocates the individual
finger angles explicitly and reports resulting fingertip errors.

This is **kinematic right-arm replay**, not a complete faithful upper-body replay
or physical control result. Source elbow confidence drops to zero, so elbow pose
is chosen by IK continuity rather than fitted to unreliable tracking. Low-confidence
metacarpals are excluded; the little-finger metacarpal joint is held at zero.
Head and left-hand source channels are retained but not replayed. Palm residuals
measure solver fit, not ARKit accuracy. The full Phase 1 gate remains open.

Evidence is in `artifacts/source-replay/`: `source-inspection.json`,
`original-tracking.npz`, `retargeting.json`, `manifest.json`,
`replay-trajectory.npz`, per-frame `replay-metrics.json`, `replay-validation.json`,
`comparison.png`, `tracking-errors.png`, and `replay-preview-quarter-speed.gif`.
The [recorded scalar results](docs/probes/2026-10-23-source-replay.json) distinguish
the adjusted fit from the original-human discrepancy and document unsupported channels.

The [publisher's EgoDex README](https://github.com/apple-aiml-research/ml-egodex#license)
states CC-by-NC-ND without a version. Source files and derived motion/images stay
local and Git-ignored for this noncommercial probe; public redistribution of
adaptations is not assumed to be permitted. Official evaluation recordings are
excluded. Future splits must group all clips by source session.

The original scripted assembly probe remains available:

```bash
./run-probe
```

Open the printed address, normally **http://127.0.0.1:8765**. The viewer has
play/pause, single-step, restart, and 0.25–2× playback. Restart pauses at frame
zero. Frames come from the live MuJoCo state, with a bounded latest-frame buffer;
closing the browser leaves the simulation running. Stop the server with Ctrl-C.
Head POV and human-tracking errors are explicitly unavailable. Dataset recording
is off. This is a scripted assembly probe, not a recorded human episode.

The fixed shoulder drives seven Reachy arm joints and twenty-two Shadow finger
joints. Reachy owns the wrist; Shadow's two wrist joints and forearm are removed.
A 35 mm simulation spacer connects the palm to `r_wrist_out_link`, retaining
the Shadow finger limits and four tendon transmissions. The spacer and stand are
visual markers, not physical adapter designs. Details and deviations are saved
in `artifacts/arm-probe/manifest.json`.

Run the reproducible checks without a viewer:

```bash
./run-probe --validate
.venv/bin/python -m unittest discover -s tests -v
```

Validation writes `scene.xml`, `trajectory.npz`, `samples.jsonl`,
`validation.json`, open/curl/pinch PNGs, and `assembly-preview.gif` under
`artifacts/arm-probe/`. It checks all 625 configurations, all 29 joint responses,
mount transforms, joint limits, tendon sums, contacts, cross-model collider
clearance, and three saved-state image replays. The measured minimum arm/hand
collider clearance is 9.88 mm for this sequence; it is not a guarantee for other
poses. Native collision checks retain upstream exclusions. Approximate fingertip
markers show a 14.8 mm thumb–index gap in the pinch posture, not a successful grasp.
See [the saved assembly evidence](docs/probes/2026-10-22-arm-assembly.json).

Browser control checks use the installed Chromium and an optional isolated
Playwright dependency, with the server running:

```bash
uv run --no-project --python /usr/bin/python3 --with playwright tests/check_dashboard_browser.py
```

Configuration is in `config/arm_probe.json`; use `--config`, `--output`, and
`--port` to override defaults. The default uses EGL device 1 and verifies the
renderer is the RX 6800 before running.

### Environment setup and standalone model checks

Run from the repository root. Dependencies stay in `.venv`; upstream assets,
generated images, and source data are ignored by Git. No ROS stack is required
for these initial model-loading checks.

```bash
uv venv --python /usr/bin/python3 .venv
uv pip sync --python .venv/bin/python requirements.lock.txt
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/probe_render.py --viewer-seconds 5
```

The last command renders a scripted hinge, checks image changes and deterministic
state restoration, and briefly opens the native MuJoCo viewer. It is an
environment check, not a robot replay or the planned dashboard. See the
[official MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html)
for the renderer and native viewer.

Inspect the two upstream models separately:

```bash
.venv/bin/python scripts/probe_render.py --model .cache/upstream/reachy2_mujoco_assets/scenes/base_scene.xml --output artifacts/reachy-probe --viewer-seconds 30
.venv/bin/python scripts/probe_render.py --model .cache/upstream/mujoco_menagerie/shadow_hand/scene_right.xml --output artifacts/shadow-probe --distance-scale 1.6 --viewer-seconds 30
```

These model checks use unchanged static configurations. Reports and first/last
PNG frames go under the selected output directory. Default rendering uses the
desktop display. On this machine, EGL selects the Ryzen integrated GPU by default;
device 1 selects the RX 6800. For offscreen rendering without the native viewer:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 .venv/bin/python scripts/probe_render.py --output artifacts/egl-rx6800-probe
```

Both paths were checked. Device numbering may differ on another machine or after
driver changes; inspect `graphics.renderer` in the report.

Exact upstream revisions and selected licenses are in `assets.lock.json`.
Measured initial results are in `docs/probes/`; short render timings are smoke
measurements, not dataset-generation throughput estimates.

For browser checks of the recorded replay, with `./run-replay` running:

```bash
uv run --no-project --python /usr/bin/python3 --with playwright tests/check_source_replay_browser.py
```

The scripted browser check accepts `PROBE_URL` for a separate port. Source tests
use local pinned files and skip if they have not been prepared; tests never fetch
data. Next: extend the validated assembly to the second arm and head, then check
source tracking coverage and synchronized simulated head POV before accepting a
full Phase 1 replay. Training remains deferred.
