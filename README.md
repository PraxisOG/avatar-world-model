# Avatar World Model

A research pipeline for turning external human video into estimated motor
kinematics and first-person video, ultimately to generate training datasets.

Start by replaying existing tracked egocentric human recordings on a simulated
upper body. Render paired external/head-camera video with joint labels, then
train a translator and test unseen movements in the same scene.

## Current status

Planning stage. No implemented replay pipeline or completed training run is
present in this repository. The current plan was aligned on 2026-10-22.

**Phase 1:** inspect one source recording and produce one faithful head-and-hand
replay including articulated fingers, with synchronized external video,
head-mounted POV, and joint tracks. Begin with one Reachy 2 arm plus one Shadow
Hand in MuJoCo; expand to both sides and head after the integration probe.
Kinematic replay comes first; the combined model is not yet validated.
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
- Which simulator can replay the selected body and capture both cameras on Moe?
- What motion is unreachable with a fixed base, and how should it be rejected?
- How accurately can unseen movements be reconstructed from external RGB alone?
- Does rendered POV agree with the reconstructed head/hand motion?
- What additional evidence is needed before transferring to real external video?

Compute target: four AMD V620 GPUs; begin verification on one GPU.
No hardware purchase, block-pushing policy, or large-model training is required
for Phase 1. Prior dataset and software claims in references need verification.

Contributions should distinguish proposals from measured results and preserve
source provenance. See [AGENTS.md](AGENTS.md).
