# Avatar World Model

Updated 2026-10-22. This is the current design source of truth.
Implementation status: planning; no replay pipeline or training result has been demonstrated.

## Goal

Build a dataset generator that takes external human video, eventually including
YouTube footage, and estimates corresponding motor kinematics and first-person
video. The generated examples should support downstream robot learning.
Robot policy training is a downstream evaluation, not the first deliverable.

The immediate route is to replay existing egocentric recordings with tracked head
and hands on a simulated body. Render synchronized external and head-mounted
camera views and record the body's joint motion. Train a translator on these
paired examples, then test entirely unseen source movements in the same scene.

## Phase 1: one faithful replay

Inspect a small source-data sample, build one human-proportioned upper body with
a fixed base, map recorded head and both hand poses onto its joints, and record
one short replay from a fixed external camera and a moving head-mounted camera.
Include articulated finger motion, initially without object contact.
The preferred prototype is Reachy 2 with left/right Shadow Hands in MuJoCo.
First validate one arm and one hand, then add the second side and head.
The combined model is untested; verify attachment frames, joint coupling, limits,
licenses, collisions, and the wrist boundary rather than duplicating wrist joints.
No block-pushing policy, SO-101 hardware, walking, or physical grasping is required. A fixed base limits admissible head translation and arm reach; reject
or explicitly document unsupported motion rather than silently changing labels.

See [the roadmap](docs/first-training-roadmap.md) and
[the Phase 1 contract](docs/v0.1-experiment-contract.md).

## Data and labels

Source tracking is input to the replay pipeline, not input to the translator.
External source video is not required: the simulator generates that view.
The original egocentric video is a reference; synthetic POV is not expected to
reproduce its original objects, textures, or lighting.

Retargeted joint targets are one chosen embodiment-specific solution.
Start with kinematic replay and label episodes `replay_mode=kinematic`.
Record source head/hand/finger poses, solved configurations, and simulator-read
state separately. Assigned configurations matching read state are not evidence
of motor tracking. In dynamic mode, record motor targets and actual state separately.
Do not fabricate force or torque labels from kinematic replay.
Neither inferred human joints nor retargeted commands are uniquely determined
ground truth about the original human's motor commands.

At the first training milestone, predict actual joint trajectories from external
RGB clips. Replay those predictions to render POV. Predicting commanded targets
is a separately labeled experiment, since hidden controller dynamics can make
them ambiguous from images. Direct neural POV generation remains a later goal.

## First training experiment

One body, one scene, fixed external camera, head-mounted POV camera.
Split whole source recordings before clipping or augmentation; group every
derivative of a recording into the same split. Where recordings share a session
or continuous take, group those together too.

Use a modest image encoder and temporal model. Overfit one short sequence first,
then train a pilot dataset and evaluate held-out movements. Compare joint error,
head/hand and fingertip pose error, jitter, limit violations, and rendered POV against reference.
Compare constant-pose and nearest-training-motion baselines.
A small loss or attractive image alone is not evidence of useful reconstruction.

## Preserved research direction

The paired-data pipeline remains a reusable deliverable. LeRobot-compatible
export, explicit calibration, provenance, and clean observation/action timing
remain goals. Raw validated episodes come before format conversion.
Later experiments can compare direct POV prediction against rendering predicted
motion, add scene and object reconstruction, vary cameras and appearance, and
evaluate real external video. A cooking-video benchmark is a long-term direction,
not a Phase 1 acceptance test.

Synthetic success does not establish real-video transfer, force recovery, or
physical robot competence. Occlusion admits multiple solutions; generated
real-video labels need uncertainty estimates and rejection criteria.
A kinematic replay simulator is not automatically a learned world model.
Policy learning, physical-arm pilots, large-model pretraining, and learned-world-
model RL are deferred until evidence supports them.

## Compute and choices still to validate

Moe has four AMD V620 GPUs, historically operated at 150 W per GPU.
Begin with one GPU for training verification. Benchmark actual software and
rendering support; do not promise overnight training or assume CUDA compatibility.
MuJoCo is the first probe target for the existing Reachy and Shadow assets,
not a validated installation. Genesis remains an alternative if needed.
See [the Codex handoff](docs/Avatar_Codex_Handoff.txt) for sources and integration checks.

EgoDex is an existing candidate in the reference notes, not a verified selection.
Inspect actual pose fields, calibration, timestamps, units, access conditions,
and permitted derivative use before choosing a source. Do not use an official
test split for training. No hardware purchase is needed for Phase 1.

## Document precedence

This design defines the goal; the current roadmap defines ordering; the Phase 1
contract defines capture acceptance. Older PDFs and reference surveys are
historical research inputs, not current implementation instructions. Their prices,
compatibility statements, dataset details, and training estimates require
verification before use. Previous design versions remain available in Git history.
