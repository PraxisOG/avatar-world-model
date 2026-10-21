# Avatar World Model

A paired-view data pipeline for robot learning. The premise: the scarce
resource in robotics isn't model size or GPU hours, it's *paired data* —
observations locked to known actions. This project builds a workbench-scale
machine for manufacturing that data.

**Read [DESIGN.md](DESIGN.md) first.** It contains the full thesis, the
training recipe, known hard parts, and the hardware path.

## The idea in one paragraph

A teleoperator drives an avatar in simulation while two cameras render the
same ground truth: an external view (resembles internet video) and an
internal, first-person view (matches a robot's own eyes). Every frame is
auto-labeled with exact joint commands — no pose estimation, no pseudo-action
inference. A world model trained on this learns external→internal view
translation as supervised learning, which then lets it project real external
video into the internal view a trained policy can act on. The simulator,
grounded in that paired data, becomes the cheap place to fail a million times:
model-based RL for recovery behaviors that human demonstrators never produce.

## Status

Design phase. No code yet. Compute home is a 4-GPU local box (AMD V620 x4).
Hardware target starts at a two-arm SO-101 teleop kit, not a humanoid.

## Open questions for reviewers and agents

We want critiques, alternatives, and prior-art pointers — not cheerleading.

1. **Sim stack.** Isaac Lab vs MuJoCo (and mujoco_playground / dm_control)
   vs SAPIEN vs Gazebo. The hard requirement is synchronized multi-camera
   RGB at the same timestep as the action stream, plus LeRobot-compatible
   episode export. What actually works on AMD GPUs (ROCm), and where will
   we hit walls? Note: Isaac stack is CUDA-first — is that a blocker or is
   the CPU/physics side enough?

2. **Pixel ambiguity.** An external view admits redundant joint
   configurations and hides contact force. Is a fixed-rig second camera
   enough, or does the label set need proprioception as a first-class
   output? Anyone solved redundancy-to-prediction cleanly?

3. **Prediction drift.** Rolling a video world model on its own predictions
   degrades fast. Short horizons + re-grounding is our plan. Better
   architectures? Diffusion vs autoregressive token video at this scale?

4. **Is translation the right intermediate task?** We claim
   external→internal projection is the load-bearing skill. What evidence
   (for or against) exists in the cross-view / egocentric-transfer
   literature? Key prior art we should be reading: GR00T, Genie/Genie 2,
   Dreamer v3, the latent-action-from-video line, anything on
   viewpoint-equivariant representation learning.

5. **Episode format.** LeRobot dataset format for both sim and real arms —
   right choice? What breaks when you mix 30 fps sim episodes with 10–30 fps
   real teleop at different camera intrinsics?

6. **Does any of this survive contact with a $150 arm?** Cheapest honest
   path from "works in the paired-view sim" to "moves the real gripper
   tonight." What will the sim-to-real tax actually cost here?

7. **Data value.** If paired external/internal/action triples are the
   artifact, what makes a dataset publishable or shareable? Licensing,
   docs, provenance — what would make *you* trust and use a stranger's
   teleop dump?

## How to contribute

File an issue with the label `review` for critiques, `prior-art` for
pointers we've missed, or `design` for proposals. Design-doc changes should
arrive as an issue or PR with reasoning attached; this doc is the
single source of truth for the thesis.

## Feasibility study and implementation roadmap

The [report index](docs/README.md) links the feasibility study and six-milestone
roadmap, with a scope note explaining the later fixed-upper-body-avatar discussion.
The PDFs preserve their original arm-first assumptions and estimates; read the
scope note before implementation. These are proposed plans, not completed work.
