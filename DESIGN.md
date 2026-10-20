# Avatar World Model — paired-view robot learning pipeline

Origin: drive-home conversation, 2026-10-20 (Winston + Bella).

## Thesis (v2, 2026-10-20): data-first

**The deliverable is the dataset and the capture pipeline.** Models churn;
paired observation-action data in an open, standard format (LeRobot)
appreciates regardless of which architecture wins. Prior art (Mirror
Learning, EgoExo-WM, EgoMimic, EgoDemoGen) confirms every architecture's
real bottleneck is ground-truth paired data, not model quality.

Ordering that follows:
1. Build the paired-view capture factory (sim avatar, eye-cam + external
   cam, joint-true labels, LeRobot export, clean provenance).
2. Publish data + pipeline openly. That is the contribution.
3. The world model is downstream proof: it exists to demonstrate the data
   is good, and to run the pixel-head vs latent-head shootout on identical
   data. The cooking benchmark scores data quality as much as model quality.

Why data-first is the honest difficulty ordering: building a rigorous
capture pipeline is engineering with a known ceiling; training a genuinely
good world model is research with an unknown one. Betting on the pipeline
means shipping value even if the model disappoints.

## Thesis (v1, original): world-model-first

Hardware is becoming as valuable as the software that runs on it, but the
scarce resource in robotics is *paired data*: external observations locked to
known actions. Labs doing world-model / video-pretraining work (NVIDIA GR00T,
Genie, egocentric-video research) are starving for exactly this. A workbench
data shop can sit at the frontier without a $30k humanoid.

## Core idea

An avatar-based simulator where a teleoperator drives a simulated body while
two cameras render the same ground truth:

- External view (floating camera) — resembles internet video.
- Internal view (avatar eyes) — matches the robot's own perspective.
- Every frame auto-labeled with exact joint commands. No pose estimation,
  no pseudo-action guessing, no sim-to-real label gap.

The world model learns external->internal view translation as a supervised
problem, plus estimated joint states riding along. Downstream:

1. Real-world external video -> model *hallucinates* the internal view.
2. A policy trained on internal views acts on that projection.
3. The simulator (grounded world model) becomes the cheap place to fail:
   model-based RL for recovery behaviors, dropped objects, failure modes.

## Training recipe (mirrors LLM stages)

- Pretrain: passive video, cheap, teaches physics/object permanence.
- Post-train: imitation on scarce labeled teleop/avatar demos (ACT,
  diffusion policy, VLA fine-tunes like pi-0/OpenVLA scale).
- RL: on top of the learned world model, to teach what demonstrators never
  show — failure and recovery.

## Known hard parts (design constraints, not blockers)

- Pixel ambiguity: external views hide redundant joint configs and force.
  Mitigate with fixed rig, second camera on the elbow, tactile grippers.
- Video-prediction drift: compounding error when rolling out on own
  predictions. Mitigate with short horizons + real-sensor re-grounding.
- Pixels without actions teach limited motor skill — which is why the
  paired avatar data is the artifact of value, not a stepping stone.

## Hardware path (budget-ordered)

- ~$100-250: SO-101 arm kit; x2 for leader-follower teleop (LeRobot standard).
- ~$500-1000: depth camera (RealSense/Orbbec), better grippers, second arm.
- ~$2-4k: used quadruped / open biped kits / secondhand dexterous hands.
- Standalone dexterous hands (Sharpa, BrainCo): few hundred to ~$1k each;
  two hands + camera = force-control bench rig, no legs needed.
- Future humanoid reference points (Oct 2026): Unitree H2 $29.9k (full-size,
  ~3h, 7kg arms); G1 $13.5k (2kg arms, toy-payload); EngineAI T800 ~$25k
  claimed. Base models block secondary dev — EDU tiers $40-75k are the real
  price. Wait for the price war; buy in with money AND skills.

## Compute

Moe: 4x AMD V620 GPUs. Policy fine-tunes on hours of demos = overnight jobs,
no cloud needed. Sim stack candidates: Isaac Lab or MuJoCo for avatars,
LeRobot for the real arm side, open-weights video prediction models.

## Benchmark: the cooking video test

Final-model acceptance test: feed clips from fixed-camera cooking videos
(heads-and-hands framing, repetitive manipulative tasks) and score the
model on (a) predicted internal/first-person video and (b) predicted
kinematics streams. Fixed cameras make the external view stable enough to
learn from; cooking is dense in object interactions and contact events.
Honest caveat: human hands are not robot grippers (embodiment gap), so
treat kinematics output as an embodiment-agnostic hand pose first, retargeted
to the arm later. Failure on this benchmark tells us exactly which
assumption broke.

## Next steps

- [ ] Install Genesis (AMD docker path) on Moe; run the multi-cam tutorial.
- [ ] Download EgoDex test split (16 GB) as external-data starter.
- [ ] Order SO-101 teleop pair when budget allows.
- [ ] Prototype: dual-camera (external + eye) recording in sim, dump
      LeRobot-format episodes with joint labels.
- [ ] Train small ACT/diffusion policy on sim data; transfer to real arm.
