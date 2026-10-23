> Historical research notes (2026-10-20), not the current implementation plan.
> Follow [DESIGN.md](../DESIGN.md), the [current roadmap](../docs/first-training-roadmap.md),
> and the [Phase 1 contract](../docs/v0.1-experiment-contract.md).
> Dataset, license, prior-art, and software claims below have not been reverified
> in the 2026-10-22 planning update. Recommendations below are superseded where
> they conflict with source-motion replay and external-video reconstruction.

# Prior art: third-person → first-person view transformation for robot learning

Question: has someone already done "watch external video, synthesize the
internal view + infer kinematics, train a policy on it"? Short answer:
**yes, actively, several groups, 2019→now — and it got a major validation
in mid-2026.** Our angle (sim-generated paired data as the foundation,
sub-$500 hardware) is still a real niche.

## Direct hits (external→internal video + action inference)

- **Mirror Learning** (arXiv:2607.28737, Jul 2026) — THE close one.
  Fine-tuned Cosmos video diffusion ("Mirror Video Model") transforms
  third-person observer video into the demonstrator's first-person view;
  an inverse dynamics head (from VaVAM) infers actions in the learner's
  control space; behavior-clones from the resulting "mirror data."
  Key result for us: **view-synthesis model fine-tuned ONLY on CARLA sim
  transferred zero-shot to real robotaxi footage**, even hallucinating
  correct occluded content. Sim-trained perspective transforms generalize
  to real video — the load-bearing assumption of our pipeline, validated.
  Domain: driving, not manipulation.
- **EgoExo-WM** (arXiv:2605.15477, May 2026) — "Unlocking Exo Video for
  Ego World Models." Exo→ego video diffusion used to convert internet
  exo video into egocentric *training data*, then an action-conditioned
  latent world model in DINOv3 feature space. Almost exactly our
  "manufacture the internal view" thesis, feature-space flavored.
- **EgoDemoGen** (arXiv:2509.22578, Sep 2025) — generates paired
  demonstrations from novel ego viewpoints: trajectory transfer via IK
  + feasibility filtering, video diffusion for the visual side. Notably
  trained on ~600 teleoperated episodes on 4 GPUs. Manipulation domain.
- **FVTrans + RANet + IDNet** (Robotics & Autonomous Systems 2023) —
  flow-based viewpoint transformation of third-person human demos into
  the robot's fixed execution view, then inverse dynamics outputs
  **joint angles**. Pre-diffusion, GAN/flow era; same three-stage shape.
- **Decoupled hierarchical controller** (NeurIPS 2019, arXiv:1911.09676)
  — cGAN generates first-person visual sub-goals from third-person
  human video; low-level controller outputs actions. Real Baxter robot.
  The original "imagine it from your own view" formulation.

## Adjacent / competing orthodoxy (read before defending pixel prediction)

- **Latent world-action models** argue *against* predicting pixels:
  Being-H0.7 (2605.00078), EgoWAM (2607.08436), LAC-WM (2610.10846),
  V-JEPA 2 line. Claim: predict compact latent "what matters for action"
  instead of video; pixel rollouts are expensive and a shortcut risk.
  Survey: "Robotic Video World Models" (2601.07823) and
  world-action-models/awesome-world-action-models on GitHub.
  **Our position to defend:** the video head is a supervision/alignment
  device (dense signal that forces physically consistent latents), and
  the kinematics head is the product. Latent-only variants are the main
  scientific risk to the design — worth a cheap A/B early (video head
  vs latent head, same data).

## What's still open / our niche

1. Nobody's demoed the **paired-view simulator as a data factory** end to
   end: avatar teleop, eye-cam + external cam from one ground truth,
   LeRobot export, sub-$150 arms. Existing work starts from expensive
   diffusion finetunes or 4xH20 rigs; none grounds the transform in
   sim-native paired capture at hobby scale.
2. **Cooking-video benchmark** as an acceptance test: EgoExo-WM uses exo
   video but no public pass/fail benchmark of the kind we specified.
3. **Embodiment-agnostic kinematics head + retargeting** to cheap arms:
   EgoDemoGen does IK for viewpoint transfer, not human→SO-101 retarget.
4. The scaling argument stands: our contribution would be showing the
   transform is *learnable from synthetic paired capture on 4 old AMD
   GPUs* — a reproducibility/data-shop contribution, not a SOTA claim.

## Implication

The idea is validated enough to de-risk the core bet (Mirror Learning's
sim→real zero-shot is the strongest evidence), and crowded enough that
the *dataset + open pipeline* is the contribution. Move accordingly:
lean into "open, reproducible, paired-view data factory" positioning.
