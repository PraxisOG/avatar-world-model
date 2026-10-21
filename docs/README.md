# Feasibility study and roadmap

Published to this repository on 21 October 2026. Both PDFs preserve the original
20 October 2026 reports, based on repository commit
`35824898a9220405ffa6362ec4bff810c02969f3`. They are planning documents, not
evidence that implementation, hardware validation, or model training has occurred.

| Document | Use it for |
| --- | --- |
| [Feasibility study](avatar-kinematics-feasibility-study.pdf) | Risks, design critiques, evidence, data requirements, and experiments that could justify continuing. |
| [Implementation roadmap](avatar-kinematics-roadmap.pdf) | The original six proposed milestones, their deliverables, acceptance gates, and fallback decisions. |

## Scope note from the follow-up discussion

The reports assumed one simulated SO-101-compatible arm first, followed by a
small physical-arm pilot. The later discussion shifted the preferred first scene
to a **human-proportioned upper-body avatar fixed at a table**, with a stationary
torso, one moving arm, simple fixed hand geometry, one block-pushing task, an
external camera, and a head camera. Balance, walking, grasping, and articulated
fingers were deferred.

That change is **not incorporated into these PDFs or DESIGN.md**. In particular:

- The PDFs' arm-first hour estimates are not estimates for the revised avatar
  scope. Human-avatar rigging and contact/control work need their own measurements.
- An avatar simulation result is not validation of transfer to an SO-101. A
  physical pilot still needs an appropriate simulated embodiment, action adapter,
  calibration, and real demonstrations.
- Starting with a humanoid-shaped avatar does not mean buying humanoid hardware.
  The existing hardware spending gate still applies.

[DESIGN.md](../DESIGN.md) remains the repository's design source of truth.
Reconcile the follow-up scope and the feasibility review's corrections through
a design issue or PR before using either PDF as an implementation specification,
as required by [AGENTS.md](../AGENTS.md).

## Assessment: useful guides, with limits

The strongest parts are the separation of capture engineering from research,
explicit observation/action/state timing, a direct-policy baseline, controlled
paired-view comparisons, and useful outcomes even if paired views do not help.

Use the milestones as **evidence gates**, not a delivery promise. The suggested
80% success target, 10-percentage-point gain, and trial counts are initial
engineering choices; they do not guarantee statistical significance or transfer
to reality. Report uncertainty and keep test trajectories separate from training,
including all re-rendered views of the same underlying motion.

The pipeline's learning value needs closed-loop task evaluation. Correctly
aligned recordings, attractive generated views, and large token counts each
measure something different from successful control. Scaling data also requires
a source of diverse, valid behavior; a recorder alone does not supply it.

Large-model pretraining and learned-world-model RL are not prerequisites for the
first useful result. The early milestones are intended to establish reliable
capture, loading, small-model training, and held-out behavior before larger
commitments. Treat V620 throughput, software compatibility, and delivered hardware
prices as unmeasured or date-specific until checked on the actual setup.

The PDFs' statements that no repository files were changed describe the original
report-writing session. This publication adds the reports and this index; it does
not implement the roadmap or silently adopt their proposed design changes.
