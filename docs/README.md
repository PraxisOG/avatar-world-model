# Current plan and historical reports

## Current implementation guidance

- [DESIGN.md](../DESIGN.md): current goal and scope, updated 2026-10-22.
- [Roadmap to first training](first-training-roadmap.md): Phase 1 replay,
  then Phase 2 pilot dataset and translator training.
- [Phase 1 contract](v0.1-experiment-contract.md): source inspection,
  head/hand replay, cameras, labels, timing, and acceptance evidence.

Phase 1 uses recorded human motion on a fixed-base upper-body avatar, with a
moving head camera and fixed external camera. A contact-free motion is sufficient.
The previous SO-101, wrist-camera, and block-pushing specification is superseded.

Preferred prototype: Reachy 2 with Shadow Hands in MuJoCo. Validate one arm/hand
first, then both sides and head. Articulated finger replay is in Phase 1;
object contact and dynamic grasping remain deferred. Mark direct joint replay
as kinematic rather than claiming motor-control validation.

[Codex TXT handoff](Avatar_Codex_Handoff.txt) contains the detailed integration
checks, source links, data inspection steps, and evidence gates.

## Historical reports

| Document | How to use it now |
| --- | --- |
| [Feasibility study](avatar-kinematics-feasibility-study.pdf) | Historical risks and proposed experiments; arm-first assumptions are superseded. |
| [Original roadmap](avatar-kinematics-roadmap.pdf) | Historical six-milestone plan; use the current Markdown roadmap for implementation. |

These PDFs preserve the 20 October reports based on commit
`35824898a9220405ffa6362ec4bff810c02969f3`, published on 21 October.
They have not been rewritten. Their estimates and acceptance targets do not
apply automatically to the current embodiment or task.

The current plan preserves explicit timing, command/state separation, provenance,
held-out trajectories, measured hardware compatibility, and evidence gates.
It changes the first objective from policy control to motion reconstruction.
Closed-loop task success remains relevant for later claims about robot utility;
it is not a prerequisite for the first translator experiment.

The [reference surveys](../references/) remain research leads, not verified
dependency selections or current specifications. Dataset permissions, pose fields,
hardware prices, and software support must be checked before use.
