# Roadmap to the first training run

Agreed 2026-10-22. The initial one-arm assembly probe now has
[measured evidence](probes/2026-10-22-arm-assembly.json) and a live local viewer.
It passed the documented scripted sweeps. On 2026-10-23, source inspection and a
20-frame right-arm/hand kinematic EgoDex replay passed provisional one-arm checks;
see [measured source-replay evidence](probes/2026-10-23-source-replay.json).
Mean adjusted fingertip fit is 2.03 mm; original-human discrepancy is 23.91 mm.
Original timing, transforms and confidence are retained. Source elbow tracking is
unusable in this segment and is excluded from the fit; head and left-hand motion
are retained but not replayed. This does not complete the full Phase 1 gate.
Both sides/head, simulated head POV, full camera capture and training remain pending.
Goal: external video of an unseen movement -> joint trajectory + consistent POV,
initially in one fixed simulated scene.

## Phase 1 — validate one source recording and produce one faithful replay

0. Probe one Reachy 2 arm with one Shadow Hand in MuJoCo. Verify model loading,
   attachment orientation, joint coupling, limits, collisions, and wrist ownership.
   Pin upstream assets and licenses. Scripted sweeps may verify assembly but do
   not satisfy the human-replay gate. This step passed the documented assembly probe.
1. Inspect one small sample from an eligible egocentric motion dataset.
   Document head, both wrists/hands, and individual finger pose fields, timestamp alignment,
   coordinate frames, scale, calibration, tracking validity, source split,
   provenance, and permitted use. Choose a short contact-free segment.
2. Probe the simulator on the actual machine: one fixed-base human-proportioned
   upper body, articulated arms, head and fingers, plain scene, fixed
   external camera, and head-mounted POV camera. Record versions and throughput.
3. Map tracked motion to the body with continuous joint configurations.
   Begin with one arm/hand, then include both sides and head. Use kinematic replay
   initially and mark replay_mode=kinematic. Respect finger coupling and record
   thumb/fingertip errors; do not equate assigned state with motor tracking. Preserve source
   timestamps and transforms. Log residuals, unreachable poses, joint limits,
   and any filtering. Unsupported source channels must be explicit.
4. Record synchronized images, targets, and actual joint states. Reload the
   episode and verify replay, calibration, and timing.

Deliverables: source inspection report, reproducible scene and replay command,
one validated episode, side-by-side replay preview, and measured error report.
Acceptance is specified in [the contract](v0.1-experiment-contract.md).
Stop here to review the evidence before expanding scope.

## Phase 2 — pilot dataset and first training run

5. Process several distinct source recordings in the same scene. Split by
   recording/session before making clips; all views and augmentations stay with
   their parent recording. Keep official evaluation data out of training.
6. Implement an external-RGB-only translator loader and modest image encoder
   plus temporal model predicting actual joint positions over time.
   Deliberately overfit one short training sequence using the real saved loader.
   Check finite losses, improving reconstruction, and replay of its predictions.
7. Train on the pilot set, choose settings on validation data, and evaluate the
   held-out set once settings are fixed. Render POV from predicted joint motion.
   Save checkpoint, configuration, seeds, split manifest, losses, metrics,
   hardware/software versions, throughput, and comparison videos.

Report joint-angle error, head/hand position and orientation error, fingertip error, temporal
jitter, joint-limit violations, and POV consistency against reference. Compare
constant-pose and nearest-training-motion baselines with the same inputs.
Set numerical acceptance thresholds using Phase 1 validation measurements before
examining test results. A finite-loss smoke test is not a successful translator.

This baseline produces POV through the simulator; it does not demonstrate direct
neural video generation or generalization to YouTube footage.

## Later, only after review

Direct learned POV generation; larger or varied scenes; object reconstruction
and contact; appearance/camera variation; real-video evaluation and confidence
filtering; downstream policy utility; optional physical-arm transfer.
Large-scale data generation, dynamic grasping, walking, RL, and large-model
pretraining are not prerequisites for the first run.

## Preserved engineering requirements

Keep raw data and provenance even when adding LeRobot-compatible export.
Benchmark before giving training-duration estimates. Keep simulator-only state
out of translator inputs. Track reconstruction quality separately from physical
control success. Preserve original PDFs as historical reports, not active gates.
