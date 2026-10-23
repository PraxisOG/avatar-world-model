# Roadmap to the first training run

Agreed 2026-10-22. All gates below are pending, not completed results.
Goal: external video of an unseen movement -> joint trajectory + consistent POV,
initially in one fixed simulated scene.

## Phase 1 — validate one source recording and produce one faithful replay

1. Inspect one small sample from an eligible egocentric motion dataset.
   Document head and both hand position/orientation fields, timestamp alignment,
   coordinate frames, scale, calibration, tracking validity, source split,
   provenance, and permitted use. Choose a short contact-free segment.
2. Probe the simulator on the actual machine: one fixed-base human-proportioned
   upper body, articulated arms and head, simple hands, plain scene, fixed
   external camera, and head-mounted POV camera. Record versions and throughput.
3. Map tracked motion to the body with continuous joint targets. Preserve source
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

Report joint-angle error, head/hand position and orientation error, temporal
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
Large-scale data generation, articulated fingers, walking, RL, and large-model
pretraining are not prerequisites for the first run.

## Preserved engineering requirements

Keep raw data and provenance even when adding LeRobot-compatible export.
Benchmark before giving training-duration estimates. Keep simulator-only state
out of translator inputs. Track reconstruction quality separately from physical
control success. Preserve original PDFs as historical reports, not active gates.
