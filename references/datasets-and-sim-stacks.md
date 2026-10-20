# References: datasets and simulation stacks

Compiled 2026-10-20 from web research. Verify licenses and specs before
building on any of these; descriptions are as-announced by authors.

## Part 1 — Open datasets: first-person video + action labels

The pattern to look for: egocentric (internal-view) video with *native*
pose/action capture at recording time, not post-hoc annotation.

| Dataset | Year | Scale | Action labels | License / access |
|---|---|---|---|---|
| **EgoDex** (Apple) | 2025 | 829 h, 90M frames, 338k episodes, 194 tabletop tasks, 1080p30 | Full 3D skeleton: head, upper body, 25 joints/hand @30Hz + camera intrinsics/extrinsics + language | CC-by-NC-ND; direct zip download. github.com/apple/ml-egodex. Test split 16 GB; `basic_pick_place` in part2 recommended starter |
| **Open-AoE** (Ant Research) | 2026 | ~2,000 h smartphone egocentric manipulation, 500+ collectors | MANO hand poses, camera trajectory, atomic actions, bilingual text; capture-to-train tooling | Open. arXiv:2607.14183; HF `inclusionAI/OpenAoE-2000h` |
| **EgoMimic** | 2025 | ~4 h human + robot | 3D wrist poses; human+robot co-training (the canonical recipe proof) | Kareer et al. 2025 |
| **EgoSteer / EgoSmith** | 2026 | 1.54M human episodes + 192 h robot | Bimanual human-to-robot | arXiv:2607.09701 |
| **World In Your Hands** (TARS) | 2025 | 1,000+ h labeled human manipulation | video-language annotations, dexterous transfer | arXiv:2512.24310 |
| **Humanola egocentric** | 2026 | ~3.2 h, 201Hz IMU | world-frame hand keypoints, finger angles, grip; **LeRobot v2.1 format** | HF `humanola-inc/egocentric_labeled` |
| **EPIC-KITCHENS-100** | 2018-21 | 100 h kitchen video, 90k action segments | segment-level only (no native pose) — but it IS the cooking benchmark corpus | Bristol, research license |
| **Egocentric-1M / -10K** (Build AI) | 2026 | 1M h claimed; 10K h factory | hand-visibility rich; verify independently | claimed free commercial |
| **EgoVid-5M** | 2025 | 5M clips | built for egocentric video *generation* (world-model pretraining) | NeurIPS 2025 |
| **Ego4D** (Meta) | 2021+ | 3,670 h | mostly no native hand pose; breadth without actions | research license |

Curated index: github.com/player0718/awesome-ego-video-datasets

Key insight for this project: these give internal view + actions but no
external view. Our sim can **replay their camera trajectories and hand
skeletons and render the missing external view** — instant paired data
without collecting anything. (Note: NC/ND licenses like EgoDex's permit
research use but constrain redistribution of derivatives.)

## Part 2 — Simulation world / engine options

Requirements: synchronized multi-camera RGB at same timestep as action
stream; joint-level ground truth; loads URDF/MJCF; LeRobot export;
runs on AMD ROCm (4x V620) without heroics.

### Primary candidate: Genesis (genesis-world) — Apache 2.0
- Unified multi-physics engine (rigid, FEM, MPM, SPH, PBD, cloth, fluids),
  photo-realistic Nyx ray-tracing renderer, 3DGS support.
- **Cross-platform compiler (Quadrants) targets CUDA, AMD ROCm, Metal,
  Vulkan, CPU.** Official AMD docker: `docker/Dockerfile.amdgpu`.
- Multi-cam multi-env rendering tutorials; attached/follow cameras as
  first-class camera sensors = our internal+external pair.
- Published AMD pipeline (arXiv:2607.22997): Genesis → LeRobot dataset →
  SmolVLA training → real Franka arm, **entire loop on ROCm in under an
  hour.** This is nearly our exact pipeline, already demonstrated on AMD.
- Python-native. URDF/MJCF/OBJ/GLB/USD loading. Robot-cloth teleop demo
  exists.

### Alternates
- **MuJoCo**: best contact fidelity, cheap (CPU works), MJCF ecosystem.
  Rendering is the weak half. SIMPLE (psi-lab.ai) proves a hybrid:
  MuJoCo physics + photoreal renderer replayed offline, VR teleop,
  zero-shot transfer to real humanoids. Good fallback / physics backend.
- **ManiSkill3 / SAPIEN**: strongest manipulation + articulated-object
  asset ecosystem (PartNet-Mobility: cabinets, drawers, appliances),
  GPU-parallel visual data collection. CUDA-first — friction on V620s.
- **Isaac Lab**: richest humanoid assets + Mimic augmentation + XR hand
  teleop, but CUDA/Omniverse-only. Treat as blocker on this box;
  borrow ideas (teleop → HDF5 demos → Mimic augmentation).
- **Game engines proper (Unity/ML-Agents, Godot)**: beautiful visuals,
  cheap assets, and SimXR proved Unity renders synthetic egocentric
  views from mocap with domain randomization. BUT: game physics is
  not contact-accurate, joint truth needs custom plumbing, and the
  Python↔engine bridge is glue-code hell. Verdict: use for *looks*
  (later, optional render pass), not for the data-generating physics.
- **Blender + pybullet/mujoco**: free pipeline for offline high-quality
  re-rendering of logged trajectories. Cheap domain randomization
  (lighting/textures/skyboxes) exactly like SimXR did.

### Recommended stack (v0)
Genesis on the AMD docker path → dual camera sensors (attached eye-cam +
fixed external rig cam) → LeRobot-format episodes with joint streams →
small policy trained on Moe. MuJoCo kept as physics cross-check. Blender
as later-day beauty renderer. Revisit SAPIEN if articulated-kitchen
objects dominate the task set and a CUDA rental is acceptable for
those runs only.
