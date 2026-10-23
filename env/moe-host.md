# Moe environment record — generated 2026-10-21T18:23Z

## Host
Linux moe 6.8.0-139-generic #139-Ubuntu SMP PREEMPT_DYNAMIC Sat Aug  1 03:52:05 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

## GPU/driver
/sys/module/amdgpu/version: 6.16.13
6.16.13

## ROCm
/opt/rocm
/opt/rocm-7.2.0
rocminfo not on host (containerized backend only)

## Docker images (backend)
moe-qwen-path-b:a6ab43cf9 e6bf856be6e7

## Container runtime versions (backend)
torch 2.12.0+git6bbd260
vllm 0.30.0rc1+moe.ga6ab43cf9

## Project venvs (this repo)
- .venv-harness: python3.12 venv; numpy 2.5.4, pyarrow 26.0.0, pillow 13.0.0 — offline harness/tests
- .venv-sim: python3.12 venv; mujoco 3.16.0 — MuJoCo engine branch (CPU physics + EGL render probe passed)
- .venv-genesis: python3.12 venv; genesis 1.4.4 + torch 2.5.1+rocm6.2 (hip 6.2.41133) — Genesis branch; needs HSA_OVERRIDE_GFX_VERSION=10.3.0 for torch on gfx1030

## Probe log 2026-10-21
- torch rocm6.2 wheel sees 4x V620; fwd/bwd/optimizer 30 steps finite on cuda:3; hipBLASLt-unsupported warning auto-falls back to hipblas (benign)
- genesis gs.cpu backend: scene build + 20 physics steps OK; add_camera 320x240 rasterizer render OK (240,320,3 uint8) headless
- mujoco EGL on Moe: mjrender Renderer + mj_forward render OK with MUJOCO_GL=egl
