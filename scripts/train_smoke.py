#!/usr/bin/env python3
"""v0.1 training smoke + warm benchmark on a recorded episode (contract §6).

Loads a captured episode through the real on-disk loader (transitions.parquet
+ PNG streams), trains a tiny CNN+MLP policy to predict recorded actions from
(external image, head image, measured proprioception), and:
  stage 1: >=2 optimizer steps, assert finite loss each step
  stage 2: warmed-up 500 (default) step run, measure steps/sec and peak VRAM
Writes training_benchmark.json next to the episode. GPU device via --device (default
cuda:0). Pure execution gate: no task-success claims.

Run inside a venv with torch(+rocm) and numpy/pyarrow/pillow.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avatar_harness.contract import (ARMS, COMMAND_JOINT_ORDER, CONTRACT_REVISION,
                                     JOINT_ORDER, TRANSITION_SCHEMA)

# Explicit policy input allowlist. Actions are targets; privileged tables and
# even unused transition columns never enter this loader.
INPUT_COLUMNS = ("external_path", "head_path", "measured_joint_position_rad", "measured_gripper_opening")
TARGET_COLUMNS = ("command_joint_position_target_rad", "command_gripper_opening")


def load_episode(ep_dir):
    with open(os.path.join(ep_dir, "manifest.json")) as stream:
        manifest = json.load(stream)
    if (manifest["contract_revision"] != CONTRACT_REVISION or manifest["joint_order"] != list(JOINT_ORDER)
            or manifest["command_joint_order"] != list(COMMAND_JOINT_ORDER) or manifest["gripper_order"] != list(ARMS)):
        raise ValueError("training requires the Rev B left-then-right schema")
    table = pq.read_table(os.path.join(ep_dir, "transitions.parquet"), columns=[*INPUT_COLUMNS, *TARGET_COLUMNS])
    for name in table.column_names:
        if table.schema.field(name).type != TRANSITION_SCHEMA.field(name).type:
            raise ValueError(f"wrong training column type: {name}")
    t = table.to_pydict()
    n = table.num_rows
    if not n:
        raise ValueError("cannot train on an empty episode")

    def images(column):
        arrays = []
        for relative in t[column]:
            path = Path(ep_dir) / relative
            if not path.resolve().is_relative_to(Path(ep_dir).resolve()):
                raise ValueError("RGB path escapes episode directory")
            with Image.open(path) as image:
                pixels = np.asarray(image)
                if image.mode != "RGB" or pixels.shape != (240, 320, 3):
                    raise ValueError("training requires RGB [240,320,3]")
                arrays.append(pixels)
        return np.stack(arrays)

    ext, head = images("external_path"), images("head_path")
    proprio = np.concatenate([t["measured_joint_position_rad"], t["measured_gripper_opening"]], axis=1).astype(np.float32)
    action = np.concatenate([t["command_joint_position_target_rad"], t["command_gripper_opening"]], axis=1).astype(np.float32)
    if proprio.shape != (n, 14) or action.shape != (n, 12) or not all(np.isfinite(v).all() for v in (proprio, action)):
        raise ValueError("expected finite proprio [N,14] and actions [N,12]")
    return ext, head, proprio, action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warm-steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.warm_steps < 1 or args.batch < 1:
        ap.error("--warm-steps and --batch must be positive")
    if args.device.split(":", 1)[0] == "cpu":
        # Deny device access before importing a ROCm-enabled torch build. No
        # CUDA/HIP availability probes or runtime initialization on this path.
        from avatar_harness.mujoco_env import _deny_devices
        _deny_devices()
        os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    import torch
    import torch.nn as nn

    dev = torch.device(args.device)
    if dev.type not in ("cpu", "cuda"):
        ap.error("--device must be cpu or cuda[:index]")
    is_cuda = dev.type == "cuda"
    if is_cuda:
        torch.cuda.init()
        gpu_name = torch.cuda.get_device_name(dev)
        free_b, total_b = torch.cuda.mem_get_info(dev)
    else:
        gpu_name, free_b, total_b = "cpu", 0, 0
        torch.set_num_threads(2)

    ext, head_images, proprio, action = load_episode(args.episode_dir)
    n = len(ext)
    if n < args.batch:
        sys.exit(f"episode too short: {n} transitions < batch {args.batch}")

    class Policy(nn.Module):
        def __init__(self, feat_dim):
            super().__init__()
            enc = lambda: nn.Sequential(  # noqa: E731
                nn.Conv2d(3, 8, 8, stride=4), nn.ReLU(),
                nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(), nn.Flatten())
            self.external_encoder, self.head_encoder = enc(), enc()
            self.action_head = nn.Sequential(nn.Linear(2 * feat_dim + 14, 64),
                                      nn.ReLU(), nn.Linear(64, 12))
        def forward(self, external, head, proprio):
            return self.action_head(torch.cat([self.external_encoder(external), self.head_encoder(head), proprio], dim=1))

    # Measure encoder output width using a reloaded RGB frame.
    def make_encoder():
        return nn.Sequential(nn.Conv2d(3, 8, 8, stride=4), nn.ReLU(),
                             nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(),
                             nn.Flatten())
    with torch.no_grad():
        probe = torch.from_numpy(ext[:1].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
        feat_dim = make_encoder().to(dev)(probe).shape[1]
        del probe
    model = Policy(feat_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    lossf = nn.MSELoss()
    rng = np.random.default_rng(0)

    def batch():
        idx = rng.integers(0, n, size=args.batch)
        e = torch.from_numpy(ext[idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
        w = torch.from_numpy(head_images[idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
        p = torch.from_numpy(proprio[idx]).to(dev)
        a = torch.from_numpy(action[idx]).to(dev)
        return e, w, p, a

    # stage 1: basic gate — finite loss over optimizer steps
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    losses = []
    for _ in range(4):
        e, w, p, a = batch()
        loss = lossf(model(e, w, p), a)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    if not all(math.isfinite(l) for l in losses):
        sys.exit(f"FAIL: non-finite loss in smoke steps: {losses}")

    # stage 2: warm benchmark (discard 50 warmup steps)
    steps = 0
    t0 = time.perf_counter()
    for i in range(args.warm_steps + 50):
        if i == 50:
            if is_cuda: torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
        e, w, p, a = batch()
        loss = lossf(model(e, w, p), a)
        opt.zero_grad(); loss.backward(); opt.step()
        if not math.isfinite(float(loss.item())):
            sys.exit(f"FAIL: non-finite warm loss at step {i}")
        if i >= 50:
            steps += 1
    if is_cuda: torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(dev) if is_cuda else 0

    if not is_cuda and torch.cuda.is_initialized():
        raise RuntimeError("CPU training unexpectedly initialized CUDA/HIP")

    bench = {
        "stage": "v0.1_training_smoke_and_warm",
        "smoke_losses": losses, "smoke_steps": len(losses),
        "proprio_dim": 14, "action_dim": 12, "image_streams": ["external", "head"],
        "warmup_steps": 50,
        "warm_steps_measured": steps,
        "elapsed_s": round(dt, 3),
        "steps_per_sec": round(steps / dt, 2),
        "batch_size": args.batch,
        "peak_vram_bytes": int(peak),
        "device": {"name": gpu_name, "requested": str(dev),
                   "gpu_runtime_initialized": torch.cuda.is_initialized(),
                   "free_at_start_bytes": int(free_b), "total_bytes": int(total_b)},
        "software": {"torch": torch.__version__, "hip": str(torch.version.hip),
                     "python": sys.version.split()[0],
                     "numpy": np.__version__},
        "episode": {"dir": os.path.abspath(args.episode_dir), "transitions": n,
                    "frames_loaded": 2 * n,
                    "frames_per_sec_avg_capture_unmeasured_here": None},
    }
    out = args.out or os.path.join(args.episode_dir, "training_benchmark.json")
    with open(out, "w") as f:
        json.dump(bench, f, indent=2)
    print("TRAIN_SMOKE_OK losses_finite=True",
          f"smoke_steps={len(losses)} warm_steps={steps} proprio_dim=14 action_dim=12",
          f"steps/sec={bench['steps_per_sec']}",
          f"peak_vram={bench['peak_vram_bytes']/2**20:.0f} MiB",
          f"wrote {out}")


if __name__ == "__main__":
    main()
