#!/usr/bin/env python3
"""v0.1 training smoke + warm benchmark on a recorded episode (contract §6).

Loads a captured episode through the real on-disk loader (transitions.parquet
+ PNG streams), trains a tiny CNN+MLP policy to predict recorded actions from
(external image, wrist image, measured proprioception), and:
  stage 1: >=2 optimizer steps, assert finite loss each step
  stage 2: warmed-up 500 (default) step run, measure steps/sec and peak VRAM
Writes benchmark.json next to the episode. GPU device via --device (default
cuda:0). Pure execution gate: no task-success claims.

Run inside a venv with torch(+rocm) and numpy/pyarrow/pillow.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def load_episode(ep_dir):
    t = pq.read_table(os.path.join(ep_dir, "transitions.parquet")).to_pydict()
    n = len(t["step_index"])
    ext = np.stack([np.asarray(Image.open(os.path.join(ep_dir, t["external_path"][i])))
                    for i in range(n)])
    wri = np.stack([np.asarray(Image.open(os.path.join(ep_dir, t["wrist_path"][i])))
                    for i in range(n)])
    proprio = np.stack([np.concatenate([t["measured_joint_position_rad"][i],
                                        [t["measured_gripper_opening"][i]]])
                        for i in range(n)]).astype(np.float32)
    action = np.stack([np.concatenate([t["command_joint_position_target_rad"][i],
                                       [t["command_gripper_opening"][i]]])
                       for i in range(n)]).astype(np.float32)
    return ext, wri, proprio, action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warm-steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    dev = torch.device(args.device)
    is_cuda = dev.type == "cuda"
    if is_cuda:
        torch.cuda.init()
        gpu_name = torch.cuda.get_device_name(dev)
        free_b, total_b = torch.cuda.mem_get_info(dev)
    else:
        gpu_name, free_b, total_b = "cpu", 0, 0

    ext, wri, proprio, action = load_episode(args.episode_dir)
    n = len(ext)
    if n < args.batch:
        sys.exit(f"episode too short: {n} transitions < batch {args.batch}")

    class Policy(nn.Module):
        def __init__(self, feat_dim):
            super().__init__()
            enc = lambda: nn.Sequential(  # noqa: E731
                nn.Conv2d(3, 8, 8, stride=4), nn.ReLU(),
                nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(), nn.Flatten())
            self.ext, self.wri = enc(), enc()
            self.head = nn.Sequential(nn.Linear(2 * feat_dim + 7, 64),
                                      nn.ReLU(), nn.Linear(64, 6))
        def forward(self, e, w, p):
            return self.head(torch.cat([self.ext(e), self.wri(w), p], dim=1))

    # measure encoder output width on a real-shaped zero batch (320x240)
    def make_encoder():
        return nn.Sequential(nn.Conv2d(3, 8, 8, stride=4), nn.ReLU(),
                             nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(),
                             nn.Flatten())
    with torch.no_grad():
        probe = torch.zeros(1, 3, 240, 320, device=dev)
        feat_dim = make_encoder().to(dev)(probe).shape[1]
        del probe
    model = Policy(feat_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    lossf = nn.MSELoss()
    rng = np.random.default_rng(0)

    def batch():
        idx = rng.integers(0, n, size=args.batch)
        e = torch.from_numpy(ext[idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
        w = torch.from_numpy(wri[idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
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
        e, w, p, a = batch()
        loss = lossf(model(e, w, p), a)
        opt.zero_grad(); loss.backward(); opt.step()
        if i >= 50 and steps == 0:
            if is_cuda: torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
        if i >= 50:
            steps += 1
    if is_cuda: torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(dev) if is_cuda else 0

    bench = {
        "stage": "v0.1_training_smoke_and_warm",
        "smoke_losses": losses,
        "warm_steps_measured": steps,
        "elapsed_s": round(dt, 3),
        "steps_per_sec": round(steps / dt, 2),
        "batch_size": args.batch,
        "peak_vram_bytes": int(peak),
        "device": {"name": gpu_name, "requested": str(dev),
                   "free_at_start_bytes": int(free_b), "total_bytes": int(total_b)},
        "software": {"torch": torch.__version__, "hip": str(torch.version.hip),
                     "python": sys.version.split()[0],
                     "numpy": np.__version__},
        "episode": {"dir": os.path.abspath(args.episode_dir), "transitions": n,
                    "frames_loaded": 2 * n,
                    "frames_per_sec_avg_capture_unmeasured_here": None},
    }
    out = args.out or os.path.join(args.episode_dir, "benchmark.json")
    with open(out, "w") as f:
        json.dump(bench, f, indent=2)
    print("TRAIN_SMOKE_OK losses_finite=True",
          f"steps/sec={bench['steps_per_sec']}",
          f"peak_vram={bench['peak_vram_bytes']/2**20:.0f} MiB",
          f"wrote {out}")


if __name__ == "__main__":
    main()
