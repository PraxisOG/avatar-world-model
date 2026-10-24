"""Build and launch the one-arm probe, or validate it without the dashboard."""

import argparse
import json
import os
from pathlib import Path
import signal
import traceback


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'config/arm_probe.json')
    parser.add_argument('--output', type=Path, default=root / 'artifacts/arm-probe')
    parser.add_argument('--validate', action='store_true', help='Save measured evidence and exit; no HTTP server')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 0 <= args.port <= 65535:
        parser.error('port must be between 0 and 65535')
    os.environ.setdefault('MUJOCO_GL', config['render_backend'])
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', config['egl_device_id'])
    from fetch_assets import main as fetch_assets
    from arm_assembly import build_scene, trajectory
    from probe_validation import validate
    fetch_assets()
    model, manifest = build_scene(config, args.output)
    frames, stages = trajectory(model, config)
    if args.validate:
        validate(model, config, frames, stages, args.output)
        return
    from probe_dashboard import Dashboard
    dashboard = Dashboard(args.port)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: dashboard.stop.set())
    print(f'Live assembly probe: {dashboard.url}', flush=True)
    print('Scripted kinematic motion only. No head POV or human source recording yet. Ctrl-C stops the server.', flush=True)
    try:
        dashboard.run(model, config, frames, stages)
    except Exception as error:
        traceback.print_exc()
        with dashboard.lock:
            dashboard.snapshot.update(status='error', error=str(error))
        # Leave the actionable error visible until the user stops the process.
        dashboard.stop.wait()
    finally:
        dashboard.close()


if __name__ == '__main__':
    main()
