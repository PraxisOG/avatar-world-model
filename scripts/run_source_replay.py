"""Prepare the selected real-source right-hand replay and show it locally."""

import argparse
import json
import os
from pathlib import Path
import signal


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Inspect source, fit joints, validate and save evidence, then exit')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output', type=Path, default=root / 'artifacts/source-replay')
    args = parser.parse_args()
    config = json.loads((root / 'config/arm_probe.json').read_text())
    selection = json.loads((root / 'config/source_replay.json').read_text())
    os.environ.setdefault('MUJOCO_GL', config['render_backend'])
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', config['egl_device_id'])
    from source_replay import prepare, load_replay
    if args.prepare:
        from fetch_assets import main as fetch_assets
        from fetch_egodex_sample import fetch
        fetch_assets()
        missing = [str((root / selection[key]).relative_to(root / 'data/egodex'))
                   for key in ('source_hdf5', 'source_mp4') if not (root / selection[key]).exists()]
        if missing:
            fetch('part2', missing, root / 'data/egodex')
        prepare(selection, config, args.output)
        return
    model, config, frames, replay = load_replay(args.output)
    from probe_dashboard import Dashboard
    dashboard = Dashboard(args.port)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: dashboard.stop.set())
    print(f'Recorded-source replay: {dashboard.url}', flush=True)
    print('Right arm/hand only. Source POV is a reference; no simulated head camera yet. Ctrl-C stops the server.', flush=True)
    try:
        dashboard.run(model, config, frames, ['Recorded reach before contact'] * len(frames), replay=replay)
    finally:
        dashboard.close()


if __name__ == '__main__':
    main()
