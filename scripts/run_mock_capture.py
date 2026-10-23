#!/usr/bin/env python3
"""Contract sections 3, 4, 6: capture a 300-step CPU fixture and validate it.

Run from a checkout without installation:
    ./.venv-harness/bin/python scripts/run_mock_capture.py /tmp/mock_ep
For the module CLI without installation, set PYTHONPATH=src.
Output directories must be empty; omitting the directory creates a temporary one.
"""

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avatar_harness.mock_env import MockEnv, scripted_actions
from avatar_harness.recorder import EpisodeRecorder
from avatar_harness.validate import validate_episode


def main(argv=None) -> int:
    """Section 6: record nonconstant commands, reload the episode, exit 0/1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, nargs="?")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="avatar-mock-episode-"))
    try:
        EpisodeRecorder(MockEnv(), out_dir, {"seed": args.seed, "episode_id": f"mock-{args.seed}"}).record(
            scripted_actions(args.steps)
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FAIL capture: {error}")
        return 1
    print(f"Episode: {out_dir.resolve()}")
    print("Scope: CPU mock capture integrity; physical simulation and ROCm training are not exercised.")
    report = validate_episode(out_dir)
    print(report.format())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
