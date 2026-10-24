"""Fetch pinned, unmodified model assets into the ignored local cache."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    assets = json.loads((ROOT / "assets.lock.json").read_text())
    for name, asset in assets.items():
        path = ROOT / ".cache" / "upstream" / name
        if path.exists():
            if not (path / ".git").is_dir():
                raise RuntimeError(f"Refusing to replace non-checkout directory {path}")
            if git(path, "remote", "get-url", "origin") != asset["url"]:
                raise RuntimeError(f"Unexpected origin in {path}")
            if git(path, "status", "--porcelain"):
                raise RuntimeError(f"Preserving local changes in {path}; use a clean asset cache")
            if git(path, "rev-parse", "HEAD") != asset["revision"]:
                raise RuntimeError(f"Unexpected asset revision in {path}; use a fresh asset cache")
        else:
            path.mkdir(parents=True)
            git(path, "init")
            git(path, "remote", "add", "origin", asset["url"])
            git(path, "config", "remote.origin.promisor", "true")
            git(path, "config", "remote.origin.partialclonefilter", "blob:none")
            git(path, "fetch", "--depth", "1", "--filter=blob:none", "origin", asset["revision"])
            git(path, "sparse-checkout", "set", "--no-cone", *asset["sparse_paths"])
            git(path, "checkout", "--detach", asset["revision"])
        if not (path / asset["license_path"]).is_file():
            raise RuntimeError(f"License missing in {path}")
        print(f"{name}: {asset['revision']} ({asset['license']})")


if __name__ == "__main__":
    main()
