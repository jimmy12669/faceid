#!/usr/bin/env python3
"""Download the face detection and recognition models.

They are pulled from OpenCV's model zoo over HTTPS and checked against pinned
SHA-256 digests before anything is written into place. A model file is code as
far as your process is concerned -- if the download is tampered with, or the
upstream file is silently replaced, the digest check is what catches it.

    python fetch_models.py            # into $FACEID_HOME/models
    python fetch_models.py --dir ./m  # somewhere else
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import config

# Git LFS pointers live at raw.githubusercontent.com; the media host serves the
# actual bytes.
BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

MODELS = [
    {
        "name": "face_detection_yunet_2023mar.onnx",
        "url": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "size": 232589,
        "what": "YuNet face detector",
    },
    {
        "name": "face_recognition_sface_2021dec.onnx",
        "url": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size": 38696353,
        "what": "SFace recogniser",
    },
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(model: dict, target_dir: Path, *, force: bool = False) -> bool:
    target = target_dir / model["name"]
    if target.exists() and not force:
        found = digest(target)
        if found == model["sha256"]:
            print(f"  {model['name']}: already present and verified")
            return True
        print(f"  {model['name']}: on disk but digest does not match -- refetching")

    print(f"  {model['name']}: downloading {model['size'] / 1e6:.1f} MB ...")
    tmp_fd, tmp_name = tempfile.mkstemp(dir=target_dir, suffix=".part")
    tmp = Path(tmp_name)
    os.close(tmp_fd)
    try:
        with urllib.request.urlopen(model["url"], timeout=120) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        found = digest(tmp)
        if found != model["sha256"]:
            print(f"    REJECTED: sha256 {found}\n    expected: {model['sha256']}")
            return False
        tmp.chmod(0o644)
        tmp.replace(target)
        print(f"    ok, sha256 verified")
        return True
    except Exception as exc:
        print(f"    failed: {exc}")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    settings = config.load()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=settings.models_dir)
    parser.add_argument("--force", action="store_true", help="redownload even if verified")
    args = parser.parse_args(argv)

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {args.dir}")
    ok = all(download(model, args.dir, force=args.force) for model in MODELS)
    if ok:
        print("\nAll models verified. You can run `python cli.py init` now.")
        return 0
    print("\nOne or more models could not be verified. Not usable -- try again.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
