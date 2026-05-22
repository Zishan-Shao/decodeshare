#!/usr/bin/env python3
"""Build and optionally upload the DecodeShare Gradio demo to Hugging Face Spaces."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "demo" / "hf_space"
DEFAULT_BUNDLE_DIR = REPO_ROOT / "outputs" / "hf_space_bundle"
DEFAULT_SPACE_ID = "Zishan-Shao/decodeshare-demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default=DEFAULT_SPACE_ID, help="HF Space repo id, e.g. user/name")
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    parser.add_argument("--bundle-only", action="store_true", help="Only build the local Space bundle")
    parser.add_argument(
        "--hardware",
        default="",
        help="Optional hardware request after upload, e.g. t4-small or a10g-small",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_bundle(bundle_dir: Path) -> Path:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    for name in ["README.md", "requirements.txt", "app.py"]:
        copy_file(TEMPLATE_DIR / name, bundle_dir / name)

    copy_file(REPO_ROOT / "demo" / "app.py", bundle_dir / "demo" / "app.py")
    copy_file(
        REPO_ROOT / "demo" / "run_steering_projection_demo.py",
        bundle_dir / "demo" / "run_steering_projection_demo.py",
    )
    copy_file(
        REPO_ROOT / "demo" / "assets" / "interactive_tinyllama_chat_cache.pt",
        bundle_dir / "demo" / "assets" / "interactive_tinyllama_chat_cache.pt",
    )
    (bundle_dir / "demo" / "__init__.py").write_text("", encoding="utf-8")
    return bundle_dir


def upload_bundle(bundle_dir: Path, space_id: str, private: bool, hardware: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    api = HfApi()
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(bundle_dir),
        commit_message="Update DecodeShare Gradio demo",
    )
    print(f"Uploaded Space bundle to https://huggingface.co/spaces/{space_id}")
    print(f"Commit: {commit.oid}")

    if hardware:
        runtime = api.request_space_hardware(repo_id=space_id, hardware=hardware)
        print(f"Requested hardware: {hardware}")
        print(f"Runtime stage: {runtime.stage}")


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser()
    if not bundle_dir.is_absolute():
        bundle_dir = REPO_ROOT / bundle_dir
    build_bundle(bundle_dir)
    print(f"Built Space bundle at {bundle_dir}")

    if args.bundle_only:
        return
    upload_bundle(bundle_dir, args.space_id, args.private, args.hardware)


if __name__ == "__main__":
    sys.exit(main())
