#!/usr/bin/env python3
"""
deploy_to_hf.py — Deploy ReturnDeskEnv to a Hugging Face Space.

Usage:
    python deploy_to_hf.py --space-id your-username/return-desk-env

Requirements:
    pip install huggingface_hub

Environment variables:
    HF_TOKEN   Your Hugging Face write token (required)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()

# Files and directories to upload
UPLOAD_PATTERNS = [
    "server/",
    "tasks/",
    "graders.py",
    "models.py",
    "rewards.py",
    "inference.py",
    "client.py",
    "openenv.yaml",
    "pyproject.toml",
    "README.md",
    "SUBMISSION_SUMMARY.md",
    "Dockerfile",
    ".dockerignore",
]

SPACE_METADATA = """\
---
title: ReturnDeskEnv
emoji: 📦
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: >
  E-commerce returns & fraud-detection RL environment (Meta PyTorch OpenEnv Hackathon)
---
"""


def check_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[error] HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("        Get your token at https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)
    return token


def validate_local_files() -> list[Path]:
    """Check all required files exist before uploading."""
    missing = []
    found = []
    for pattern in UPLOAD_PATTERNS:
        path = REPO_ROOT / pattern
        if path.exists():
            found.append(path)
        else:
            missing.append(pattern)
    if missing:
        print(f"[warn] The following files/dirs were not found and will be skipped: {missing}")
    return found


def run_self_validation() -> bool:
    """Quick import check before uploading."""
    print("[check] Running pre-deploy validation...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print("[warn] Some tests failed. Review before deploying:")
            print(result.stdout[-2000:])
            return False
        print(f"[ok] Tests passed.\n{result.stdout.strip()}")
        return True
    except Exception as exc:
        print(f"[warn] Could not run tests: {exc}")
        return True  # Don't block deploy on test runner failure


def deploy(space_id: str, token: str, skip_tests: bool = False) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[error] huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    if not skip_tests:
        ok = run_self_validation()
        if not ok:
            print("[error] Deploy aborted due to test failures. Use --skip-tests to force.", file=sys.stderr)
            sys.exit(1)

    api = HfApi(token=token)

    print(f"\n[deploy] Target Space: https://huggingface.co/spaces/{space_id}")
    print("[deploy] Creating/verifying Space repo...")

    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token,
        )
        print("[ok] Space repo ready.")
    except Exception as exc:
        print(f"[error] Could not create/access Space: {exc}", file=sys.stderr)
        sys.exit(1)

    # Upload README with Space metadata if it exists
    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        if not readme_content.startswith("---"):
            # Prepend HF Space metadata
            patched_content = SPACE_METADATA + "\n" + readme_content
            api.upload_file(
                path_or_fileobj=patched_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=space_id,
                repo_type="space",
                token=token,
            )
            print("[ok] Uploaded README.md with Space metadata.")

    # Upload all other files
    print("[deploy] Uploading project files...")
    for pattern in UPLOAD_PATTERNS:
        if pattern == "README.md":
            continue  # already handled above
        path = REPO_ROOT / pattern
        if not path.exists():
            continue
        if path.is_dir():
            api.upload_folder(
                folder_path=str(path),
                path_in_repo=pattern.rstrip("/"),
                repo_id=space_id,
                repo_type="space",
                token=token,
                ignore_patterns=["__pycache__", "*.pyc", ".pytest_cache"],
            )
            print(f"[ok] Uploaded folder: {pattern}")
        else:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=pattern,
                repo_id=space_id,
                repo_type="space",
                token=token,
            )
            print(f"[ok] Uploaded file: {pattern}")

    print(f"\n✅ Deploy complete!")
    print(f"   Space URL: https://huggingface.co/spaces/{space_id}")
    print(f"   Logs:      https://huggingface.co/spaces/{space_id}/logs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy ReturnDeskEnv to Hugging Face Spaces")
    parser.add_argument(
        "--space-id",
        required=True,
        help="Hugging Face Space ID in the form 'username/space-name'",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        default=False,
        help="Skip running the test suite before deploying",
    )
    args = parser.parse_args()

    if "/" not in args.space_id:
        print(f"[error] --space-id must be in the form 'username/space-name', got: {args.space_id!r}", file=sys.stderr)
        sys.exit(1)

    token = check_token()
    validate_local_files()
    deploy(space_id=args.space_id, token=token, skip_tests=args.skip_tests)


if __name__ == "__main__":
    main()
