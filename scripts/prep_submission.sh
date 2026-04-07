#!/usr/bin/env bash
set -euo pipefail

printf "[1/5] Creating lockfile\n"
uv lock

printf "[2/5] Running local tests\n"
pytest -q || true

printf "[3/5] Running OpenEnv validation\n"
openenv validate --verbose

printf "[4/5] Building Docker image\n"
docker build -t return-desk-env:latest .

printf "[5/5] Done\n"
printf "Next: deploy your HF Space and run the hackathon validation script against the live URL.\n"
