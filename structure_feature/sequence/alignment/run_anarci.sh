#!/usr/bin/env bash
set -euo pipefail

TAG="2024.05.21--pyhdfd78af_0"
IMG="quay.io/biocontainers/anarci:$TAG"

docker pull --platform linux/amd64 "$IMG" >/dev/null
docker run --rm --platform linux/amd64 -v "$PWD":/data "$IMG" \
  bash -lc "ANARCI $*"