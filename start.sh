#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="turjaun985123/photo-vector-api"
ASSET="photo-vector-api-linux-x86_64.tar.gz"
RUNTIME_DIR="${TMPDIR:-/tmp}/photo-vector-api-runtime"
ARCHIVE="$RUNTIME_DIR/$ASSET"
ENGINE="$RUNTIME_DIR/rust-vision-engine"
MODEL="$RUNTIME_DIR/models/dinov2_vits14.onnx"

mkdir -p "$RUNTIME_DIR"

if [[ ! -x "$ENGINE" || ! -f "$MODEL" ]]; then
  echo "Downloading latest Photo Vector API release..."
  curl --fail --location --retry 5 --retry-all-errors \
    "https://github.com/$REPOSITORY/releases/latest/download/$ASSET" \
    --output "$ARCHIVE"
  tar --extract --gzip --file "$ARCHIVE" --directory "$RUNTIME_DIR"
fi

export VISION_TIER1_MODEL="$MODEL"
PORT="${PORT:-10000}"
INTRA_OP_THREADS="${VISION_INTRA_OP_THREADS:-2}"

exec "$ENGINE" \
  --tier1-model "$VISION_TIER1_MODEL" \
  --embedding-dim 384 \
  --max-batch-size 1 \
  --tensor-pool-capacity 2 \
  --intra-op-threads "$INTRA_OP_THREADS" \
  serve \
  --mode embedding-search \
  --bind "0.0.0.0:$PORT"
