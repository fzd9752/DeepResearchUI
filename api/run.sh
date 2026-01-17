#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables from $ENV_FILE..."
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "Warning: .env not found at $ENV_FILE"
fi

export PYTHONPATH="$ROOT_DIR/inference:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

exec python -m uvicorn api.server:app --reload --port "${API_PORT:-8000}"
