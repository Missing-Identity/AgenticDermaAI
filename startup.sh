#!/usr/bin/env bash
# startup.sh — Pull required Ollama models on first deploy.
#
# Run this AFTER "docker compose up -d" once the ollama container is healthy.
# Models are stored in the "ollama_data" named volume, so subsequent container
# restarts do NOT need to re-pull them.
#
# Usage:
#   chmod +x startup.sh
#   ./startup.sh

set -euo pipefail

OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-$(docker compose ps -q ollama)}"

if [[ -z "$OLLAMA_CONTAINER" ]]; then
  echo "[startup] ERROR: Could not find the running ollama container."
  echo "          Make sure you have run: docker compose up -d"
  exit 1
fi

echo "[startup] Pulling medgemma (vision model) — ~2.5 GB …"
docker exec "$OLLAMA_CONTAINER" ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M

echo "[startup] Pulling qwen2.5:7b-instruct (text / orchestrator model) — ~4.7 GB …"
docker exec "$OLLAMA_CONTAINER" ollama pull qwen2.5:7b-instruct

echo ""
echo "[startup] All models pulled successfully."
echo "[startup] Installed models:"
docker exec "$OLLAMA_CONTAINER" ollama list
