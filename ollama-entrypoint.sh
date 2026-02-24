#!/usr/bin/env bash
# ollama-entrypoint.sh
#
# Starts the Ollama server, then auto-pulls required models if they are not
# already present in the volume. On subsequent restarts the models are already
# on disk so the pull step is skipped (takes ~2 seconds to check).

set -euo pipefail

MODELS=(
  "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
  "qwen2.5:7b-instruct"
)

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait until the Ollama API is accepting connections
echo "[ollama-entrypoint] Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo "[ollama-entrypoint] Ollama is ready."

# Pull each model only if it is not already downloaded
for MODEL in "${MODELS[@]}"; do
  if ollama list | grep -q "${MODEL%%:*}"; then
    echo "[ollama-entrypoint] Model already present: $MODEL — skipping pull."
  else
    echo "[ollama-entrypoint] Pulling model: $MODEL ..."
    ollama pull "$MODEL"
    echo "[ollama-entrypoint] Done: $MODEL"
  fi
done

echo "[ollama-entrypoint] All models ready. Ollama is serving."

# Keep the container alive by waiting on the Ollama process
wait $OLLAMA_PID
