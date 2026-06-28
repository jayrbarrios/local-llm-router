#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ChatGB10 — local model router launcher
# Ollama serves every tier on :11434, so we only launch the router on :8000.
#   ./start.sh   ->  open http://<this-host>:8000 in a browser
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama not responding on :11434 — starting it in the background..."
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi
echo "Ollama is up: $(curl -s http://127.0.0.1:11434/api/version)"

echo "Starting ChatGB10 router on :8000  (open http://<this-host>:8000)"
exec python3 -m uvicorn router:app --host 0.0.0.0 --port 8000
