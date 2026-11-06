#!/usr/bin/env bash
# One-shot installer for Linux. The service needs an NVIDIA GPU with CUDA.
#
# Creates .venv, installs the host tools, downloads the model weights, prepares
# the local model folder, builds the Docker image and waits for /health to be ready.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "MCP Image to 3D installer"

# --- prerequisites ------------------------------------------------------------
command -v docker >/dev/null 2>&1 || {
    echo "Docker is required. Install Docker Desktop or Docker Engine first." >&2
    exit 1
}
docker compose version >/dev/null 2>&1 || {
    echo "Docker Compose v2 is required (the 'docker compose' command)." >&2
    exit 1
}

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
    PY=python
fi
command -v "$PY" >/dev/null 2>&1 || {
    echo "Python 3.10+ is required." >&2
    exit 1
}
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "Python 3.10+ is required." >&2
    exit 1
}

# --- host environment ---------------------------------------------------------
VENV_PY=.venv/bin/python
if [ -x .venv/Scripts/python.exe ]; then
    VENV_PY=.venv/Scripts/python.exe
fi
if [ ! -x "$VENV_PY" ]; then
    "$PY" -m venv .venv
    if [ -x .venv/Scripts/python.exe ]; then
        VENV_PY=.venv/Scripts/python.exe
    else
        VENV_PY=.venv/bin/python
    fi
fi

# On Windows pip retries endlessly when it cannot write its cache, and these few
# small packages do not need one.
export PIP_NO_CACHE_DIR=1
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements-dev.txt

TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$TOKEN_FILE" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
    echo
    echo "No Hugging Face token found."
    echo "Accept the DINOv3 terms first:"
    echo "  https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m"
    echo "Then log in:"
    if [ -x .venv/bin/hf ]; then
        .venv/bin/hf auth login
    elif [ -x .venv/Scripts/hf.exe ]; then
        .venv/Scripts/hf.exe auth login
    elif [ -x .venv/bin/huggingface-cli ]; then
        .venv/bin/huggingface-cli login
    elif [ -x .venv/Scripts/huggingface-cli.exe ]; then
        .venv/Scripts/huggingface-cli.exe login
    else
        echo "Neither 'hf' nor 'huggingface-cli' is available." >&2
        exit 1
    fi
fi

# --- weights ------------------------------------------------------------------
"$VENV_PY" tools/prefetch.py
"$VENV_PY" tools/prepare_local_weights.py

# --- service ------------------------------------------------------------------
docker compose up -d --build

echo "Waiting for the model to become ready..."
state=""
i=0
while [ "$i" -lt 60 ]; do
    state="$("$VENV_PY" - <<'PY'
import json
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8043/health", timeout=5) as response:
        print("ready" if json.load(response).get("ready") else "")
except Exception:
    print("")
PY
)"
    if [ "$state" = "ready" ]; then
        break
    fi
    printf '.'
    sleep 3
    i=$((i + 1))
done
echo

if [ "$state" != "ready" ]; then
    echo "The model did not become ready. Recent logs:" >&2
    docker compose logs --tail=50 mcp-image-to-3d
    exit 1
fi

echo "MCP Image to 3D is ready: http://localhost:8043"
echo "MCP endpoint:            http://localhost:8043/mcp"
