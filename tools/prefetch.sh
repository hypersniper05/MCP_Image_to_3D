#!/usr/bin/env bash
# Download everything Idol needs into data/hf-cache.
# This is a thin wrapper around tools/prefetch.py.
set -eu
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
    PY=".venv/bin/python"
fi
if [ ! -x "$PY" ]; then
    echo "Python venv not found. Create .venv and install requirements-dev.txt first." >&2
    echo "  Windows:  .venv\\Scripts\\pip install -r requirements-dev.txt" >&2
    echo "  Linux:    .venv/bin/pip install -r requirements-dev.txt" >&2
    exit 1
fi

exec "$PY" tools/prefetch.py
