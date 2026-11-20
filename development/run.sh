#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_ROOT"
DEVELOPMENT_PYTHON="$(python3 -c 'from app.config import interpreter; print(interpreter("packet"))')"
exec "$DEVELOPMENT_PYTHON" "$APP_ROOT/development/run_legacy.py" "$@"
