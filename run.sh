#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"
LAUNCH_PYTHON="${EMBSYS_PYTHON:-python3}"
# Resolve the configured launcher interpreter using only the standard library/config reader.
if [[ -z "${EMBSYS_PYTHON:-}" ]]; then
    LAUNCH_PYTHON="$(python3 -c 'from app.config import interpreter; print(interpreter("purity"))')"
fi
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR
export PYTHONNOUSERSITE=1
exec "$LAUNCH_PYTHON" "$APP_ROOT/launcher.py" "$@"
