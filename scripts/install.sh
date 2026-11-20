#!/usr/bin/env bash
# Reuse working environments without installing/upgrading any model libraries.
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_ROOT"
python3 scripts/prepare_runtime.py
python3 scripts/configure.py
./run.sh --smoke-ui
