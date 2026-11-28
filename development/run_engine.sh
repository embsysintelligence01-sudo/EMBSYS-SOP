#!/usr/bin/env bash
set -euo pipefail
SOP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$SOP_ROOT/development/run_engine.py" "$@"
