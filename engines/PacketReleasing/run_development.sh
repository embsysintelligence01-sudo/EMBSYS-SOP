#!/usr/bin/env bash
set -euo pipefail
SOP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$SOP_ROOT/development/run_engine.sh" releasing "$@"
