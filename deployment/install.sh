#!/usr/bin/env bash
set -euo pipefail
SOP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOP_ROOT"
python3 -m venv runtime/environments/controller
runtime/environments/controller/bin/python -m pip install -r deployment/requirements-controller.txt
python3 deployment/provision_local.py
printf '%s\n' 'Standalone local environments prepared. See docs/PRODUCTION_INSTALLATION.md for service installation.'
