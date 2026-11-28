#!/usr/bin/env bash
set -euo pipefail

standalone_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$standalone_dir/../../.." && pwd)"
python_path="${PICKUP_BOX_PYTHON:-}"
if [[ -z "$python_path" ]]; then
    python_path="$(cd -- "$project_dir" && python3 -c 'from app.config import interpreter; print(interpreter("packet"))')"
fi
exec "$python_path" "$standalone_dir/pickup_trigger_test_box.py" "$@"
