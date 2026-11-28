#!/usr/bin/env bash
set -euo pipefail
SOP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
systemctl --user start embsys-sop.service
"$SOP_ROOT/runtime/environments/controller/bin/python" - <<'PY'
import time
from urllib.request import urlopen
for _ in range(40):
    try:
        with urlopen('http://127.0.0.1:8765/',timeout=1) as response:
            if response.status==200:break
    except OSError:time.sleep(.25)
else:raise SystemExit('Controller did not become available')
PY
xdg-open http://127.0.0.1:8765/ >/dev/null 2>&1 &
if [[ -f "$SOP_ROOT/configs/service.env" ]]; then
    set -a
    source "$SOP_ROOT/configs/service.env"
    set +a
fi
exec "$SOP_ROOT/runtime/environments/purity/bin/python" "$SOP_ROOT/ui/virtual_panel.py"
