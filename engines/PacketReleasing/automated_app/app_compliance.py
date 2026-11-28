"""Single-START Automated Appraiser Workflow Monitoring & Compliance Application.

Run from project root:
    .\\.venv\\Scripts\\python.exe automated_app/app_compliance.py
Or from inside automated_app/:
    ..\\.venv\\Scripts\\python.exe app_compliance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
for d in (_APP_DIR, _ROOT_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from workflow_monitor.ui_window import main

if __name__ == "__main__":
    main()

