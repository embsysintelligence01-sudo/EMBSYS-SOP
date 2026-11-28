#!/usr/bin/env python3
"""Install separate desktop clients; the service remains independent of desktop lifetime."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = "\n".join(
    [
        "[Desktop Entry]",
        "Type=Application",
        "Name=EMBSYS SOP",
        "Comment=Loan scheduling and virtual hardware controls",
        'Exec="' + str(ROOT / "deployment/desktop_clients.sh") + '"',
        "Icon=utilities-system-monitor",
        "Terminal=false",
        "X-GNOME-Autostart-enabled=true",
        "",
    ]
)
for path in (
    Path.home() / ".config/autostart/embsys-sop.desktop",
    Path.home() / ".local/share/applications/embsys-sop.desktop",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("Installed:", path)
