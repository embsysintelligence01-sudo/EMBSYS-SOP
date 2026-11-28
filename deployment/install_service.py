#!/usr/bin/env python3
"""Install the controller unit; system scope requires root, desktop panel stays separate."""

from pathlib import Path
import argparse, os, pwd, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", action="store_true")
    parser.add_argument(
        "--account",
        default=os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name,
    )
    args = parser.parse_args()
    if not args.user and os.geteuid() != 0:
        parser.error(
            "System service installation requires root: sudo python3 deployment/install_service.py --account embsys"
        )
    template = (
        ROOT
        / "deployment"
        / ("embsys-sop-user.service.in" if args.user else "embsys-sop.service.in")
    )
    text = (
        template.read_text()
        .replace("@ROOT@", str(ROOT))
        .replace("@USER@", args.account)
    )
    destination = (
        Path.home() / ".config/systemd/user"
        if args.user
        else Path("/etc/systemd/system")
    ) / "embsys-sop.service"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    prefix = ["systemctl"] + (["--user"] if args.user else [])
    subprocess.run(prefix + ["daemon-reload"], check=True)
    subprocess.run(prefix + ["enable", "--now", "embsys-sop.service"], check=True)
    print("Installed:", destination)
    if args.user:
        print(
            "For boot without desktop login, enable user lingering or install the system unit. No hardware was commissioned."
        )


if __name__ == "__main__":
    main()
