#!/usr/bin/env python3
"""Single-controller service entry point; desktop clients are separate processes."""

import argparse, importlib.util, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if importlib.util.find_spec("fastapi") is None:
    managed = ROOT / "runtime/environments/controller/bin/python"
    if managed.exists():
        os.execv(str(managed), [str(managed), __file__, *sys.argv[1:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("production", "debug"), default="production")
    parser.add_argument("--hardware", choices=("virtual", "serial"), default="virtual")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument(
        "--replay-source",
        help="Recorded-video source; requires an isolated --runtime directory",
    )
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    args = parser.parse_args()
    from core.config import Settings, ROOT

    settings = Settings.load(
        mode=args.mode,
        hardware=args.hardware,
        host=args.host,
        port=args.port,
        runtime=args.runtime,
        replay_source=args.replay_source,
    )
    if args.replay_source:
        if not args.runtime or not settings.runtime.is_relative_to(
            (ROOT / "runtime/validation").resolve()
        ):
            parser.error(
                "Replay requires --runtime inside runtime/validation; production records must not be used"
            )
        if not Path(args.replay_source).is_file():
            parser.error("Replay source must be an existing recorded-video file")
        settings.enabled = ("purity", "packet", "releasing")
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not (
        args.tls_cert and args.tls_key
    ):
        parser.error("LAN access requires TLS certificate/key and an API token")
    if args.hardware == "serial":
        from hardware.serial import SerialAdapter

        SerialAdapter()
    os.umask(0o077)
    settings.runtime.mkdir(parents=True, exist_ok=True)
    from services.ownership import ExclusiveLock
    from core.store import Store
    from core.controller import Controller
    from backend.api import create_app
    import uvicorn

    with ExclusiveLock(settings.runtime / "controller.lock"):
        store = Store(settings.runtime / "queue.sqlite3")
        controller = Controller(store, settings)
        try:
            uvicorn.run(
                create_app(controller),
                host=settings.host,
                port=settings.port,
                workers=1,
                ssl_certfile=args.tls_cert,
                ssl_keyfile=args.tls_key,
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
