#!/usr/bin/env python3
"""Real worker lifecycle acceptance using recorded fixture video and isolated records."""

from pathlib import Path
import argparse, json, sys, time, uuid, subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.store import Store
from core.controller import Controller
from core.config import Settings
from app.config import interpreter


def wait(controller, predicate, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if controller.fault:
            raise RuntimeError(controller.fault)
        if predicate():
            return
        time.sleep(0.2)
    raise TimeoutError("Timed out: " + str(controller.state()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("releasing", "packet", "linked"), default="releasing"
    )
    parser.add_argument(
        "--scenario",
        choices=("normal", "reset", "crash", "debug-close", "cache"),
        default="normal",
    )
    args = parser.parse_args()
    base = ROOT / "runtime/validation" / (args.engine + "-" + uuid.uuid4().hex[:8])
    base.mkdir(parents=True)
    video = base / "input.avi"
    code = """import cv2,sys
image=cv2.imread(sys.argv[1]);size=(640,480) if sys.argv[3]=='releasing' else (2560,1440)
image=cv2.resize(image,size);out=cv2.VideoWriter(sys.argv[2],cv2.VideoWriter_fourcc(*'MJPG'),10,size)
assert out.isOpened()
for _ in range(300):out.write(image)
out.release()
"""
    subprocess.run(
        [
            str(interpreter("releasing")),
            "-c",
            code,
            str(ROOT / "engines/PacketReleasing/packet_cut_outputs/baseline_frame.jpg"),
            str(video),
            args.engine,
        ],
        check=True,
    )
    settings = Settings(
        runtime=base,
        enabled=("purity", "packet", "releasing"),
        replay_source=str(video),
        mode="debug" if args.scenario == "debug-close" else "production",
    )
    store = Store(base / "queue.sqlite3")
    controller = Controller(store, settings)
    controller.run()
    try:
        controller.enqueue(
            "REPLAY-" + args.engine.upper(),
            ["purity", "packet"] if args.engine == "linked" else [args.engine],
        )
        results = []
        attempts = (
            2
            if args.engine == "linked" or args.scenario in ("reset", "crash", "cache")
            else 1
        )
        for index in range(attempts):
            if index == 1 and args.scenario == "cache":
                controller.enqueue("REPLAY-CACHED-SECOND", ["releasing"])
            started = time.monotonic()
            controller.command("start-" + str(index), "START")
            wait(
                controller,
                lambda: controller.active is not None
                and controller.active["status"] == "RUNNING",
                120,
            )
            startup = time.monotonic() - started
            time.sleep(4)
            attempt_id = controller.active["attempt_id"]
            if args.scenario == "debug-close":
                from services.evidence import publish

                debug_directory = Path(controller.active["directory"]) / "debug"
                wait(controller, lambda: (debug_directory / "frame.jpg").exists(), 30)
                publish(debug_directory / "closed.json", {"closed": True})
                time.sleep(1)
                assert (
                    controller.active and controller.active["status"] == "RUNNING"
                ), "Closing debug viewer interrupted capture"
            interrupted = index == 0 and args.scenario in ("reset", "crash")
            if interrupted and args.scenario == "crash":
                host = controller.hosts[controller.active["session_id"]]
                assert (
                    base.is_relative_to(ROOT / "runtime/validation")
                    and settings.replay_source
                )
                host.process.kill()  # Deliberate crash of this isolated recorded-input test only.
                deadline = time.monotonic() + 20
                while controller.active and time.monotonic() < deadline:
                    time.sleep(0.1)
                assert controller.active is None
                assert (
                    store.rows("SELECT status FROM attempts WHERE id=?", (attempt_id,))[
                        0
                    ]["status"]
                    == "INTERRUPTED"
                )
                controller.command("recover-crash", "RESET", True)
            else:
                controller.command(
                    "stop-" + str(index),
                    "RESET" if interrupted else "STOP",
                    confirmed=interrupted,
                )
            wait(controller, lambda: controller.active is None)
            released = time.monotonic() - started - startup - 4
            terminal = "INTERRUPTED" if interrupted else "COMPLETED"
            wait(
                controller,
                lambda: store.rows(
                    "SELECT status FROM attempts WHERE id=?", (attempt_id,)
                )[0]["status"]
                == terminal,
                180,
            )
            attempt = store.rows("SELECT * FROM attempts WHERE id=?", (attempt_id,))[0]
            result = json.loads(Path(attempt["manifest"]).read_text())
            results.append(
                dict(
                    attempt=attempt,
                    host_pid=next(
                        (
                            h.process.pid
                            for h in controller.all_hosts
                            if attempt_id in getattr(h, "attempt_ids", set())
                            and h.process
                        ),
                        None,
                    ),
                    startup_seconds=round(startup, 2),
                    release_seconds=round(released, 2),
                    manifest=result,
                )
            )
        wait(
            controller,
            lambda: not store.rows(
                "SELECT id FROM jobs WHERE status IN ('QUEUED','RUNNING')"
            ),
            60,
        )
        if args.scenario == "cache":
            assert (
                results[0]["host_pid"] == results[1]["host_pid"]
            ), "Compatible release weights were unnecessarily reloaded"
        for item in results:
            manifest = json.loads(Path(item["attempt"]["manifest"]).read_text())
            item["manifest"] = manifest
            if "engine_result" in manifest and item["attempt"]["status"] == "COMPLETED":
                assert manifest.get(
                    "common_report"
                ), "Common report was not durably published"
                from pypdf import PdfReader

                pdf = Path(manifest["common_report"]["path"])
                assert len(PdfReader(str(pdf)).pages) >= 1
                assert (
                    manifest["activity"]["loan_id"]
                    in PdfReader(str(pdf)).pages[0].extract_text()
                )
        report = dict(
            engine=args.engine,
            scenario=args.scenario,
            runtime=str(base),
            results=results,
            health=controller.health,
            metrics=list(controller.metrics_history),
            physical_camera_test=False,
        )
        output = (
            ROOT
            / "outputs"
            / ("production-" + args.engine + "-" + args.scenario + "-replay.json")
        )
        output.write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                dict(
                    engine=args.engine,
                    result="PASSED",
                    output=str(output),
                    runtime=str(base),
                ),
                indent=2,
            )
        )
    finally:
        controller.shutdown()
        store.close()


if __name__ == "__main__":
    main()
