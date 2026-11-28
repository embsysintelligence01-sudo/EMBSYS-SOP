#!/usr/bin/env python3
"""Real API restart/browser/panel acceptance with an isolated recorded-video session."""

from pathlib import Path
import json, os, socket, sqlite3, subprocess, sys, time, uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import interpreter, engine_environment
import httpx


def main():
    base = ROOT / "runtime/validation" / ("http-" + uuid.uuid4().hex[:8])
    base.mkdir(parents=True)
    sample = next((ROOT / "runtime/validation").glob("releasing-*/input.avi"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = "http://127.0.0.1:" + str(port)
    log = (base / "backend.log").open("w")
    server = None
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("QT_QPA_PLATFORM", None)

    def launch():
        return subprocess.Popen(
            [
                str(ROOT / "runtime/environments/controller/bin/python"),
                str(ROOT / "main.py"),
                "--mode",
                "production",
                "--hardware",
                "virtual",
                "--port",
                str(port),
                "--runtime",
                str(base),
                "--replay-source",
                str(sample),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    def wait(predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                value = httpx.get(url + "/api/state", timeout=2).json()
                if predicate(value):
                    return value
            except (httpx.HTTPError, ValueError):
                pass
            if server and server.poll() is not None:
                raise RuntimeError(
                    "API server exited; see " + str(base / "backend.log")
                )
            time.sleep(0.2)
        raise TimeoutError("API state did not reach expected condition")

    try:
        server = launch()
        wait(lambda state: state["state"] == "IDLE")
        with httpx.Client(base_url=url, timeout=10) as browser:
            assert browser.get("/").status_code == 200
            for loan in ("REPLAY-HTTP-1", "REPLAY-HTTP-2"):
                response = browser.post(
                    "/api/queue", json={"loan_id": loan, "engines": ["releasing"]}
                )
                assert response.status_code == 201, response.text
        # Closing a real HTTP client does not stop the API or affect persistent queue records.
        assert len(wait(lambda state: len(state["queue"]) == 2)["queue"]) == 2
        payload = {"name": "START", "command_id": "http-start"}
        first = httpx.post(url + "/api/commands", json=payload, timeout=10)
        second = httpx.post(url + "/api/commands", json=payload, timeout=10)
        assert first.status_code == 200 and first.json() == second.json()
        assert (
            httpx.post(
                url + "/api/commands",
                json={"name": "START", "command_id": "another-start"},
                timeout=10,
            ).status_code
            == 409
        )
        current = wait(lambda state: state["state"] == "RUNNING")["current"]
        attempt_id = current["attempt_id"]
        panel_env = engine_environment("purity")
        panel_env["QT_QPA_PLATFORM"] = "offscreen"
        panel = subprocess.run(
            [
                str(interpreter("purity")),
                str(ROOT / "ui/virtual_panel.py"),
                "--url",
                url,
                "--smoke",
            ],
            env=panel_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        assert panel.returncode == 0
        assert (
            wait(lambda state: state["state"] == "RUNNING")["current"]["attempt_id"]
            == attempt_id
        )
        assert base.is_relative_to(ROOT / "runtime/validation")
        server.kill()
        server.wait(timeout=10)  # Deliberate crash of this isolated test API only.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not list(
            base.rglob("engine_result.json")
        ):
            time.sleep(0.2)
        assert list(
            base.rglob("engine_result.json")
        ), "Orphan host failed to preserve its partial evidence"
        server = launch()
        state = wait(lambda state: state["state"] == "INTERRUPTED")
        assert len(state["queue"]) == 2
        response = httpx.post(
            url + "/api/commands",
            json={"name": "RESET", "command_id": "confirm-recovery", "confirmed": True},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        state = wait(lambda state: state["state"] == "READY")
        assert state["current"] is None
        db = sqlite3.connect(base / "queue.sqlite3")
        db.row_factory = sqlite3.Row
        attempt = dict(
            db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        )
        db.close()
        assert (
            attempt["status"] == "INTERRUPTED"
            and attempt["evidence_status"] == "REVIEW_REQUIRED"
        )
        result = dict(
            result="PASSED",
            runtime=str(base),
            browser_close=True,
            panel_close=True,
            repeated_start=True,
            api_crash_restart=True,
            previous_attempt_retained=True,
            physical_reboot_test=False,
        )
        (ROOT / "outputs/http-service-acceptance.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))
    finally:
        if server and server.poll() is None:
            server.terminate()
            server.wait(timeout=45)
        log.close()


if __name__ == "__main__":
    main()
