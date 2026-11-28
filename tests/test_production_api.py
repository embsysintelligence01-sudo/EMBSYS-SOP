import tempfile, unittest
from pathlib import Path
import importlib.util

if importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest(
        "Run API tests with runtime/environments/controller/bin/python"
    )
from fastapi.testclient import TestClient
from backend.api import create_app
from core.config import Settings
from core.store import Store
from core.controller import Controller
from test_production_core import FakeHost


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(
            runtime=Path(self.tmp.name), enabled=("purity", "packet", "releasing")
        )
        self.store = Store(self.settings.runtime / "queue.sqlite")
        self.addCleanup(self.store.close)
        self.controller = Controller(self.store, self.settings, FakeHost)
        self.client = TestClient(
            create_app(self.controller), base_url="http://localhost"
        )
        self.addCleanup(self.client.close)

    def test_queue_persistence_multiple_loans_and_linkage(self):
        for loan, engines in [
            ("GL1001", ["purity", "packet"]),
            ("GL1002", ["packet"]),
            ("GL0985", ["releasing"]),
        ]:
            self.assertEqual(
                self.client.post(
                    "/api/queue", json={"loan_id": loan, "engines": engines}
                ).status_code,
                201,
            )
        data = self.client.get("/api/state").json()
        self.assertEqual(len(data["queue"]), 4)
        self.assertEqual(data["queue"][0]["session_id"], data["queue"][1]["session_id"])
        second = Store(self.store.path)
        try:
            self.assertEqual(len(second.queue()), 4)
        finally:
            second.close()

    def test_browser_has_only_scheduling_and_no_control_or_video(self):
        page = self.client.get("/").text
        for text in [
            "Loan ID",
            "Purity Testing",
            "Packet Sealing",
            "Packet Releasing",
            "Add to Queue",
            "Current activity",
            "Next activity",
        ]:
            self.assertIn(text, page)
        for text in [
            "<video",
            "<canvas",
            "START button",
            "RESET button",
            "STOP button",
        ]:
            self.assertNotIn(text, page)
        self.assertEqual(self.client.get("/assets/../main.py").status_code, 404)

    def test_authentication_lan_and_csrf(self):
        self.settings.token = "a" * 40
        secure = TestClient(create_app(self.controller), base_url="http://localhost")
        try:
            self.assertEqual(secure.get("/api/state").status_code, 401)
            headers = {"Authorization": "Bearer " + "a" * 40}
            self.assertEqual(secure.get("/api/state", headers=headers).status_code, 200)
            self.assertEqual(
                secure.post(
                    "/api/queue",
                    headers={**headers, "Origin": "https://evil.example"},
                    json={"loan_id": "X", "engines": ["packet"]},
                ).status_code,
                403,
            )
        finally:
            secure.close()
        with self.assertRaises(ValueError):
            Settings.load(host="0.0.0.0")

    def test_ack_and_browser_close_do_not_stop_controller(self):
        self.client.post("/api/queue", json={"loan_id": "X", "engines": ["packet"]})
        body = {"name": "START", "command_id": "command-1"}
        first = self.client.post("/api/commands", json=body)
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json()["acknowledged"])
        self.assertEqual(
            self.client.post("/api/commands", json=body).json(), first.json()
        )
        self.client.close()
        self.assertIsNotNone(self.controller.active)

    def test_reset_requires_confirmation_and_pending_removal(self):
        value = self.client.post(
            "/api/queue", json={"loan_id": "X", "engines": ["packet"]}
        ).json()
        self.assertEqual(
            self.client.delete("/api/queue/" + value["activity_ids"][0]).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(
                "/api/commands", json={"name": "RESET", "command_id": "r"}
            ).status_code,
            409,
        )
