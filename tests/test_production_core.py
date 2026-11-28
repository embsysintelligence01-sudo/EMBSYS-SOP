import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from core.store import Store, Conflict
from core.config import Settings
from core.controller import Controller


class FakeHost:
    def __init__(self, controller, activity):
        self.session_id = activity["session_id"]
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(
            runtime=Path(self.temp.name), enabled=("purity", "packet", "releasing")
        )
        self.store = Store(self.settings.runtime / "queue.sqlite")
        self.addCleanup(self.store.close)
        self.controller = Controller(self.store, self.settings, FakeHost)

    def command(self, name, **kwargs):
        return self.controller.command(__import__("uuid").uuid4().hex, name, **kwargs)

    def event(self, event, **kwargs):
        active = self.controller.active
        if active:
            self.host = self.controller.hosts[active["session_id"]]
            self.attempt = active["attempt_id"]
        self.controller.engine_event(
            self.host, dict(event=event, attempt_id=self.attempt, **kwargs)
        )

    def stop(self):
        self.command("STOP")
        self.event("released")
        self.event("finished", evidence_status="NOT_VERIFIED")

    def test_linked_order_identity_owner_and_no_automatic_start(self):
        self.controller.enqueue("GL1001", ["packet", "purity"])
        self.controller.enqueue("GL1002", ["purity", "packet"])
        self.controller.enqueue("GL0985", ["releasing"])
        self.assertEqual(
            [a["engine"] for a in self.store.queue()],
            ["purity", "packet", "purity", "packet", "releasing"],
        )
        self.command("START")
        self.event("running")
        owner = self.host
        self.stop()
        self.assertIsNone(self.controller.active)
        self.assertEqual(self.controller.state()["state"], "READY")
        self.command("START")
        self.assertIs(
            self.controller.hosts[self.controller.active["session_id"]], owner
        )
        self.assertEqual(owner.messages[-1]["activity"]["engine"], "packet")
        self.stop()
        self.assertTrue(owner.closed)
        self.assertEqual(self.store.queue()[0]["evidence_status"], "NOT_VERIFIED")

    def test_duplicate_and_parallel_start(self):
        self.controller.enqueue("GL1", ["releasing"])
        first = self.controller.command("same", "START")
        self.assertEqual(first, self.controller.command("same", "START"))
        with self.assertRaises(Conflict):
            self.controller.command("same", "STOP")
        with self.assertRaises(Conflict):
            self.command("START")
        self.assertEqual(len(self.store.rows("SELECT * FROM attempts")), 1)

    def test_parallel_start_admits_one(self):
        self.controller.enqueue("GL1", ["releasing"])

        def start(i):
            try:
                self.controller.command(str(i), "START")
                return True
            except Conflict:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(start, range(12))), 1)

    def test_reset_confirmation_and_old_attempt_retained(self):
        self.controller.enqueue("GL1", ["releasing"])
        self.command("START")
        self.event("running")
        old = self.attempt
        old_host = self.host
        with self.assertRaises(Conflict):
            self.command("RESET")
        self.command("RESET", confirmed=True)
        self.event("released")
        self.command("START")
        new = self.controller.active["attempt_id"]
        self.assertNotEqual(new, old)
        self.controller.engine_event(
            old_host,
            dict(event="finished", attempt_id=old, evidence_status="INCOMPLETE"),
        )
        self.controller.engine_event(old_host, dict(event="host_exit"))
        self.assertEqual(self.controller.active["attempt_id"], new)
        self.assertEqual(len(self.store.rows("SELECT * FROM attempts")), 2)
        self.assertTrue(old_host.closed)
        self.assertEqual(
            self.store.rows("SELECT status FROM attempts WHERE id=?", (old,))[0][
                "status"
            ],
            "INTERRUPTED",
        )

    def test_crash_and_restart_preserve_audit(self):
        self.controller.enqueue("GL1", ["purity", "packet"])
        self.command("START")
        self.event("running")
        recovered = Controller(self.store, self.settings, FakeHost)
        self.assertEqual(recovered.state()["state"], "INTERRUPTED")
        self.assertEqual(
            self.store.rows("SELECT * FROM attempts")[0]["evidence_status"],
            "REVIEW_REQUIRED",
        )
        recovered.command("reset", "RESET", True)
        self.assertEqual(
            recovered.state()["state"], "READY"
        )  # one confirmed reset prepares this interrupted session

    def test_release_linkage_missing_then_actual_pledge(self):
        self.controller.enqueue("OLD", ["releasing"])
        self.assertEqual(self.store.queue()[0]["linkage"], "MISSING_PLEDGE_RECORD")
        self.command("START")
        self.event("running")
        self.stop()
        pledge = self.controller.enqueue("GL1", ["packet"])
        self.command("START")
        self.event("running")
        self.stop()
        release = self.controller.enqueue("GL1", ["releasing"])
        self.assertEqual(release["pledge_id"], pledge["session_id"])

    def test_duplicate_pending_and_linked_deletion(self):
        batch = self.controller.enqueue("GL1", ["purity", "packet"])
        with self.assertRaises(Conflict):
            self.controller.enqueue("GL1", ["purity"])
        with self.assertRaises(Conflict):
            self.store.remove(batch["activity_ids"][0])
        self.store.remove(batch["activity_ids"][1])
        self.store.remove(batch["activity_ids"][0])

    def test_finalizing_background_does_not_auto_start(self):
        self.controller.enqueue("GL1", ["releasing"])
        self.controller.enqueue("GL2", ["packet"])
        self.command("START")
        self.event("running")
        self.command("STOP")
        self.event("released")
        self.assertEqual(self.controller.state()["state"], "READY")
        self.assertIsNone(self.controller.active)
        self.assertEqual(self.store.rows("SELECT * FROM jobs")[0]["status"], "RUNNING")

    def test_late_event_cannot_confirm_another_attempt(self):
        self.controller.enqueue("GL1", ["releasing"])
        self.command("START")
        self.event("running")
        other = FakeHost(self.controller, {"session_id": "wrong"})
        self.controller.engine_event(
            other,
            dict(event="finished", attempt_id=self.attempt, evidence_status="VERIFIED"),
        )
        self.assertEqual(
            self.store.rows("SELECT * FROM attempts")[0]["status"], "RUNNING"
        )

    def test_bad_loan_and_disabled_workflow(self):
        with self.assertRaises(ValueError):
            self.controller.enqueue("../../escape", ["packet"])
        self.settings.enabled = ()
        with self.assertRaises(Conflict):
            self.controller.enqueue("GL1", ["packet"])

    def test_warm_release_cache_changes_session_and_tracker_attempt(self):
        first = self.controller.enqueue("GL1", ["releasing"])
        self.command("START")
        self.event("running")
        host = self.host
        old = self.attempt
        self.stop()
        self.assertFalse(host.closed)
        second = self.controller.enqueue("GL2", ["releasing"])
        self.command("START")
        self.assertIs(self.controller.hosts[second["session_id"]], host)
        self.assertNotEqual(old, self.controller.active["attempt_id"])
        self.assertEqual(host.messages[-1]["activity"]["loan_id"], "GL2")
        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_failed_idle_cache_is_not_reused(self):
        self.controller.enqueue("GL1", ["releasing"])
        self.command("START")
        self.event("running")
        host = self.host
        self.stop()
        self.controller.engine_event(host, {"event": "host_exit"})
        self.assertEqual(self.controller.idle_hosts, [])
        self.controller.enqueue("GL2", ["releasing"])
        self.command("START")
        self.assertIsNot(
            self.controller.hosts[self.controller.active["session_id"]], host
        )

    def test_reset_cannot_silently_replace_an_in_progress_stop(self):
        self.controller.enqueue("GL1", ["releasing"])
        self.command("START")
        self.event("running")
        self.command("STOP")
        self.assertFalse(self.controller.state()["controls"]["reset"])
        with self.assertRaises(Conflict):
            self.command("RESET", confirmed=True)

    def test_started_linked_pledge_cannot_drop_its_pending_sealing_stage(self):
        batch = self.controller.enqueue("GL1", ["purity", "packet"])
        self.command("START")
        with self.assertRaises(Conflict):
            self.controller.remove(batch["activity_ids"][1])

    def test_service_shutdown_is_interrupted_not_an_operator_completion(self):
        self.controller.enqueue("GL1", ["releasing"])
        self.command("START")
        self.event("running")
        self.controller._stop("SERVICE_STOP")
        self.event("released")
        self.event("finished", evidence_status="INCOMPLETE")
        self.assertEqual(
            self.store.rows("SELECT status FROM attempts")[0]["status"], "INTERRUPTED"
        )
        self.assertEqual(self.store.queue()[0]["status"], "INTERRUPTED")

    def test_state_persists_after_reopen(self):
        self.controller.enqueue("GL1", ["purity", "packet"])
        second = Store(self.store.path)
        try:
            self.assertEqual(len(second.queue()), 2)
        finally:
            second.close()


class PublicationRecoveryTests(unittest.TestCase):
    def test_completed_atomic_result_is_recovered_without_claiming_verification(self):
        import json
        from services.evidence import publish

        with tempfile.TemporaryDirectory() as temp:
            settings = Settings(runtime=Path(temp), enabled=("releasing",))
            store = Store(Path(temp) / "queue.sqlite")
            try:
                controller = Controller(store, settings, FakeHost)
                controller.enqueue("X", ["releasing"])
                controller.command("start", "START")
                active = controller.active
                host = controller.hosts[active["session_id"]]
                controller.command("stop", "STOP")
                controller.engine_event(
                    host, dict(event="released", attempt_id=active["attempt_id"])
                )
                path = Path(active["directory"]) / "engine_result.json"
                publish(
                    path,
                    dict(
                        event="finished",
                        attempt_id=active["attempt_id"],
                        evidence_status="VERIFIED",
                        reports={},
                    ),
                )
                recovered = Controller(store, settings, FakeHost)
                manifest = json.loads((path.parent / "activity.json").read_text())
                self.assertTrue(manifest["publication_recovered"])
                self.assertEqual(manifest["evidence_status"], "REVIEW_REQUIRED")
                self.assertEqual(
                    store.rows("SELECT * FROM jobs")[0]["status"], "RECOVERED"
                )
            finally:
                store.close()
