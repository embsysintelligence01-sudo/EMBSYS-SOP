import hashlib, json, tempfile, unittest
from pathlib import Path
import importlib.util

if importlib.util.find_spec("pypdf") is None:
    raise unittest.SkipTest("Run publication tests with the controller interpreter")
from core.store import Store, now, uid
from services.jobs import DurablePublisher
from services.evidence import publish


class PublicationTests(unittest.TestCase):
    def test_atomic_copy_and_failed_retry_preserve_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = Store(root / "queue.sqlite")
            try:
                batch = store.enqueue("TEST", ["packet"])
                aid = batch["activity_ids"][0]
                attempt = uid("attempt")
                directory = root / "attempt"
                directory.mkdir()
                manifest = directory / "activity.json"
                publish(
                    manifest,
                    {
                        "activity": {
                            "engine": "packet",
                            "loan_id": "TEST",
                            "session_id": "pledge-test",
                            "id": aid,
                        },
                        "attempt_id": attempt,
                        "workflow_status": "COMPLETED",
                        "evidence_status": "NOT_VERIFIED",
                    },
                )
                with store.transaction() as db:
                    db.execute(
                        "INSERT INTO attempts(id,activity_id,ordinal,status,started,manifest) VALUES(?,?,?,?,?,?)",
                        (attempt, aid, 1, "COMPLETED", now(), str(manifest)),
                    )
                source = directory / "original.pdf"
                from reportlab.pdfgen import canvas

                document = canvas.Canvas(str(source))
                document.drawString(40, 750, "Original evidence")
                document.save()
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                publisher = DurablePublisher(store, root)
                publisher.submit(attempt, {"reports": {"pdf": str(source)}})
                result = json.loads(manifest.read_text())
                self.assertEqual(result["evidence_status"], "NOT_VERIFIED")
                self.assertEqual(result["published_reports"][0]["sha256"], digest)
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(), digest
                )
                missing = directory / "missing.pdf"
                publisher.submit(attempt, {"reports": {"pdf": str(missing)}})
                failed = store.rows("SELECT * FROM jobs WHERE status='ERROR'")[0]
                self.assertIn("missing.pdf", failed["detail"])
                self.assertTrue(source.exists())
                document = canvas.Canvas(str(missing))
                document.drawString(40, 750, "Recovered source")
                document.save()
                publisher.execute(failed["id"])
                self.assertEqual(
                    store.rows("SELECT status FROM jobs WHERE id=?", (failed["id"],))[
                        0
                    ]["status"],
                    "COMPLETED",
                )
            finally:
                store.close()

    def test_existing_sealing_testing_report_resolves_from_packet_export(self):
        from services.jobs import pdf_paths

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packet = root / "packet"
            purity = root / "purity"
            packet.mkdir()
            purity.mkdir()
            (packet / "jewel_testing_report.pdf").write_bytes(b"fixture")
            value = {
                "directory": str(packet),
                "testing_session": {
                    "directory": str(purity),
                    "report": "jewel_testing_report.pdf",
                },
            }
            self.assertEqual(pdf_paths(value), [packet / "jewel_testing_report.pdf"])
