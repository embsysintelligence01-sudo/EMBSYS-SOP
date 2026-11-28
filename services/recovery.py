"""Recover durable result publications; never invent results for interrupted native work."""

import json
from pathlib import Path
from services.evidence import publish
from core.store import now, uid


def recover_publications(store):
    recovered = []
    for row in store.rows("SELECT * FROM attempts WHERE status='INTERRUPTED'"):
        if not row.get("manifest"):
            continue
        path = Path(row["manifest"])
        source = path.parent / "engine_result.json"
        if not source.is_file():
            continue
        try:
            event = json.loads(source.read_text())
            if event.get("attempt_id") != row["id"] or event.get("event") != "finished":
                continue
            manifest = json.loads(path.read_text())
            if manifest.get("publication_recovered"):
                continue
            # Preserve the engine's actual report/status, while requiring review of the interrupted execution.
            manifest.update(
                workflow_status="INTERRUPTED",
                evidence_status="REVIEW_REQUIRED",
                engine_result=event,
                publication_recovered=True,
                recovery_timestamp=now(),
            )
            publish(path, manifest)
            with store.transaction() as db:
                db.execute(
                    "UPDATE jobs SET status='RECOVERED',detail=?,updated=? WHERE attempt_id=? AND kind='ENGINE_FINALIZATION'",
                    (json.dumps(event.get("reports", {})), now(), row["id"]),
                )
                if not db.execute(
                    "SELECT 1 FROM jobs WHERE attempt_id=? AND kind='REPORT_PUBLICATION'",
                    (row["id"],),
                ).fetchone():
                    db.execute(
                        "INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                        (
                            uid("publication"),
                            row["id"],
                            "REPORT_PUBLICATION",
                            "QUEUED",
                            json.dumps(event),
                            now(),
                        ),
                    )
                store.event(
                    db, "RESULT_PUBLICATION_RECOVERED", dict(attempt_id=row["id"])
                )
            recovered.append(row["id"])
        except (OSError, ValueError, KeyError):
            continue
    return recovered
