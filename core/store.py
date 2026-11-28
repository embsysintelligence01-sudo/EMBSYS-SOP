"""Transactional queue/audit store. Execution completion never supplies verification."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def uid(prefix):
    return prefix + "_" + uuid.uuid4().hex


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,loan_id TEXT NOT NULL,kind TEXT NOT NULL,
            pledge_id TEXT REFERENCES sessions(id),linkage TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS activities(id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions(id),
            engine TEXT NOT NULL,position INTEGER NOT NULL UNIQUE,status TEXT NOT NULL DEFAULT 'PENDING',
            linked INTEGER NOT NULL DEFAULT 0,evidence_status TEXT NOT NULL DEFAULT 'NOT_VERIFIED',created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,activity_id TEXT NOT NULL REFERENCES activities(id),
            ordinal INTEGER NOT NULL,status TEXT NOT NULL,started TEXT NOT NULL,stopped TEXT,ended TEXT,
            evidence_status TEXT NOT NULL DEFAULT 'NOT_VERIFIED',stop_kind TEXT,error TEXT,manifest TEXT,
            UNIQUE(activity_id,ordinal));
        CREATE UNIQUE INDEX IF NOT EXISTS single_live_attempt ON attempts((1))
            WHERE status IN ('STARTING','RUNNING','STOPPING');
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,attempt_id TEXT NOT NULL REFERENCES attempts(id),
            kind TEXT NOT NULL,status TEXT NOT NULL,detail TEXT,updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS commands(id TEXT PRIMARY KEY,digest TEXT NOT NULL,result TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,created TEXT NOT NULL,kind TEXT NOT NULL,data TEXT NOT NULL);
        """)

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args)]

    def event(self, db, kind, data):
        db.execute(
            "INSERT INTO events(created,kind,data) VALUES(?,?,?)",
            (now(), kind, json.dumps(data)),
        )

    def enqueue(self, loan_id, engines):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", loan_id):
            raise ValueError(
                "Loan ID must contain 1–64 letters, numbers, dots, underscores or hyphens"
            )
        if (
            not engines
            or len(set(engines)) != len(engines)
            or set(engines) - {"purity", "packet", "releasing"}
        ):
            raise ValueError("Select valid activities")
        if "releasing" in engines and len(engines) != 1:
            raise ValueError("Add Packet Releasing as a separate release session")
        engines = [e for e in ("purity", "packet", "releasing") if e in engines]
        with self.transaction() as db:
            pending = db.execute(
                "SELECT 1 FROM activities a JOIN sessions s ON s.id=a.session_id WHERE s.loan_id=? AND a.status IN ('PENDING','STARTING','RUNNING','STOPPING','FINALIZING','INTERRUPTED')",
                (loan_id,),
            ).fetchone()
            if pending:
                raise Conflict(
                    "This Loan ID already has pending or unfinished activities"
                )
            sid = uid("release" if engines == ["releasing"] else "pledge")
            pledge = None
            if engines == ["releasing"]:
                pledge = db.execute(
                    "SELECT s.id FROM sessions s JOIN activities a ON a.session_id=s.id WHERE s.loan_id=? AND s.kind='pledge' AND a.engine='packet' AND a.status='COMPLETED' ORDER BY s.created DESC LIMIT 1",
                    (loan_id,),
                ).fetchone()
            pledge_id = pledge[0] if pledge else None
            db.execute(
                "INSERT INTO sessions VALUES(?,?,?,?,?,?)",
                (
                    sid,
                    loan_id,
                    "release" if engines == ["releasing"] else "pledge",
                    pledge_id,
                    (
                        "LINKED"
                        if pledge_id
                        else (
                            "MISSING_PLEDGE_RECORD"
                            if engines == ["releasing"]
                            else "PLEDGE_SESSION"
                        )
                    ),
                    now(),
                ),
            )
            position = db.execute(
                "SELECT COALESCE(MAX(position),0) FROM activities"
            ).fetchone()[0]
            ids = []
            for offset, engine in enumerate(engines, 1):
                aid = uid("activity")
                ids.append(aid)
                db.execute(
                    "INSERT INTO activities(id,session_id,engine,position,linked,created) VALUES(?,?,?,?,?,?)",
                    (
                        aid,
                        sid,
                        engine,
                        position + offset,
                        int(engines == ["purity", "packet"]),
                        now(),
                    ),
                )
            self.event(
                db, "ENQUEUED", dict(session_id=sid, loan_id=loan_id, activities=ids)
            )
        return dict(session_id=sid, activity_ids=ids, pledge_id=pledge_id)

    def queue(self):
        return self.rows(
            """SELECT a.*,s.loan_id,s.kind,s.pledge_id,s.linkage,
            (SELECT id FROM attempts t WHERE t.activity_id=a.id ORDER BY ordinal DESC LIMIT 1) attempt_id
            FROM activities a JOIN sessions s ON s.id=a.session_id ORDER BY a.position"""
        )

    def next_activity(self):
        for activity in self.queue():
            if activity["status"] in ("INTERRUPTED", "ERROR"):
                return None  # Explicit RESET/repair is required; do not silently skip interrupted custody.
            if activity["status"] == "PENDING":
                return activity
        return None

    def remove(self, activity_id):
        with self.transaction() as db:
            activity = db.execute(
                "SELECT * FROM activities WHERE id=?", (activity_id,)
            ).fetchone()
            if (
                not activity
                or activity["status"] != "PENDING"
                or db.execute(
                    "SELECT 1 FROM attempts WHERE activity_id=?", (activity_id,)
                ).fetchone()
            ):
                raise Conflict("Only activities that have never started may be removed")
            if (
                activity["linked"]
                and db.execute(
                    "SELECT 1 FROM attempts t JOIN activities sibling ON sibling.id=t.activity_id WHERE sibling.session_id=?",
                    (activity["session_id"],),
                ).fetchone()
            ):
                raise Conflict(
                    "A linked pledge has started; its pending sealing step cannot be removed"
                )
            if (
                activity["engine"] == "purity"
                and activity["linked"]
                and db.execute(
                    "SELECT 1 FROM activities WHERE session_id=? AND engine='packet' AND status!='CANCELLED'",
                    (activity["session_id"],),
                ).fetchone()
            ):
                raise Conflict("Remove the linked sealing activity first")
            db.execute(
                "UPDATE activities SET status='CANCELLED' WHERE id=?", (activity_id,)
            )
            self.event(db, "REMOVED", dict(activity_id=activity_id))

    def recover(self):
        with self.transaction() as db:
            attempts = db.execute(
                "SELECT * FROM attempts WHERE status IN ('STARTING','RUNNING','STOPPING','FINALIZING')"
            ).fetchall()
            for attempt in attempts:
                db.execute(
                    "UPDATE attempts SET status='INTERRUPTED',evidence_status='REVIEW_REQUIRED',error='Controller restarted; source evidence retained' WHERE id=?",
                    (attempt["id"],),
                )
                db.execute(
                    "UPDATE activities SET status='INTERRUPTED',evidence_status='REVIEW_REQUIRED' WHERE id=? AND status!='PENDING'",
                    (attempt["activity_id"],),
                )
            db.execute(
                "UPDATE jobs SET status='INTERRUPTED',updated=? WHERE status IN ('QUEUED','RUNNING')",
                (now(),),
            )
            # A stopped purity host cannot carry live identity objects across a process crash.
            db.execute(
                "UPDATE activities SET status='INTERRUPTED',evidence_status='REVIEW_REQUIRED' WHERE status='PENDING' AND engine='packet' AND linked=1 AND EXISTS(SELECT 1 FROM attempts t JOIN activities p ON p.id=t.activity_id WHERE p.session_id=activities.session_id AND p.engine='purity')"
            )
            if attempts:
                self.event(
                    db,
                    "RECOVERED_INTERRUPTION",
                    dict(attempt_ids=[a["id"] for a in attempts]),
                )
        return len(attempts)

    def cached_command(self, command_id, payload):
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        rows = self.rows("SELECT * FROM commands WHERE id=?", (command_id,))
        if rows:
            if rows[0]["digest"] != digest:
                raise Conflict("Command ID was reused with different contents")
            return json.loads(rows[0]["result"])
        return None

    def save_command(self, command_id, payload, result):
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        with self.transaction() as db:
            db.execute(
                "INSERT INTO commands VALUES(?,?,?,?)",
                (command_id, digest, json.dumps(result), now()),
            )

    def close(self):
        self.db.close()
