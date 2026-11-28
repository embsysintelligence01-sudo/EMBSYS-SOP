"""One authoritative lifecycle controller; commands serialized across API and hardware."""

import json
import os
from pathlib import Path
import queue
import threading
import time
from collections import deque
from core.store import Conflict, now, uid
from core.states import SOPState
from services.evidence import attempt_directory, publish


class Controller:
    def __init__(self, store, settings, host_factory=None):
        self.store, self.settings = store, settings
        self.lock = threading.RLock()
        self.hosts = {}
        self.all_hosts = set()
        self.idle_hosts = []
        self.active = None
        self.fault = None
        self.last_command = None
        self.health = {}
        self.metrics_history = deque(maxlen=3600)
        self.stopping = False
        self.events = queue.Queue(maxsize=256)
        self.store.recover()
        from services.recovery import recover_publications

        recover_publications(self.store)
        if host_factory is None:
            from services.engines import EngineProcess

            host_factory = EngineProcess
        self.host_factory = host_factory
        self.thread = None
        self.last_recovery = 0
        from services.jobs import DurablePublisher

        self.publisher = DurablePublisher(self.store, self.settings.runtime)

    def run(self):
        self.publisher.start()
        self.thread = threading.Thread(
            target=self._events, name="sop-controller-events", daemon=True
        )
        self.thread.start()

    def _events(self):
        while not self.stopping:
            try:
                host, event = self.events.get(timeout=0.2)
                with self.lock:
                    self.engine_event(host, event)
            except queue.Empty:
                with self.lock:
                    self._expire_idle()
                    if time.monotonic() - self.last_recovery > 5:
                        self.last_recovery = time.monotonic()
                        from services.recovery import recover_publications

                        recover_publications(self.store)
            except Exception as exc:
                self.fault = "Controller event error: " + str(exc)

    def receive(self, host, event):
        # Lifecycle events cannot be dropped. A full bounded queue backpressures its reader.
        self.events.put((host, event))

    def enqueue(self, loan_id, engines):
        with self.lock:
            unavailable = set(engines) - set(self.settings.enabled)
            if unavailable:
                raise Conflict(
                    "Workflow not enabled after validation: "
                    + ", ".join(sorted(unavailable))
                )
            return self.store.enqueue(loan_id, engines)

    def remove(self, activity_id):
        with self.lock:
            self.store.remove(activity_id)
            for sid, host in list(self.hosts.items()):
                needed = any(
                    a["session_id"] == sid
                    and a["status"]
                    in ("PENDING", "STARTING", "RUNNING", "STOPPING", "FINALIZING")
                    for a in self.store.queue()
                )
                if not needed:
                    host.close()
                    self.hosts.pop(sid, None)

    def state(self):
        with self.lock:
            queue_rows = self.store.queue()
            next_row = self.store.next_activity()
            interrupted = (
                next(
                    (r for r in queue_rows if r["status"] in ("INTERRUPTED", "ERROR")),
                    None,
                )
                if next_row is None
                else None
            )
            jobs = self.store.rows(
                "SELECT * FROM jobs WHERE status IN ('QUEUED','RUNNING','INTERRUPTED','ERROR')"
            )
            active = dict(self.active) if self.active else None
            camera_fault = bool(
                active
                and active["status"] == "RUNNING"
                and self.health.get(active["session_id"], {}).get("camera_connected")
                is False
            )
            publication_fault = any(j["status"] == "ERROR" for j in jobs)
            if active:
                state = active["status"]
            elif self.fault or interrupted:
                state = "ERROR" if self.fault else "INTERRUPTED"
            elif next_row:
                state = "READY"
            elif any(j["status"] in ("RUNNING", "QUEUED") for j in jobs):
                state = "FINALIZING"
            else:
                state = "IDLE"
            busy_background = sum(
                1 for j in jobs if j["status"] in ("RUNNING", "QUEUED")
            )
            engine_background = sum(
                1
                for j in jobs
                if j["status"] in ("RUNNING", "QUEUED")
                and j["kind"] != "REPORT_PUBLICATION"
            )
            ready = bool(
                next_row
                and not active
                and not self.fault
                and not interrupted
                and engine_background < self.settings.max_background
            )
            if next_row:
                family = "releasing" if next_row["engine"] == "releasing" else "pledge"
                host_ready = next_row["session_id"] in self.hosts or any(
                    getattr(host, "family", None) == family
                    for host, _ in self.idle_hosts
                )
                if not host_ready:
                    occupied = sum(
                        not getattr(host, "closed", False) for host in self.all_hosts
                    ) - len(self.idle_hosts)
                    ready = ready and occupied < self.settings.max_hosts
            if (
                state == "READY"
                and not ready
                and not active
                and not self.fault
                and not interrupted
            ):
                state = "FINALIZING"
            state = SOPState(state).value
            return dict(
                state=state,
                current=active,
                next=next_row,
                queue=queue_rows,
                jobs=jobs,
                fault=self.fault,
                enabled=list(self.settings.enabled),
                mode=self.settings.mode,
                hardware=self.settings.hardware,
                health=dict(self.health),
                controls=dict(
                    start=ready,
                    stop=bool(active and active["status"] in ("STARTING", "RUNNING")),
                    reset=bool(
                        active["status"] in ("STARTING", "RUNNING")
                        if active
                        else interrupted or self.fault
                    ),
                ),
                leds=dict(
                    ready=ready,
                    running=bool(active and active["status"] == "RUNNING"),
                    fault=bool(
                        self.fault or interrupted or camera_fault or publication_fault
                    ),
                    blink=bool(
                        state in ("STARTING", "STOPPING", "FINALIZING")
                        or busy_background
                    ),
                ),
            )

    def command(self, command_id, name, confirmed=False):
        if not command_id or len(command_id) > 128:
            raise ValueError("A bounded unique command ID is required")
        payload = dict(name=name, confirmed=confirmed)
        with self.lock:
            cached = self.store.cached_command(command_id, payload)
            if cached is not None:
                return cached
            if name == "START":
                result = self._start()
            elif name == "STOP":
                result = self._stop("STOP")
            elif name == "RESET":
                if not confirmed:
                    raise Conflict(
                        "RESET requires confirmation; previous evidence will be retained"
                    )
                result = self._reset()
            else:
                raise ValueError("Unknown hardware command")
            result.update(command_id=command_id, acknowledged=True)
            self.store.save_command(command_id, payload, result)
            self.last_command = command_id
            return result

    def _start(self):
        status = self.state()
        if not status["controls"]["start"]:
            raise Conflict(
                "START unavailable while active, interrupted, finalizing at capacity, or queue empty"
            )
        activity = self.store.next_activity()
        if activity["engine"] not in self.settings.enabled:
            raise Conflict("This workflow is disabled")
        self._expire_idle()
        sid = activity["session_id"]
        host = self.hosts.get(sid)
        family = "releasing" if activity["engine"] == "releasing" else "pledge"
        if host is None:
            cached = next(
                (
                    (h, expiry)
                    for h, expiry in self.idle_hosts
                    if getattr(h, "family", None) == family
                ),
                None,
            )
            if cached:
                host = cached[0]
                self.idle_hosts.remove(cached)
                host.session_id = sid
                host.activity = activity
                self.hosts[sid] = host
        if host is None:
            while (
                self.idle_hosts
                and sum(not getattr(h, "closed", False) for h in self.all_hosts)
                >= self.settings.max_hosts
            ):
                idle, _ = self.idle_hosts.pop(0)
                idle.close()
            if (
                sum(not getattr(h, "closed", False) for h in self.all_hosts)
                >= self.settings.max_hosts
            ):
                raise Conflict(
                    "Engine/background capacity reached; wait for evidence jobs"
                )
            host = self.host_factory(self, activity)
            self.hosts[sid] = host
            host.family = family
            self.all_hosts.add(host)
        attempt_id = uid("attempt")
        directory = attempt_directory(self.settings.runtime, activity, attempt_id)
        directory.mkdir(parents=True, exist_ok=False)
        with self.store.transaction() as db:
            ordinal = db.execute(
                "SELECT COALESCE(MAX(ordinal),0)+1 FROM attempts WHERE activity_id=?",
                (activity["id"],),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO attempts(id,activity_id,ordinal,status,started,manifest) VALUES(?,?,?,?,?,?)",
                (
                    attempt_id,
                    activity["id"],
                    ordinal,
                    "STARTING",
                    now(),
                    str(directory / "activity.json"),
                ),
            )
            db.execute(
                "UPDATE activities SET status='STARTING' WHERE id=?", (activity["id"],)
            )
            self.store.event(
                db, "START", dict(attempt_id=attempt_id, activity_id=activity["id"])
            )
        self.active = dict(
            activity, attempt_id=attempt_id, status="STARTING", directory=str(directory)
        )
        publish(
            directory / "activity.json",
            dict(
                activity=activity,
                attempt_id=attempt_id,
                ordinal=ordinal,
                workflow_status="STARTING",
                evidence_status="NOT_VERIFIED",
                source="real-engine",
                started=now(),
            ),
        )
        history = None
        if activity["engine"] == "packet" and activity["linked"]:
            rows = self.store.rows(
                "SELECT t.manifest FROM attempts t JOIN activities a ON a.id=t.activity_id WHERE a.session_id=? AND a.engine='purity' ORDER BY t.started DESC LIMIT 1",
                (sid,),
            )
            if rows:
                history = str(Path(rows[0]["manifest"]).parent / "identity.json")
        if not hasattr(host, "attempt_ids"):
            host.attempt_ids = set()
        host.attempt_ids.add(attempt_id)
        try:
            host.send(
                dict(
                    command="START",
                    activity=activity,
                    attempt_id=attempt_id,
                    directory=str(directory),
                    history=history,
                )
            )
        except Exception as exc:
            self._fail_attempt(attempt_id, str(exc))
            self.active = None
            host.close()
            self.hosts.pop(sid, None)
            raise Conflict(
                "Engine failed to start; attempt preserved: " + str(exc)
            ) from exc
        return dict(accepted=True, attempt_id=attempt_id, state="STARTING")

    def _stop(self, kind):
        if not self.active:
            raise Conflict("No live activity to stop")
        if self.active["status"] == "STOPPING":
            previous = self.store.rows(
                "SELECT stop_kind FROM attempts WHERE id=?",
                (self.active["attempt_id"],),
            )[0]["stop_kind"]
            if kind == "RESET" and previous != "RESET":
                raise Conflict("Wait for the current stop to finish before resetting")
            return dict(accepted=True, state="STOPPING", already_requested=True)
        aid = self.active["attempt_id"]
        with self.store.transaction() as db:
            db.execute(
                "UPDATE attempts SET status='STOPPING',stop_kind=?,stopped=? WHERE id=?",
                (kind, now(), aid),
            )
            db.execute(
                "UPDATE activities SET status='STOPPING' WHERE id=?",
                (self.active["id"],),
            )
            self.store.event(db, kind, dict(attempt_id=aid))
        self.active["status"] = "STOPPING"
        self.hosts[self.active["session_id"]].send(
            dict(command="STOP", attempt_id=aid, reason=kind)
        )
        return dict(accepted=True, state="STOPPING", attempt_id=aid)

    def _reset(self):
        if self.active:
            return self._stop("RESET")
        rows = [
            r for r in self.store.queue() if r["status"] in ("INTERRUPTED", "ERROR")
        ]
        if rows:
            row = rows[0]
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE activities SET status='PENDING' WHERE session_id=? AND status IN ('INTERRUPTED','ERROR')",
                    (row["session_id"],),
                )
                self.store.event(
                    db,
                    "RESET_PREPARED",
                    dict(activity_id=row["id"], previous_attempts_retained=True),
                )
        self.fault = None
        return dict(
            accepted=True, state=self.state()["state"], new_attempt_on_start=True
        )

    def _fail_attempt(self, attempt_id, error):
        with self.store.transaction() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if not row:
                return
            db.execute(
                "UPDATE attempts SET status='INTERRUPTED',evidence_status='REVIEW_REQUIRED',error=?,ended=? WHERE id=?",
                (error, now(), attempt_id),
            )
            # A late old attempt failure must not overwrite a newer reset attempt.
            latest = db.execute(
                "SELECT id FROM attempts WHERE activity_id=? ORDER BY ordinal DESC LIMIT 1",
                (row["activity_id"],),
            ).fetchone()
            background = row["status"] == "FINALIZING" and row["stop_kind"] in (
                "STOP",
                "RESET",
                "SERVICE_STOP",
            )
            if latest[0] == attempt_id and row["stop_kind"] != "RESET":
                db.execute(
                    "UPDATE activities SET status=?,evidence_status='REVIEW_REQUIRED' WHERE id=?",
                    (
                        (
                            "COMPLETED"
                            if background and row["stop_kind"] == "STOP"
                            else "INTERRUPTED"
                        ),
                        row["activity_id"],
                    ),
                )
            db.execute(
                "UPDATE jobs SET status='ERROR',detail=?,updated=? WHERE attempt_id=?",
                (error, now(), attempt_id),
            )
            self.store.event(
                db, "ENGINE_ERROR", dict(attempt_id=attempt_id, error=error)
            )
        if not background:
            self.fault = error

    def engine_event(self, host, event):
        kind = event.get("event")
        attempt_id = event.get("attempt_id")
        if kind == "host_exit":
            affected = self.store.rows(
                "SELECT t.* FROM attempts t JOIN activities a ON a.id=t.activity_id WHERE a.session_id=? AND t.status IN ('STARTING','RUNNING','STOPPING','FINALIZING')",
                (host.session_id,),
            )
            for row in affected:
                if row["id"] in getattr(host, "attempt_ids", set()):
                    self._fail_attempt(
                        row["id"], "Engine host exited; evidence retained for review"
                    )
            if self.active and self.active["attempt_id"] in getattr(
                host, "attempt_ids", set()
            ):
                self.active = None
            owned = self.hosts.get(host.session_id) is host
            if owned:
                self.hosts.pop(host.session_id, None)
            self.all_hosts.discard(host)
            self.idle_hosts = [item for item in self.idle_hosts if item[0] is not host]
            # Completed purity still needs this identity owner until sealing starts.
            for activity in self.store.queue():
                if (
                    owned
                    and not getattr(host, "closed", False)
                    and activity["session_id"] == host.session_id
                    and activity["engine"] == "packet"
                    and activity["linked"]
                    and activity["status"] == "PENDING"
                ):
                    with self.store.transaction() as db:
                        db.execute(
                            "UPDATE activities SET status='INTERRUPTED' WHERE id=?",
                            (activity["id"],),
                        )
            return
        rows = self.store.rows(
            "SELECT t.*,a.session_id,a.engine,a.linked FROM attempts t JOIN activities a ON a.id=t.activity_id WHERE t.id=?",
            (attempt_id,),
        )
        if (
            not rows
            or rows[0]["session_id"] != host.session_id
            or attempt_id not in getattr(host, "attempt_ids", set())
        ):
            return
        row = rows[0]
        if kind == "health":
            self.health[host.session_id] = {
                k: v for k, v in event.items() if k not in ("event", "attempt_id")
            }
            self.metrics_history.append(
                dict(
                    timestamp=now(),
                    session_id=host.session_id,
                    attempt_id=attempt_id,
                    **self.health[host.session_id],
                )
            )
        elif kind == "running" and row["status"] == "STARTING":
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE attempts SET status='RUNNING' WHERE id=?", (attempt_id,)
                )
                db.execute(
                    "UPDATE activities SET status='RUNNING' WHERE id=?",
                    (row["activity_id"],),
                )
            if self.active and self.active["attempt_id"] == attempt_id:
                self.active["status"] = "RUNNING"
        elif kind == "released" and row["status"] in (
            "STARTING",
            "RUNNING",
            "STOPPING",
        ):
            reset = row["stop_kind"] == "RESET"
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE attempts SET status='FINALIZING',stopped=COALESCE(stopped,?) WHERE id=?",
                    (now(), attempt_id),
                )
                db.execute(
                    "UPDATE activities SET status=? WHERE id=?",
                    ("PENDING" if reset else "FINALIZING", row["activity_id"]),
                )
                db.execute(
                    "INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                    (
                        uid("job"),
                        attempt_id,
                        "ENGINE_FINALIZATION",
                        "RUNNING",
                        "Camera released; evidence work retained",
                        now(),
                    ),
                )
                self.store.event(db, "CAMERA_RELEASED", dict(attempt_id=attempt_id))
            if self.active and self.active["attempt_id"] == attempt_id:
                self.active = None
            if reset:
                # Drain old host, but do not reuse its state for the new attempt.
                host.retiring = True
                self.hosts.pop(host.session_id, None)
        elif kind == "finished" and row["status"] in (
            "FINALIZING",
            "STOPPING",
            "RUNNING",
            "STARTING",
        ):
            # Hosts must release the acquisition barrier before publishing completion.
            if row["status"] != "FINALIZING":
                self._fail_attempt(
                    attempt_id,
                    "Finalization arrived without camera release acknowledgement",
                )
                return
            verification = (
                "REVIEW_REQUIRED"
                if row["error"]
                else event.get("evidence_status", "NOT_VERIFIED")
            )
            if verification not in (
                "VERIFIED",
                "NOT_VERIFIED",
                "INCOMPLETE",
                "REVIEW_REQUIRED",
            ):
                verification = "REVIEW_REQUIRED"
            terminal = (
                "INTERRUPTED"
                if row["error"] or row["stop_kind"] in ("RESET", "SERVICE_STOP")
                else "COMPLETED"
            )
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE attempts SET status=?,ended=?,evidence_status=? WHERE id=?",
                    (terminal, now(), verification, attempt_id),
                )
                latest = db.execute(
                    "SELECT id FROM attempts WHERE activity_id=? ORDER BY ordinal DESC LIMIT 1",
                    (row["activity_id"],),
                ).fetchone()[0]
                if latest == attempt_id and row["stop_kind"] != "RESET":
                    db.execute(
                        "UPDATE activities SET status=?,evidence_status=? WHERE id=?",
                        (
                            (
                                "COMPLETED"
                                if row["error"] and row["stop_kind"] == "STOP"
                                else terminal
                            ),
                            verification,
                            row["activity_id"],
                        ),
                    )
                db.execute(
                    "UPDATE jobs SET status='COMPLETED',updated=?,detail=? WHERE attempt_id=?",
                    (now(), json.dumps(event.get("reports", {})), attempt_id),
                )
                self.store.event(
                    db,
                    "EVIDENCE_FINALIZED",
                    dict(attempt_id=attempt_id, verification=verification),
                )
            path = Path(row["manifest"])
            manifest = json.loads(path.read_text())
            manifest.update(
                workflow_status=terminal,
                evidence_status=verification,
                engine_result=event,
                ended=now(),
            )
            publish(path, manifest)
            self.publisher.submit(attempt_id, event)
            held = not getattr(host, "retiring", False) and any(
                a["session_id"] == host.session_id
                and a["engine"] == "packet"
                and a["linked"]
                and a["status"] == "PENDING"
                for a in self.store.queue()
            )
            if not held and not (
                self.active
                and self.active["session_id"] == host.session_id
                and self.hosts.get(host.session_id) is host
            ):
                if (
                    row["engine"] == "releasing"
                    and not getattr(host, "retiring", False)
                    and self.settings.model_cache_seconds > 0
                ):
                    self.idle_hosts.append(
                        (host, time.monotonic() + self.settings.model_cache_seconds)
                    )
                else:
                    host.close()
                if self.hosts.get(host.session_id) is host:
                    self.hosts.pop(host.session_id, None)
        elif kind == "error":
            error = event.get("error", "Engine error")
            live = self.active and self.active["attempt_id"] == attempt_id
            if live:
                self.fault = error
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE attempts SET error=?,evidence_status=? WHERE id=?",
                    (error, "REVIEW_REQUIRED", attempt_id),
                )
                self.store.event(
                    db, "ENGINE_ERROR", dict(attempt_id=attempt_id, error=error)
                )
            if self.active and self.active["attempt_id"] == attempt_id:
                self._stop("ERROR")
            # Keep the acquisition owner until an actual release/exit, even after an error.

    def _expire_idle(self):
        current = time.monotonic()
        for host, expiry in list(self.idle_hosts):
            process = getattr(host, "process", None)
            if expiry <= current or (
                process is not None and process.poll() is not None
            ):
                host.close()
                self.idle_hosts.remove((host, expiry))

    def shutdown(self):
        with self.lock:
            if self.active:
                try:
                    self._stop("SERVICE_STOP")
                except Exception:
                    pass
        deadline = time.monotonic() + 30
        while self.active and time.monotonic() < deadline:
            time.sleep(0.1)
        # No kill: inherited acquisition owners preserve evidence if shutdown takes longer.
        for host in list(self.hosts.values()):
            if not self.active or host.session_id != self.active["session_id"]:
                host.close()
        for host, _ in self.idle_hosts:
            host.close()
        self.idle_hosts.clear()
        self.stopping = True
        if self.thread:
            self.thread.join(2)
        self.publisher.stop()
