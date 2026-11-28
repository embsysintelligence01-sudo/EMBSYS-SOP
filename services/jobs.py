"""One bounded durable publication worker. Copies reports atomically; never reclassifies evidence."""

import hashlib, json, os, queue, shutil, tempfile, threading
from pathlib import Path
from core.store import now, uid
from services.evidence import publish


def pdf_paths(value, base=None, root_directory=None):
    found = []
    if isinstance(value, dict):
        directory = (
            Path(value["directory"])
            if isinstance(value.get("directory"), str)
            else base
        )
        root_directory = root_directory or directory
        for key, item in value.items():
            if (
                key in ("pdf", "report")
                and isinstance(item, str)
                and item.lower().endswith(".pdf")
            ):
                path = Path(item)
                candidate = (
                    path
                    if path.is_absolute()
                    else (directory / path if directory else path)
                )
                # The existing sealing exporter stores its copied testing report
                # relative to the packet export, while retaining the original testing directory.
                if (
                    not candidate.is_file()
                    and not path.is_absolute()
                    and root_directory is not None
                    and (root_directory / path).is_file()
                ):
                    candidate = root_directory / path
                found.append(candidate)
            elif isinstance(item, (dict, list)):
                found.extend(pdf_paths(item, directory, root_directory))
    elif isinstance(value, list):
        for item in value:
            found.extend(pdf_paths(item, base, root_directory))
    return list(dict.fromkeys(found))


class DurablePublisher:
    def __init__(self, store, runtime):
        self.store = store
        self.runtime = Path(runtime)
        self.pending = queue.Queue(maxsize=32)
        self.running = False
        self.thread = None
        self.enqueued = set()
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self.loop, name="report-publication", daemon=True
        )
        self.thread.start()
        # The worker pulls durable backlog lazily; startup never waits on queued PDFs.

    def submit(self, attempt_id, result):
        job_id = uid("publication")
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                (
                    job_id,
                    attempt_id,
                    "REPORT_PUBLICATION",
                    "QUEUED",
                    json.dumps(result),
                    now(),
                ),
            )
        self.schedule(job_id)

    def schedule(self, job_id):
        if not self.running:
            self.execute(job_id)
            return
        with self.lock:
            if job_id in self.enqueued:
                return
            self.enqueued.add(job_id)
        try:
            self.pending.put_nowait(job_id)
        except queue.Full:
            with self.lock:
                self.enqueued.discard(job_id)
            # The durable QUEUED row will be picked up by the worker.

    def loop(self):
        while self.running or not self.pending.empty():
            queued = True
            try:
                job = self.pending.get(timeout=0.2)
            except queue.Empty:
                if not self.running:
                    return
                rows = self.store.rows(
                    "SELECT id FROM jobs WHERE kind='REPORT_PUBLICATION' AND status IN ('QUEUED','INTERRUPTED') ORDER BY updated LIMIT 1"
                )
                if not rows:
                    continue
                job = rows[0]["id"]
                queued = False
            try:
                state = self.store.rows("SELECT status FROM jobs WHERE id=?", (job,))
                if state and state[0]["status"] in ("QUEUED", "INTERRUPTED"):
                    self.execute(job)
            finally:
                with self.lock:
                    self.enqueued.discard(job)
                if queued:
                    self.pending.task_done()

    def execute(self, job_id):
        rows = self.store.rows(
            "SELECT j.*,a.manifest FROM jobs j JOIN attempts a ON a.id=j.attempt_id WHERE j.id=?",
            (job_id,),
        )
        if not rows:
            return
        row = rows[0]
        event = json.loads(row["detail"])
        manifest_path = Path(row["manifest"])
        with self.store.transaction() as db:
            db.execute(
                "UPDATE jobs SET status='RUNNING',updated=? WHERE id=?", (now(), job_id)
            )
        try:
            published = []
            for index, source in enumerate(
                pdf_paths(event.get("reports", {}), manifest_path.parent)
            ):
                source = source.resolve()
                if not source.is_relative_to(self.runtime.resolve()):
                    raise ValueError(
                        "Report path is outside the session runtime: " + str(source)
                    )
                with source.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise ValueError("Invalid PDF header: " + str(source))
                    stream.seek(max(0, source.stat().st_size - 1024))
                    if b"%%EOF" not in stream.read():
                        raise ValueError("Incomplete PDF: " + str(source))
                destination = (
                    manifest_path.parent / "reports" / (str(index) + "_" + source.name)
                )
                destination.parent.mkdir(exist_ok=True)
                fd, temporary = tempfile.mkstemp(
                    prefix=".publication-", dir=destination.parent
                )
                try:
                    with os.fdopen(fd, "wb") as target, source.open("rb") as original:
                        shutil.copyfileobj(original, target, 1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                published.append(
                    dict(
                        source=str(source),
                        published=str(destination),
                        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    )
                )
            manifest = json.loads(manifest_path.read_text())
            destination = manifest_path.parent / "reports"
            destination.mkdir(exist_ok=True)
            from services.reporting import build_common_pdf

            common = destination / "activity-report.pdf"
            pages = build_common_pdf(
                common, manifest, [Path(item["source"]) for item in published]
            )
            manifest["common_report"] = {
                "path": str(common),
                "pages": pages,
                "sha256": hashlib.sha256(common.read_bytes()).hexdigest(),
            }
            manifest["published_reports"] = published
            publish(manifest_path, manifest)
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE jobs SET status='COMPLETED',updated=? WHERE id=?",
                    (now(), job_id),
                )
                self.store.event(
                    db,
                    "REPORTS_PUBLISHED",
                    dict(attempt_id=row["attempt_id"], reports=published),
                )
        except Exception as exc:
            # Retain the original retry payload and record errors separately in audit events.
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE jobs SET status='ERROR',updated=? WHERE id=?",
                    (now(), job_id),
                )
                self.store.event(
                    db,
                    "REPORT_PUBLICATION_ERROR",
                    dict(attempt_id=row["attempt_id"], error=str(exc), job_id=job_id),
                )

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(10)
