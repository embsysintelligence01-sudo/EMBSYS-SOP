"""Loan/session cover plus untouched source PDF pages; no inference or verification changes."""

from pathlib import Path
from io import BytesIO
import os, tempfile, textwrap


def build_common_pdf(destination, manifest, sources):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from pypdf import PdfReader, PdfWriter

    destination = Path(destination)
    buffer = BytesIO()
    page = canvas.Canvas(buffer, pagesize=A4)
    height = A4[1]
    y = height - 55

    def line(label, value, bold=False):
        nonlocal y
        page.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        for text in textwrap.wrap(str(label) + str(value), width=94) or [""]:
            if y < 55:
                page.showPage()
                y = height - 55
            page.drawString(45, y, text)
            y -= 17

    activity = manifest["activity"]
    names = {
        "purity": "Purity Testing",
        "packet": "Packet Sealing",
        "releasing": "Packet Releasing",
    }
    line("EMBSYS SOP | ", names.get(activity["engine"], activity["engine"]), True)
    y -= 12
    for label, value in [
        ("Loan ID: ", activity["loan_id"]),
        ("Session ID: ", activity["session_id"]),
        ("Activity ID: ", activity["id"]),
        ("Attempt ID: ", manifest["attempt_id"]),
        ("Execution: ", manifest["workflow_status"]),
        ("Evidence: ", manifest["evidence_status"]),
    ]:
        line(label, value)
    if activity.get("kind") == "release":
        line(
            "Historical pledge: ",
            activity.get("pledge_id")
            or "MISSING - no matching historical pledge record available",
        )
        line("Linkage: ", activity.get("linkage", "MISSING_PLEDGE_RECORD"))
    y -= 12
    line("", "Evidence verification is independent of execution completion.", True)
    line(
        "",
        "Missing or uncertain checks remain unverified. This report does not authorize a loan, a physical release or an identity match.",
    )
    event = manifest.get("engine_result", {})
    checks = event.get("check_statuses") or event.get("reports", {}).get(
        "evidence_statuses", {}
    )
    if checks:
        y -= 12
        line("Independent check outcomes", "", True)
        for key, value in checks.items():
            line(key.replace("_", " ").title() + ": ", value)
    y -= 12
    line("Source reports attached: ", len(sources))
    line(
        "",
        "Original engine evidence and source reports remain preserved. Attached pages retain their original detection and evidence results.",
    )
    page.save()
    buffer.seek(0)
    writer = PdfWriter()
    cover = PdfReader(buffer)
    writer.append(cover)
    expected = len(cover.pages)
    for source in sources:
        reader = PdfReader(str(source))
        count = len(reader.pages)
        if count < 1:
            raise ValueError("Source report has no pages: " + str(source))
        from urllib.parse import urlsplit
        from pypdf.generic import TextStringObject, NameObject

        for original_page in reader.pages:
            for reference in original_page.get("/Annots", []):
                annotation = reference.get_object()
                action = annotation.get("/A")
                if action is not None and hasattr(action, "get_object"):
                    action = action.get_object()
                if action is not None and action.get("/URI"):
                    uri = str(action["/URI"])
                    if not urlsplit(uri).scheme and not uri.startswith("#"):
                        action[NameObject("/URI")] = TextStringObject(
                            (Path(source).parent / uri).resolve().as_uri()
                        )
        writer.append(reader)
        expected += count
    fd, temporary = tempfile.mkstemp(
        prefix=".combined-", suffix=".pdf", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if len(PdfReader(temporary).pages) != expected:
            raise ValueError("Combined report page integrity check failed")
        os.replace(temporary, destination)
    finally:
        writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
    return expected
