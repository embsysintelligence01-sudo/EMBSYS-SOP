"""Read-only rendering of the purity worker's diagnostic snapshots."""

import time
from html import escape
from PyQt5 import QtCore, QtGui, QtWidgets
from services.diagnostics import decode_image


class PurityView:
    def __init__(self, window, tracker):
        self.window = window
        self.tracker = tracker
        self.seen = {}
        self.log_sequence = 0
        self.stones = {}
        window.hand_model_status.setText("Hand model: waiting for worker status")

    def picture(self, label, image, maximum=(360, 150)):
        if image is not None:
            label.setPixmap(
                QtGui.QPixmap.fromImage(self.tracker.to_qimage(image)).scaled(
                    *maximum, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
            )

    def stone_picture(self, kind, payload):
        image = decode_image(payload.get("image"))
        if image is not None:
            self.stones[kind] = (image, payload.get("timestamp", 0))
        canvas = QtGui.QPixmap(720, 140)
        canvas.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(canvas)
        labels = []
        for index, key in enumerate(("reference", "baseline")):
            if key not in self.stones:
                continue
            image, timestamp = self.stones[key]
            pix = QtGui.QPixmap.fromImage(self.tracker.to_qimage(image)).scaled(
                350, 140, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            painter.drawPixmap(index * 360, 0, pix)
            labels.append(
                ("Before stone" if key == "reference" else "Baseline")
                + ": CAPTURED "
                + time.strftime("%H:%M:%S", time.localtime(timestamp))
            )
        painter.end()
        self.window.before_preview.setPixmap(
            canvas.scaled(
                480, 140, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
        )
        self.window.before_instruction.setText(" | ".join(labels))

    def event(self, p):
        w = self.window
        kind = p["kind"]
        if "completed" in p:
            w.done_label.setText(
                " | ".join(f"J{jid} DONE" for jid in p["completed"])
                or "Completed jewels: none"
            )
        if kind == "hand_model":
            w.hand_model_status.setText(p["message"])
            w.hand_model_status.setToolTip(p.get("detail", ""))
        elif kind == "storage":
            w.storage_label.setText("Storage: " + p["message"])
        elif kind == "report":
            url = QtCore.QUrl.fromLocalFile(p["path"]).toString()
            w.report_label.setText(f'<a href="{escape(url)}">Open session PDF</a>')
        elif kind == "status":
            w.acid_instruction.setText(
                (p.get("message") or "") + " " + (p.get("residue_warning") or "")
            )
        elif kind == "phase":
            jid = p.get("jewel_id")
            prefix = f"J{jid} | " if jid is not None else ""
            if p.get("recording_assignment") == "PROVISIONAL":
                prefix = f"J{p.get('recording_candidate_id')} (provisional) | "
            title = self.tracker.PHASE_TITLES.get(p["phase"], p["phase"])
            if p.get("remaining") is not None:
                title += f" ({p['remaining']:.1f} s left)"
            w.phase_label.setText(prefix + "PROCESS: " + title)
            w.acid_instruction.setText(
                (p.get("message") or "") + " " + (p.get("residue_warning") or "")
            )
        elif kind in ("reference", "baseline"):
            self.stone_picture(kind, p)
        elif kind == "begin":
            w.contact_preview.clear()
            w.rub_preview.clear()
            self.stones.pop("baseline", None)
            self.stone_picture("begin", {})
            w.contact_instruction.setText("Rubbing contact: not observed")
            w.rub_instruction.setText(
                "New rubbed mark: not yet confirmed for this recording. Acid monitoring continues."
            )
        elif kind == "contact":
            self.picture(w.contact_preview, decode_image(p.get("image")))
            text = (
                "Contact + motion confirmed"
                if p.get("confirmed")
                else "Gold contact visible; rubbing unverified"
            )
            w.contact_instruction.setText(
                text + " | " + time.strftime("%H:%M:%S", time.localtime(p["timestamp"]))
            )
        elif kind == "mask":
            self.picture(w.rub_preview, decode_image(p.get("image")), (360, 180))
            w.rub_instruction.setText(
                "New rubbed mark: DETECTED; detection rectangle shown."
                + (
                    " Previous acid overlap: contamination possible."
                    if p.get("contamination_possible")
                    else ""
                )
            )
        elif kind == "mask_decision":
            text = (
                "New rubbed mark: checking in background."
                if p.get("pending")
                else "New rubbed mask: "
                + ("DETECTED" if p.get("detected") else "NOT DETECTED")
                + "."
            )
            w.rub_instruction.setText(
                text
                + " Acid baseline capture has started."
                + (
                    " Previous acid overlap: contamination possible."
                    if p.get("contamination_possible")
                    else ""
                )
            )
        elif kind == "residue":
            labels = {
                "PREVIOUS_ACID_VISIBLE": "Advisory only: previous acid is visible; testing continues.",
                "PREVIOUS_ACID_UNCERTAIN": "Advisory only: previous acid clearance is uncertain; testing continues.",
                "CHECKING_VISUAL_CLEARANCE": "Advisory only: observing previous acid; testing continues.",
                "VISUALLY_CLEARED": "Previous acid visually cleared; chemical cleanliness is not measured.",
                "NO_PREVIOUS_ACID": "Clear stone check: no previous acid recorded.",
            }
            w.acid_instruction.setText(labels.get(p["status"], p["status"]))
        elif kind == "acid_area":
            w.acid_instruction.setText(
                "ACID VERIFIED - measuring estimated acid area; original 5-second timer continues."
            )
        elif kind == "acid_detection":
            w.acid_instruction.setText(
                "Acid candidate detected; waiting for a usable baseline."
                if p.get("waiting_for_baseline")
                else (
                    "ACID VERIFIED - recording 5-second reaction."
                    if p.get("detected")
                    else (
                        "POSSIBLE ACID - checking movement and a persistent patch."
                        if p.get("provisional")
                        else "WAITING FOR ACID / CLEAR PATCH - freshness is advisory in the PDF."
                    )
                )
            )
        elif kind == "captured":
            w.acid_instruction.setText(
                "Five-second reaction captured. Processing the result."
                if p.get("capture_complete", True)
                else "Session stopped before a complete reaction; saving available evidence."
            )
        elif kind == "result":
            record = p["record"]
            if record.get("capture_complete") and record.get("prediction") is not None:
                name = (
                    f"J{record['jewel_id']}"
                    if record.get("jewel_id") is not None
                    else "Unassigned test"
                )
                if record.get("recording_assignment") == "PROVISIONAL":
                    name = f"J{record.get('recording_candidate_id')} (provisional recording)"
                w.phase_label.setText(
                    name
                    + " DONE | "
                    + self.tracker.prediction_text(record["prediction"])
                )
                notes = [record["prediction"].get("invalid_reason")]
                if record.get("acid_target_status") == "ACID_AT_DIFFERENT_LOCATION":
                    notes.append(
                        "New rubbing and acid are at different locations; application to this jewel mark was not verified."
                    )
                if (record.get("rub_previous_acid_overlap") or 0) > 0:
                    notes.append(
                        "New rubbing overlaps previous acid; contamination possible."
                    )
                w.acid_instruction.setText(
                    " ".join(n for n in notes if n) or "Acid result saved to PDF."
                )
            elif record.get("close_reason"):
                w.phase_label.setText(
                    "FINALIZED | "
                    + (
                        "Acid not confirmed"
                        if record.get("acid_onset") is None
                        else "Material result unavailable"
                    )
                )
                w.acid_instruction.setText(
                    "No acid result generated. "
                    + record["close_reason"]
                    + ". Audit evidence saved."
                )

    def update(self, data):
        rows = []
        for source in data.get("table", []):
            row = list(source)
            for index in (1, 3, 8):
                if len(row) > index:
                    row[index] = decode_image(row[index])
            rows.append(row)
        self.window.show_table(rows)
        for index, row in enumerate(rows):
            record, pending = row[6], row[7]
            text = (
                "DONE"
                if row[5]
                else (
                    "INCOMPLETE"
                    if record
                    and record.get("close_reason")
                    and not record.get("prediction")
                    else (
                        "Saved"
                        if row[4]
                        else "Recording" if pending else "Processing" if record else ""
                    )
                )
            )
            if record and record.get("recording_assignment") == "PROVISIONAL":
                text = "Provisional\n" + text
            self.window.table.item(index, 4).setText(text)
        diagnostic = data.get("purity", {})
        if diagnostic.get("directory"):
            self.window.recordings_location.setText(
                "Recordings saved automatically to: " + diagnostic["directory"]
            )
        for event in diagnostic.get("workflow", []):
            kind = event["payload"]["kind"]
            if event["sequence"] > self.seen.get(kind, 0):
                if kind == "result":
                    record = event["payload"]["record"]
                    if diagnostic.get("cycle_id") is not None or (
                        record.get("cycle_id") is not None
                        and record["cycle_id"] != diagnostic.get("latest_token")
                    ):
                        continue
                self.event(event["payload"])
                self.seen[kind] = event["sequence"]
        for event in diagnostic.get("logs", []):
            if event["sequence"] > self.log_sequence:
                item = QtWidgets.QListWidgetItem(event["message"])
                if event.get("level") == "alert":
                    item.setForeground(QtGui.QColor("#e53935"))
                self.window.log.insertItem(0, item)
                self.log_sequence = event["sequence"]
        while self.window.log.count() > 40:
            self.window.log.takeItem(40)
