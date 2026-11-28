"""Automatic Comprehensive Final Inspection & Compliance Report Generator (Section 8).

Generates JSON and Markdown reports containing:
1. Session Information
2. Operational Results
3. Appraiser Compliance
4. Deviation Analysis
5. Final Summary (strictly derived from recorded facts; zero fabrication)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .activity_log import ActivityLogger
from .alerts import RealTimeAlertManager, utc_now_iso
from .ocr_protector import OCRIdentitySummary
from .procedure_monitor import AppraiserProcedureMonitor
from .visual_report import render_visual_report


class FinalReportGenerator:
    """Builds and persists the complete end-of-session compliance & inspection report."""

    @staticmethod
    def build_report(
        session_id: str,
        started_at: str,
        completed_at: str,
        duration_seconds: float,
        ocr_summary: OCRIdentitySummary,
        cut_result: Dict[str, Any],
        post_cut_packet_result: Dict[str, Any],
        jewellery_result: Dict[str, Any],
        procedure_monitor: AppraiserProcedureMonitor,
        alert_manager: RealTimeAlertManager,
        activity_logger: ActivityLogger,
        processing_completed: bool,
        model_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        compliance = procedure_monitor.compliance_metrics()
        deviations = [d.to_dict() for d in procedure_monitor.deviations]

        # Build factual, non-fabricated Final Summary
        review_items: List[str] = []
        auto_recovered: List[str] = []
        mistakes: List[str] = []
        unfulfilled: List[str] = []

        if ocr_summary.identity_status == "UNRESOLVED":
            review_items.append("Packet OCR identity remained UNRESOLVED; manual verification of packet number required.")
            unfulfilled.append("OCR packet identification was not completed.")
        elif ocr_summary.identity_status == "RECOVERED":
            auto_recovered.append(
                f"Packet ID '{ocr_summary.packet_id}' was recovered automatically via continuous OCR after initial omission."
            )

        if cut_result.get("state") != "CUT_CONFIRMED":
            review_items.append(f"Physical cut confirmation did not reach CUT_CONFIRMED (final state: {cut_result.get('state', 'NOT_VERIFIED')}).")
            unfulfilled.append("Confirmed physical separation of cut packet pieces was not achieved.")

        pkt_status = post_cut_packet_result.get("status", "UNKNOWN")
        if not post_cut_packet_result.get("inspection_valid", False) or pkt_status == "UNKNOWN":
            review_items.append(f"Post-cut packet OPEN/CLOSED condition remained {pkt_status} (unverified).")
            unfulfilled.append("Valid post-cut packet interior view (OPEN/CLOSED) was not confirmed.")

        jwl_status = jewellery_result.get("count_status", "NOT_STARTED")
        if jwl_status == "COUNT_UNCERTAIN":
            review_items.append("Jewellery count remained COUNT_UNCERTAIN due to overlapping or ambiguous segmentation.")
            unfulfilled.append("Unambiguous individual jewellery count was not achieved.")
        elif jwl_status not in ("CONFIRMED", "AUTO_CONFIRMED"):
            review_items.append(f"Jewellery counting did not complete stably (status: {jwl_status}).")
            unfulfilled.append("Stable final jewellery count was not completed.")

        for dev in procedure_monitor.deviations:
            mistakes.append(
                f"[{dev.deviation_type}] {dev.code}: {dev.actual_action}"
            )
            if dev.corrected:
                auto_recovered.append(
                    f"Deviation '{dev.code}' was corrected by appraiser ({dev.correction_action})."
                )
            else:
                review_items.append(
                    f"Uncorrected procedural deviation '{dev.code}': {dev.why_non_compliant}"
                )

        for step in procedure_monitor.steps:
            if step.status == "UNVERIFIED":
                unfulfilled.append(f"{step.title}: not observed/verified during session.")

        procedure_fully_followed = (
            compliance["confirmed_deviations"] == 0
            and compliance["unverified_actions"] == 0
            and compliance["correctly_followed_procedures"] == compliance["total_expected_procedures"]
        )

        narrative_parts: List[str] = []
        narrative_parts.append(
            f"Packet processing {'completed' if processing_completed else 'did NOT fully complete'} "
            f"for packet '{ocr_summary.packet_id}' (identity: {ocr_summary.identity_status}) "
            f"in {duration_seconds:.1f}s."
        )
        if procedure_fully_followed:
            narrative_parts.append("All prescribed appraiser procedures were followed in order with zero deviations.")
        else:
            narrative_parts.append(
                f"Prescribed procedure compliance: {compliance['correctly_followed_procedures']}/"
                f"{compliance['total_expected_procedures']} steps followed strictly, "
                f"{compliance['confirmed_deviations']} deviation(s) recorded "
                f"({compliance['corrected_mistakes']} corrected, "
                f"{compliance['uncorrected_deviations']} uncorrected), "
                f"{compliance['unverified_actions']} step(s) unverified."
            )
        if review_items:
            narrative_parts.append(f"{len(review_items)} item(s) require supervisor review.")
        else:
            narrative_parts.append("No unresolved items require further review.")

        final_summary = {
            "packet_processing_completed": bool(processing_completed),
            "prescribed_procedure_followed": bool(procedure_fully_followed),
            "mistakes_occurred": mistakes,
            "expectations_not_fulfilled": unfulfilled,
            "issues_recovered_automatically": auto_recovered,
            "issues_requiring_review": review_items,
            "summary_text": " ".join(narrative_parts),
        }

        return {
            "report_generated_at": utc_now_iso(),
            "session_information": {
                "session_id": session_id,
                "packet_id": ocr_summary.packet_id,
                "packet_identity_status": ocr_summary.identity_status,
                "start_timestamp": started_at,
                "completion_timestamp": completed_at,
                "total_processing_duration_seconds": round(float(duration_seconds), 2),
            },
            "operational_results": {
                "ocr_identification": ocr_summary.to_dict(),
                "cutting_verification": dict(cut_result),
                "packet_open_closed_inspection": dict(post_cut_packet_result),
                "jewellery_detection_and_counting": dict(jewellery_result),
            },
            "appraiser_compliance": compliance,
            "deviation_analysis": deviations,
            "alerts_generated": alert_manager.to_list(),
            "activity_log": activity_logger.to_list(),
            "final_summary": final_summary,
            "model_report": dict(model_report) if model_report else {},
        }

    @staticmethod
    def render_markdown(report: Dict[str, Any]) -> str:
        si = report["session_information"]
        op = report["operational_results"]
        comp = report["appraiser_compliance"]
        devs = report["deviation_analysis"]
        fs = report["final_summary"]

        ocr = op["ocr_identification"]
        cut = op["cutting_verification"]
        pkt = op["packet_open_closed_inspection"]
        jwl = op["jewellery_detection_and_counting"]

        lines: List[str] = [
            f"# EMBSYS Packet-Cut — Automated Inspection & Compliance Report",
            "",
            "## 1. Session Information",
            f"- **Session ID**: `{si['session_id']}`",
            f"- **Packet ID**: `{si['packet_id']}` (Identity Status: **{si['packet_identity_status']}**)",
            f"- **Start Timestamp**: `{si['start_timestamp']}`",
            f"- **Completion Timestamp**: `{si['completion_timestamp']}`",
            f"- **Total Processing Duration**: `{si['total_processing_duration_seconds']:.2f} s`",
            "",
            "## 2. Operational Results",
            f"- **OCR Identification**: `{ocr['packet_id']}` | Status: `{ocr['identity_status']}` | Confidence: `{ocr['confidence']:.1%}` | Attempts: `{ocr['total_attempts']}`",
            f"- **Cutting Verification**: `{cut.get('state', 'NOT_VERIFIED')}` | Pieces: `{cut.get('pieces_count', 0)}` | Min Gap: `{cut.get('min_gap_px', 0.0):.1f} px` | Reason: {cut.get('reason', '—')}",
            f"- **Post-Cut Packet Condition**: `{pkt.get('status', 'UNKNOWN')}` | Gold Visible: `{pkt.get('gold_visible')}` | Valid View: `{pkt.get('inspection_valid', False)}`",
            f"- **Jewellery Detection & Count**: Count = `{jwl.get('count') if jwl.get('count') is not None else 'NOT_VERIFIED'}` | Status = `{jwl.get('count_status', 'NOT_STARTED')}` | Reason: {jwl.get('reason', '—')}",
            "",
            "## 3. Appraiser Compliance",
            f"- **Total Expected Procedures**: {comp['total_expected_procedures']}",
            f"- **Correctly Followed Procedures**: {comp['correctly_followed_procedures']}",
            f"- **Recovered Procedures**: {comp['recovered_procedures']}",
            f"- **Confirmed Deviations**: {comp['confirmed_deviations']}",
            f"- **Skipped Procedures**: {comp['skipped_procedures']}",
            f"- **Out-of-Order Actions**: {comp['out_of_order_actions']}",
            f"- **Unverified Actions**: {comp['unverified_actions']}",
            f"- **Corrected Mistakes**: {comp['corrected_mistakes']}",
            f"- **Strict Compliance Rate**: {comp['strict_compliance_rate']:.0%}",
            "",
            "### Prescribed Procedure Checklist",
            "| # | Procedure Step | Status | Observed Action |",
            "|---:|---|---|---|",
        ]
        for st in comp.get("procedure_steps", []):
            lines.append(f"| {st['order']} | {st['title']} | **{st['status']}** | {st['actual_action']} |")

        lines.extend(["", "## 4. Deviation Analysis"])
        if not devs:
            lines.append("No procedural deviations were detected during this session.")
        else:
            for d in devs:
                lines.extend([
                    f"### {d['deviation_id']} — `{d['code']}` ({d['deviation_type']})",
                    f"- **Expected Action**: {d['expected_action']}",
                    f"- **Actual Action**: {d['actual_action']}",
                    f"- **Why Non-Compliant**: {d['why_non_compliant']}",
                    f"- **Timestamp**: `{d['timestamp']}` (Stage: `{d['stage']}`)",
                    f"- **Alert Generated**: `{d.get('alert_id') or 'None'}` — {d.get('alert_message') or '—'}",
                    f"- **Issue Corrected**: `{'YES' if d['corrected'] else 'NO'}`" + (f" ({d['correction_action']})" if d.get("correction_action") else ""),
                    f"- **Packet Processing Continued**: `{'YES' if d['processing_continued'] else 'NO'}`",
                    f"- **Evidence Image**: `{d.get('evidence_image') or 'None'}`",
                    "",
                ])

        lines.extend([
            "## 5. Final Summary",
            fs["summary_text"],
            "",
            f"- **Packet Processing Completed**: `{'YES' if fs['packet_processing_completed'] else 'NO'}`",
            f"- **Prescribed Procedure Followed**: `{'YES' if fs['prescribed_procedure_followed'] else 'NO'}`",
        ])
        if fs["mistakes_occurred"]:
            lines.append("- **Mistakes Occurred**:")
            for m in fs["mistakes_occurred"]:
                lines.append(f"  - {m}")
        if fs["expectations_not_fulfilled"]:
            lines.append("- **Expectations Not Fulfilled**:")
            for e in fs["expectations_not_fulfilled"]:
                lines.append(f"  - {e}")
        if fs["issues_recovered_automatically"]:
            lines.append("- **Recovered Automatically / Corrected**:")
            for r in fs["issues_recovered_automatically"]:
                lines.append(f"  - {r}")
        if fs["issues_requiring_review"]:
            lines.append("- **Issues Requiring Further Review**:")
            for rev in fs["issues_requiring_review"]:
                lines.append(f"  - {rev}")

        if report.get("release_checks"):
            lines.extend(["", "## Release checks — observation only", "",
                          "Missing checks did not block other observations.", "",
                          "| Check | Status | Observation |", "|---|---|---|"])
            for check in report["release_checks"]:
                reason = str(check["reason"]).replace("|", "/").replace("\n", " ")
                suffix = " (observed later)" if check.get("recovered_late") else ""
                lines.append(f"| {check['title']} | {check['status']}{suffix} | {reason} |")
        return "\n".join(lines) + "\n"

    @classmethod
    def save_reports(
        cls,
        report: Dict[str, Any],
        output_dir: Path,
        save_json: bool = True,
        save_markdown: bool = True,
    ) -> Dict[str, Optional[str]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sid = report["session_information"]["session_id"]
        pid = report["session_information"]["packet_id"]
        safe_pid = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(pid))
        stem = f"{sid}_{safe_pid}_compliance_report"

        visual = output_dir / f"{stem}.html"
        visual.write_text(render_visual_report(report, output_dir), encoding="utf-8")
        paths: Dict[str, Optional[str]] = {"json": None, "markdown": None, "html": str(visual)}
        paths["pdf"] = None
        try:
            from .pdf_report import save_pdf_report

            paths["pdf"] = save_pdf_report(report, output_dir / f"{stem}.pdf")
        except Exception as exc:
            # PDF export must not discard the JSON evidence or hold inspection.
            report.setdefault("report_export_errors", []).append(f"PDF export: {type(exc).__name__}: {exc}")
            visual.write_text(render_visual_report(report, output_dir), encoding="utf-8")
        if save_json:
            jp = output_dir / f"{stem}.json"
            jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            paths["json"] = str(jp)
        if save_markdown:
            mp = output_dir / f"{stem}.md"
            markdown = cls.render_markdown(report)
            markdown += f"\n## Images and video clips\n\n[Open visual report]({visual.name})\n"
            if paths["pdf"]:
                markdown += f"\n[Open PDF report]({Path(paths['pdf']).name})\n"
            for entry in report.get("evidence_media", []):
                markdown += f"\n### {entry['key'].replace('_', ' ').title()} — {entry['result']}\n"
                for field, label in (("image_path", "Evidence image"), ("video_path", "Video clip")):
                    raw = entry.get(field)
                    if raw:
                        try:
                            relative = Path(raw).resolve().relative_to(output_dir.resolve()).as_posix()
                            markdown += f"\n[{label}]({relative})\n"
                        except ValueError:
                            pass
            mp.write_text(markdown, encoding="utf-8")
            paths["markdown"] = str(mp)
        return paths

