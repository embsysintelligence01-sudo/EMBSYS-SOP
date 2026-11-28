"""Thread-safe PDF evidence export; videos remain separate linked files."""

from html import escape
from pathlib import Path
from threading import Lock

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .visual_report import local_time, media_sections

_FONT_LOCK = Lock()


def save_pdf_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _FONT_LOCK:
        if 'PacketSans' not in pdfmetrics.getRegisteredFontNames():
            fonts = Path(reportlab.__file__).parent / 'fonts'
            pdfmetrics.registerFont(TTFont('PacketSans', str(fonts / 'Vera.ttf')))
            pdfmetrics.registerFont(TTFont('PacketSans-Bold', str(fonts / 'VeraBd.ttf')))
            pdfmetrics.registerFontFamily('PacketSans', normal='PacketSans', bold='PacketSans-Bold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('PacketTitle', fontName='PacketSans-Bold', fontSize=20, leading=26, spaceAfter=10))
    styles.add(ParagraphStyle('PacketHeading', fontName='PacketSans-Bold', fontSize=13, leading=18, spaceAfter=6))
    styles.add(ParagraphStyle('PacketBody', fontName='PacketSans', fontSize=9, leading=13, spaceAfter=5))
    styles.add(ParagraphStyle('PacketSmall', parent=styles['PacketBody'], fontSize=8, leading=11, textColor=colors.HexColor('#506070')))

    def paragraph(text, style='PacketBody'):
        return Paragraph(text, styles[style])

    def text(value):
        return escape(str(value if value is not None else 'Not recorded'))

    def asset(raw):
        if not raw:
            return None
        candidate = Path(raw).resolve()
        if not candidate.is_file():
            candidate = (path.parent / raw).resolve()
        try:
            candidate.relative_to(path.parent.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def image_flowable(source, max_height):
        image = Image(str(source))
        scale = min(490 / image.imageWidth, max_height / image.imageHeight, 1)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = 'CENTER'
        return image

    si = report.get('session_information', {})
    operational = report.get('operational_results', {})
    jewellery = operational.get('jewellery_detection_and_counting', {})
    count_label = 'Retained clear count' if report.get('flow_mode') == 'PASS_BY' else 'Jewellery count'
    count_value = jewellery.get('provisional_count', jewellery.get('count', 0))
    count_text = 'Not verified' if count_value is None else text(count_value)
    story = [paragraph('Packet Release — Engine 3', 'PacketTitle'),
             paragraph(f"<b>Packet ID:</b> {text(si.get('packet_id', 'UNRESOLVED'))} &nbsp; "
                       f"<b>OCR status:</b> {text(si.get('packet_identity_status', 'UNRESOLVED'))}"),
             paragraph(f"<b>Session:</b> {text(si.get('session_id'))}"),
             paragraph(f"<b>Started:</b> {text(local_time(si.get('start_timestamp')))}<br/>"
                       f"<b>Saved:</b> {text(local_time(si.get('completion_timestamp')))}"),
             paragraph(f"<b>{count_label}:</b> {count_text} "
                       f"({text(jewellery.get('count_status', 'NOT_STARTED'))})<br/>"
                       f"<b>Stage reached:</b> {text(si.get('stage_reached', 'Not recorded'))}"),
             paragraph('This records inspection evidence. OCR resolution is not a database ID match or a release approval. '
                       'Unverified results and provisional counts retain their recorded status.', 'PacketSmall'), Spacer(1, 8)]
    if report.get('flow_mode') == 'PASS_BY':
        visible = jewellery.get('current_visible_count')
        story.append(paragraph(f"Currently visible: {text(visible) if visible is not None else 'Not verified'}. "
                               'Retained count is the highest stable simultaneous count, not a cumulative inventory.', 'PacketSmall'))

    for entry in media_sections(report):
        section = [paragraph(text(entry['title']), 'PacketHeading')]
        count = f" · Count: {text(entry['count'])}" if 'count' in entry else ''
        section.append(paragraph(f"<b>{text(entry['result'])}</b>{count} · {text(local_time(entry.get('timestamp')))}", 'PacketSmall'))
        source = asset(entry.get('image_path'))
        section.append(image_flowable(source, 205) if source else paragraph('Image not captured for this step.', 'PacketSmall'))
        crop = asset(entry.get('detail_image_path'))
        if crop:
            section.extend([paragraph('Printed packet number', 'PacketSmall'), image_flowable(crop, 55)])
        clip = asset(entry.get('video_path'))
        if clip:
            relative = clip.relative_to(path.parent.resolve()).as_posix()
            section.append(paragraph(f"Video: {text(relative)} ({text(entry.get('duration_seconds'))} seconds).", 'PacketSmall'))
        else:
            section.append(paragraph(f"Video: {text(entry.get('video_status', 'NOT_RECORDED'))}.", 'PacketSmall'))
        if entry.get('reason'):
            section.append(paragraph(text(entry['reason']), 'PacketSmall'))
        section.append(Spacer(1, 12))
        story.append(KeepTogether(section))

    if report.get('release_checks'):
        story.append(paragraph('Release checks — observation only', 'PacketHeading'))
        rows = [[paragraph(label, 'PacketSmall') for label in ('Check', 'Status', 'Recorded observation')]]
        for entry in report['release_checks']:
            status = entry['status'] + (' (observed later)' if entry.get('recovered_late') else '')
            rows.append([paragraph(text(entry['title']), 'PacketSmall'), paragraph(text(status), 'PacketSmall'),
                         paragraph(text(entry['reason']), 'PacketSmall')])
        table = Table(rows, colWidths=[110, 100, A4[0] - 278], repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf4f8')),
                                   ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#dce3e9'))]))
        story.extend([paragraph('Missing checks did not block other observations.', 'PacketSmall'), table, Spacer(1, 12)])

    warnings = [str(item.get('message') or item.get('reason') or '') for item in report.get('observed_mistakes', [])]
    warnings.extend(str(item.get('message', '')) for item in report.get('alerts_generated', []) if item.get('active'))
    warnings.extend(str(item) for item in report.get('media_errors', []))
    warnings.extend(str(item) for item in report.get('report_export_errors', []))
    if jewellery.get('uncertainty_recorded'):
        warnings.append('Jewellery overlap or ambiguity was recorded. Review the count evidence.')
    warnings = list(dict.fromkeys(item for item in warnings if item))
    story.append(paragraph('Warnings and video access', 'PacketHeading'))
    for warning in warnings[:8]:
        story.append(paragraph(text(warning)))
    if not warnings:
        story.append(paragraph('No additional warnings recorded.'))
    if len(warnings) > 8:
        story.append(paragraph('Additional warnings are available in the HTML/JSON report.', 'PacketSmall'))
    story.append(paragraph('Images are embedded in this PDF. Videos are separate files in the session clips folder; '
                           'open the accompanying HTML report to play them. Keep the session folder together when sharing.', 'PacketSmall'))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont('PacketSans', 8)
        canvas.setFillColor(colors.HexColor('#506070'))
        canvas.drawString(34, 20, f"Packet {si.get('packet_id', 'UNRESOLVED')} · Engine 3 inspection evidence")
        canvas.drawRightString(A4[0] - 34, 20, f'Page {document.page}')
        canvas.restoreState()

    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=34, leftMargin=34,
                                 topMargin=30, bottomMargin=34,
                                 title=f"Packet {si.get('packet_id', 'UNRESOLVED')} — Engine 3", author='EMBSYS')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(path)
