"""Portable HTML inspection report with relative image and video links."""

from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo


def local_time(value):
    try:
        return datetime.fromisoformat(value).astimezone(ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y, %I:%M:%S %p IST')
    except (ValueError, TypeError):
        return str(value or 'Not recorded')


def media_sections(report):
    """Also show images in older reports, without inventing missing video."""
    operational = report.get('operational_results', {})
    records = {entry['key']: dict(entry) for entry in report.get('evidence_media', [])}
    evidence = report.get('evidence_images', [])
    definitions = (
        ('packet_number', 'Packet number', 'ocr_identification', 'stage2_ocr_', 'identity_status'),
        ('cutting_activity', 'Cutting action — hand and scissors', 'cutting_activity', 'stage4_cutting_observed', 'status'),
        ('cut_packet', 'Cut packet', 'cutting_verification', 'stage5_cut_confirmed', 'state'),
        ('packet_mouth', 'Packet mouth', 'packet_open_closed_inspection', 'stage6_mouth_', 'mouth_status'),
        ('jewellery_count', 'Jewellery count', 'jewellery_detection_and_counting', 'stage7_provisional_', 'count_status'),
    )
    for key, title, field, tag, result_field in definitions:
        data = operational.get(field, {})
        if key == 'cutting_activity' and not data:
            data = {'status': 'CUTTING_OBSERVED' if operational.get('cutting_activity_observed') else 'NOT_VERIFIED'}
        record = records.get(key, {})
        record['key'], record['title'] = key, title
        record.setdefault('result', data.get(result_field, 'NOT_VERIFIED'))
        record.setdefault('image_path', data.get('evidence_image'))
        if 'evidence_media' not in report or key == 'cutting_activity':
            found = next((e for e in reversed(evidence) if e.get('tag', '').startswith(tag)), {})
            if not record['image_path']:
                record['image_path'] = found.get('path')
            record.setdefault('timestamp', found.get('timestamp'))
        if key == 'packet_number':
            record['detail_image_path'] = data.get('number_crop_image')
        if key == 'cut_packet':
            record.setdefault('reason', data.get('reason'))
            if record.get('result') == 'CUT_UNVERIFIED' and str(record.get('reason', '')).startswith('Stable cut evidence'):
                record['reason'] += '. Positive candidate only; full multi-frame confirmation was not reached.'
        if key == 'jewellery_count':
            record.setdefault('count', data.get('provisional_count', data.get('count', 0)))
            if report.get('flow_mode') == 'PASS_BY':
                record['title'] = 'Jewellery count — retained clear view'
        if report.get('flow_mode') == 'PASS_BY' and record['result'] in {'NOT_STARTED', 'UNKNOWN', 'NOT_CHECKED'}:
            record['result'] = 'NOT_VERIFIED'
        yield record


def render_visual_report(report, output_dir):
    output_dir = Path(output_dir).resolve()
    si = report.get('session_information', {})
    operational = report.get('operational_results', {})
    jewels = operational.get('jewellery_detection_and_counting', {})

    def text(value):
        return escape(str(value if value is not None else 'Not recorded'))

    def asset(raw):
        if not raw:
            return None
        path = Path(raw)
        # At generation time, historical paths are relative to the project root.
        resolved = path.resolve()
        if not resolved.is_file():
            resolved = (output_dir / path).resolve()
        try:
            relative = resolved.relative_to(output_dir)
        except ValueError:
            return None
        return quote(relative.as_posix(), safe='/') if resolved.is_file() else None

    cards = []
    for entry in media_sections(report):
        image = asset(entry.get('image_path'))
        video = asset(entry.get('video_path'))
        detail = asset(entry.get('detail_image_path'))
        content = f'<img loading="lazy" src="{image}" alt="{text(entry["title"])} evidence">' if image else '<p class="missing">Image not captured for this step.</p>'
        if detail:
            content += f'<p>Printed packet number</p><img class="crop" src="{detail}" alt="Printed packet number crop">'
        if video:
            content += f'<video controls preload="metadata" src="{video}">Open the clip using the download link below.</video><p><a href="{video}" download>Save video clip</a> · {text(entry.get("duration_seconds"))} seconds</p>'
        else:
            status = entry.get('video_status') or 'NOT_RECORDED'
            content += f'<p class="missing">Video: {text(status)}. No clip is claimed for this step.</p>'
        count = f' · Count: {text(entry["count"])}' if 'count' in entry else ''
        if entry.get('reason'):
            content += f'<p class="time">{text(entry["reason"])}</p>'
        cards.append(f'<section><h2>{text(entry["title"])}</h2><p class="result">{text(entry["result"])}{count}</p><p class="time">{text(local_time(entry.get("timestamp")))}</p>{content}</section>')

    alerts = report.get('observed_mistakes', [])
    warnings = [str(item.get('message') or item.get('reason') or '') for item in alerts]
    warnings.extend(str(a.get('message', '')) for a in report.get('alerts_generated', []) if a.get('active'))
    warnings.extend(str(x) for x in report.get('media_errors', []))
    warnings.extend(str(x) for x in report.get('report_export_errors', []))
    warnings.extend('Overlapping jewellery made the count uncertain. Separate the articles for verification.'
                    if 'overlap' in str(x).lower() else str(x)
                    for x in jewels.get('uncertainty_reasons', []))
    warnings = list(dict.fromkeys(w for w in warnings if w))
    warning_html = '<ul>' + ''.join(f'<li>{text(w)}</li>' for w in warnings) + '</ul>' if warnings else '<p>No additional warnings recorded.</p>'
    pid = text(si.get('packet_id', 'UNRESOLVED'))
    count = text(jewels.get('provisional_count', jewels.get('count', 0)))
    if report.get('flow_mode') == 'PASS_BY' and jewels.get('count') is None:
        count = 'Not verified'
    count_label = 'Retained clear count' if report.get('flow_mode') == 'PASS_BY' else 'Jewellery count'
    count_details = ''
    if report.get('flow_mode') == 'PASS_BY':
        visible = jewels.get('current_visible_count')
        count_details = (f'<p><b>Currently visible:</b> {text(visible) if visible is not None else "Not verified"}</p>'
                         f'<p class="time">Retained count is the highest stable simultaneously visible count. '
                         'Removing articles does not erase that evidence; this is not a cumulative inventory.</p>')
    checks = ''
    if report.get('release_checks'):
        rows = []
        for item in report['release_checks']:
            late = ' (observed later)' if item.get('recovered_late') else ''
            rows.append(f'<tr><td>{text(item["title"])}</td><td>{text(item["status"])}{late}</td>'
                        f'<td>{text(item["reason"])}</td></tr>')
        checks = ('<section><h2>Release checks — observation only</h2>'
                  '<p>Missing checks did not block other observations.</p><table><thead><tr>'
                  '<th>Check</th><th>Status</th><th>Recorded observation</th></tr></thead><tbody>'
                  + ''.join(rows) + '</tbody></table></section>')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Packet {pid} · Inspection evidence</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;background:#f1f5f9;color:#172b3a;margin:0}}main{{max-width:1100px;margin:auto;padding:28px}}
header,section,aside{{background:white;padding:24px;border:1px solid #dce3e9;border-radius:12px;margin-bottom:20px}}h1{{margin:0;font-size:30px}}h2{{margin:0 0 8px;font-size:22px}}
.meta,.grid{{display:grid;gap:18px}}.meta{{grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr))}}.grid{{grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr))}}.meta p{{margin:8px 0}}small,.time,.missing{{color:#536579}}.result{{font-weight:650;background:#edf4f8;padding:10px;border-radius:6px}}
img,video{{display:block;width:100%;max-height:460px;object-fit:contain;background:#f5f7f9;border-radius:6px;margin:12px 0}}img.crop{{max-height:130px}}a{{color:#006c80}}aside{{border-left:4px solid #c98b28}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #dce3e9;padding:10px;overflow-wrap:anywhere}}
@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}video{{display:none}}}}
</style></head><body><main>
<header><small>EMBSYS · INSPECTION EVIDENCE</small><h1>Packet {pid}</h1>
<div class="meta"><div><p><b>Session:</b> {text(si.get('session_id'))}</p><p><b>Started:</b> {text(local_time(si.get('start_timestamp')))}</p><p><b>Saved:</b> {text(local_time(si.get('completion_timestamp')))}</p></div>
<div><p><b>Packet ID status:</b> {text(si.get('packet_identity_status', 'UNRESOLVED'))}</p><p><b>Cut:</b> {text(operational.get('cutting_verification', {}).get('state', 'NOT_VERIFIED'))}</p><p><b>{count_label}:</b> {count} · {text(jewels.get('count_status', 'NOT_STARTED'))}</p>{count_details}</div></div>
<p><b>Stage reached:</b> {text(si.get('stage_reached', 'Not recorded'))}</p>
<p class="time">Packet ID status describes the OCR reading; it does not establish a match to a loan record. Provisional or uncertain counts remain labelled as such.</p></header>
{checks}<div class="grid">{''.join(cards)}</div>
<aside><h2>Warnings and uncertainty</h2>{warning_html}</aside>
<footer><p class="time">Clips contain the inspected region, without audio. Target: 4 seconds before and 3 seconds after each event, up to 640 × 480 at 5 fps. A clip may be shorter at START or STOP. During gaps in captured frames, the last observed frame is held. Keep this HTML file with its evidence and clips folders.</p></footer>
</main></body></html>'''
