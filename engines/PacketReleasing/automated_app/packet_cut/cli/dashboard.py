"""Dashboard module."""

from ._imports import *

from .config import *

def create_dashboard(frame, stage, status, instructions, crop=None, extra_info=None):
    """
    Renders an industrial, high-clarity dashboard.
    Left: Camera viewport with live telemetry header badge.
    Right: 420px high-contrast control & monitoring panel.
    Ensures zero truncated text, crisp rendering, and full visibility.
    """
    camera = frame.copy()
    h, w = camera.shape[:2]
    panel_w = 420  # Expanded width so full sentences, coordinates, and metrics fit without clipping

    # Deep high-contrast background: BGR (22, 18, 16)
    panel = np.full((h, panel_w, 3), (22, 18, 16), dtype=np.uint8)

    # Accent color based on status
    status_lower = str(status).lower()
    if any(k in status_lower for k in ("confirm", "closed", "pass", "complete", "ready")):
        accent_color = (90, 235, 110)       # Electric Green
        badge_bg = (24, 62, 32)
    elif any(k in status_lower for k in ("detect", "running", "stabiliz", "wait", "clear")):
        accent_color = (255, 195, 45)       # Amber Gold
        badge_bg = (68, 52, 16)
    elif any(k in status_lower for k in ("fail", "unreadable", "not", "paused", "open")):
        accent_color = (75, 75, 255)        # Vivid Coral Red
        badge_bg = (32, 20, 72)
    else:
        accent_color = (245, 185, 55)       # Cyan Accent
        badge_bg = (48, 40, 22)

    # Clean border between camera & dashboard panel
    cv2.line(panel, (0, 0), (0, h), (70, 58, 52), 2)

    # 1. Header Banner Card
    draw_rounded_rect(panel, (12, 10), (panel_w - 12, 58), (38, 32, 30), radius=6, thickness=-1)
    draw_rounded_rect(panel, (12, 10), (panel_w - 12, 58), (75, 62, 56), radius=6, thickness=1)

    cv2.putText(panel, "PACKET-CUT VERIFICATION", (22, 36), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    # Status indicator dot + text
    pill_x1 = panel_w - 110
    draw_rounded_rect(panel, (pill_x1, 20), (panel_w - 20, 46), (0, 140, 230), radius=4, thickness=-1)
    cv2.putText(panel, "LIVE SYSTEM", (pill_x1 + 8, 35), cv2.FONT_HERSHEY_DUPLEX, 0.38, (10, 10, 10), 1, cv2.LINE_AA)

    # 2. Stage Card
    curr_y = 68
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 60), (32, 26, 24), radius=6, thickness=-1)
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 60), (65, 52, 46), radius=6, thickness=1)
    # Left accent bar
    cv2.rectangle(panel, (12, curr_y + 4), (17, curr_y + 56), accent_color, -1)

    cv2.putText(panel, "STAGE", (28, curr_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)
    draw_fitted_text(panel, str(stage), 28, curr_y + 45, max_w=panel_w - 55, font=cv2.FONT_HERSHEY_DUPLEX, initial_scale=0.58, color=(255, 255, 255), thickness=1)

    # 3. Status Card
    curr_y += 68
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 54), badge_bg, radius=6, thickness=-1)
    draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + 54), accent_color, radius=6, thickness=1)
    cv2.putText(panel, "STATUS", (28, curr_y + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
    draw_fitted_text(panel, str(status), 28, curr_y + 42, max_w=panel_w - 55, font=cv2.FONT_HERSHEY_DUPLEX, initial_scale=0.55, color=accent_color, thickness=1)

    # 4. Telemetry Grid Card
    curr_y += 62
    if extra_info:
        card_h = 24 + len(extra_info) * 23
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + card_h), (30, 25, 23), radius=6, thickness=-1)
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + card_h), (60, 48, 42), radius=6, thickness=1)
        cv2.putText(panel, "TELEMETRY & MODEL DATA", (24, curr_y + 17), cv2.FONT_HERSHEY_DUPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)

        row_y = curr_y + 36
        for label, val in extra_info:
            label_str = str(label)
            val_str = str(val)

            # Label on left
            cv2.putText(panel, label_str, (26, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)

            # Value color coding
            val_col = (255, 255, 255)
            if any(w in val_str.upper() for w in ("YES", "PASS", "LOADED", "FOUND", "CONFIRMED")):
                val_col = (95, 240, 115)
            elif any(w in val_str.upper() for w in ("NO", "NONE", "UNREADABLE", "FILTERED", "ERROR")):
                val_col = (115, 145, 255)

            # Right align value dynamically
            (vw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
            cv2.putText(panel, val_str, (panel_w - 26 - vw, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.46, val_col, 1, cv2.LINE_AA)
            row_y += 23

        curr_y += card_h + 8
    else:
        curr_y += 6

    # 5. Instructions Box (Always fully wrapped, no cutting off words)
    if instructions:
        # Wrap long instruction lines so every single word is visible
        wrapped_lines = []
        for line in instructions:
            line_str = str(line)
            # Check if fits
            (tw, _), _ = cv2.getTextSize(line_str, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
            if tw <= panel_w - 60:
                wrapped_lines.append(line_str)
            else:
                # Word wrap
                words = line_str.split(" ")
                cur_line = ""
                for w_item in words:
                    test_l = (cur_line + " " + w_item).strip()
                    (test_w, _), _ = cv2.getTextSize(test_l, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
                    if test_w <= panel_w - 60:
                        cur_line = test_l
                    else:
                        if cur_line:
                            wrapped_lines.append(cur_line)
                        cur_line = w_item
                if cur_line:
                    wrapped_lines.append(cur_line)

        instr_h = 24 + len(wrapped_lines) * 22
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + instr_h), (25, 30, 36), radius=6, thickness=-1)
        draw_rounded_rect(panel, (12, curr_y), (panel_w - 12, curr_y + instr_h), (45, 85, 120), radius=6, thickness=1)
        cv2.putText(panel, "INSTRUCTIONS", (24, curr_y + 17), cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 205, 255), 1, cv2.LINE_AA)

        i_y = curr_y + 36
        for line in wrapped_lines:
            cv2.circle(panel, (28, i_y - 4), 2, (0, 210, 255), -1, cv2.LINE_AA)
            cv2.putText(panel, line, (38, i_y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
            i_y += 22

        curr_y += instr_h + 8

    # 6. Number ROI Crop Card (Safe scaling and positioning at bottom)
    if crop is not None and crop.size > 0:
        ch, cw = crop.shape[:2]
        pad_x = 14
        start_y = curr_y + 2
        pad_bottom = 10
        avail_w = max(10, panel_w - 2 * pad_x)
        avail_h = max(10, h - start_y - pad_bottom - 22)

        if avail_w > 20 and avail_h > 20 and ch > 0 and cw > 0:
            card_total_h = avail_h + 22
            draw_rounded_rect(panel, (12, start_y), (panel_w - 12, start_y + card_total_h), (28, 24, 22), radius=6, thickness=-1)
            draw_rounded_rect(panel, (12, start_y), (panel_w - 12, start_y + card_total_h), (60, 48, 42), radius=6, thickness=1)
            cv2.putText(panel, "NUMBER ROI PREVIEW", (24, start_y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, (150, 220, 160), 1, cv2.LINE_AA)

            roi_avail_h = card_total_h - 26
            scale = min((avail_w - 16) / cw, (roi_avail_h - 10) / ch, 4.5)
            target_w = max(1, min(avail_w - 16, int(cw * scale)))
            target_h = max(1, min(roi_avail_h - 10, int(ch * scale)))

            resized = cv2.resize(
                crop,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST if scale >= 2.0 else cv2.INTER_LINEAR
            )

            offset_x = 12 + ((panel_w - 24) - target_w) // 2
            offset_y = start_y + 22 + (roi_avail_h - target_h) // 2

            y_end = min(h - pad_bottom, offset_y + target_h)
            x_end = min(panel_w - 12, offset_x + target_w)
            act_h = y_end - offset_y
            act_w = x_end - offset_x

            if act_h > 0 and act_w > 0:
                cv2.rectangle(panel, (offset_x - 1, offset_y - 1), (x_end, y_end), (100, 100, 100), 1)
                panel[offset_y:y_end, offset_x:x_end] = resized[:act_h, :act_w]

    # Camera Viewport HUD
    header_overlay = camera.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, 32), (12, 12, 12), -1)
    camera = cv2.addWeighted(header_overlay, 0.70, camera, 0.30, 0)
    # HUD Telemetry
    cv2.circle(camera, (15, 16), 4, (0, 255, 120), -1, cv2.LINE_AA)
    cv2.putText(camera, "LIVE CAMERA VIEW", (26, 21), cv2.FONT_HERSHEY_DUPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
    res_text = f"{w}x{h} | CPU"
    (rw, _), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
    cv2.putText(camera, res_text, (w - rw - 14, 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

    combined = np.hstack([camera, panel])
    return combined

def draw_fitted_text(img, text, x, y, max_w, font=cv2.FONT_HERSHEY_DUPLEX, initial_scale=0.55, color=(255, 255, 255), thickness=1):
    """
    Renders text cleanly without truncation. If text exceeds max_w,
    dynamically scales it down so the complete text is always visible and crisp.
    """
    scale = initial_scale
    while scale > 0.35:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_w:
            break
        scale -= 0.03
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return scale

def draw_mask(
    frame,
    mask,
    color,
    alpha=0.25
):

    if mask is None or not np.any(mask):
        return frame

    overlay = frame.copy()

    overlay[mask > 0] = color

    return cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )

def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """Draws a smooth rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return
    r = min(radius, w // 2, h // 2)
    if thickness == -1:
        # Fill
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        # Outline
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

def put_lines(
    frame,
    lines,
    color=(255, 255, 255)
):

    y = 28

    for i, line in enumerate(lines):

        cv2.putText(
            frame,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if i == 0 else 0.50,
            color if i == 0 else (255, 255, 255),
            2 if i == 0 else 1,
            cv2.LINE_AA
        )

        y += 27

def show(
    frame,
    lines,
    color=(255, 255, 255)
):
    """
    Renders an image or auto-wraps lines in the rich unified dashboard.
    Ensures seamless full-application consistency across every step.
    """
    # If the frame is already the wide combined dashboard (e.g. w > 640), display directly
    if frame.shape[1] > 640:
        cv2.imshow(WINDOW, frame)
        return cv2.waitKey(1) & 0xFF

    # If legacy lines are passed without a dashboard, wrap into the rich dashboard automatically
    stage_name = lines[0] if lines else "INSPECTION"
    status_name = "ACTIVE"
    instructions = []
    extra_items = []

    if len(lines) > 1:
        status_name = lines[1]
    if len(lines) > 2:
        instructions = lines[2:]

    dashboard = create_dashboard(
        frame,
        stage_name,
        status_name,
        instructions,
        extra_info=extra_items
    )
    cv2.imshow(WINDOW, dashboard)
    return cv2.waitKey(1) & 0xFF
