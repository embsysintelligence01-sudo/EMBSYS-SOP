"""Single-frame packet-mouth classification extracted from the desktop app."""

import cv2
import numpy as np


def detect_packet_mouth(
    frame: np.ndarray,
    packet_mask: np.ndarray,
    hand_mask: np.ndarray | None = None,
    *,
    min_focus_var: float = 25.0,
    min_brightness: float = 38.0,
    max_brightness: float = 242.0,
    min_recess_contrast: float = 22.0,
    min_cavity_area_px: int = 300,
    min_open_width_px: float = 12.0,
) -> tuple[str, str, np.ndarray]:
    """Return (state, reason, cavity_mask) for an upright packet.

    Pass an OpenCV BGR image and a same-size, single-channel segmentation
    mask for the held packet. Nonzero mask pixels mark foreground.
    The optional hand mask excludes occluded pixels. The state is OPEN,
    CLOSED, or UNKNOWN. The cavity mask is uint8, with 255 for cavity pixels.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8:
        raise ValueError("Expected a nonempty uint8 BGR frame.")
    h, w = frame.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)

    if packet_mask.shape != (h, w) or (hand_mask is not None and hand_mask.shape != (h, w)):
        raise ValueError("Masks must match the frame height and width.")
    packet_mask = packet_mask > 0
    hand_mask = np.zeros((h, w), dtype=bool) if hand_mask is None else hand_mask > 0
    pkt_area = int(np.count_nonzero(packet_mask))
    if pkt_area < 2500:
        return "UNKNOWN", "Held packet too small to inspect mouth edges.", empty

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, bw, bh = cv2.boundingRect(packet_mask.astype(np.uint8))
    blur = cv2.Laplacian(gray[y:y+bh, x:x+bw], cv2.CV_64F).var()
    brightness = frame.max(axis=2)[packet_mask]
    mean_v = brightness.mean()
    if blur < min_focus_var or mean_v < min_brightness or (mean_v > max_brightness and brightness.std() < 28):
        return "UNKNOWN", f"Mouth view out of focus or poorly lit (blur={blur:.1f}, V={mean_v:.1f}).", empty

    hand_overlap = float(np.count_nonzero((packet_mask > 0) & (hand_mask > 0))) / float(max(pkt_area, 1))
    if hand_overlap > 0.35:
        return "UNKNOWN", f"Hand occludes packet mouth ({hand_overlap:.0%}).", empty

    # Exclude outer packet walls & fingers to inspect interior mouth cavity vs lip rim
    pkt_bin = (packet_mask > 0).astype(np.uint8) * 255
    inset = cv2.erode(pkt_bin, np.ones((15, 15), np.uint8))
    if np.any(hand_mask > 0):
        inset[cv2.dilate((hand_mask > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8)) > 0] = 0
    if int(np.count_nonzero(inset)) < 500:
        return "UNKNOWN", "Usable mouth ROI too small after rim/hand exclusion.", empty

    rim_ring = (pkt_bin > 0) & (inset == 0) & (hand_mask == 0)
    rim_brightness = float(np.median(gray[rim_ring])) if np.any(rim_ring) else float(np.percentile(gray[inset > 0], 85))

    # Exposed interior cavity inside separated mouth edges is visibly darker than the packet lip rim
    dark_cavity = (inset > 0) & (gray.astype(np.float32) <= (rim_brightness - min_recess_contrast))
    cavity_mask = np.zeros((h, w), dtype=np.uint8)
    cavity_mask[dark_cavity] = 255
    cavity_mask = cv2.morphologyEx(cavity_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cavity_mask = cv2.morphologyEx(cavity_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cavity_area = int(np.count_nonzero(cavity_mask))
    if cavity_area >= min_cavity_area_px:
        # Estimate the widest local opening using the distance transform.
        dist = cv2.distanceTransform(cavity_mask, cv2.DIST_L2, 5)
        opening_width_px = float(dist.max()) * 2.0
        # Verify clear lip edges bordering the cavity
        edges = cv2.Canny(gray, 45, 130)
        cavity_border = cv2.dilate(cavity_mask, np.ones((7, 7), np.uint8)) & (~cavity_mask)
        edge_ratio = float(np.count_nonzero((edges > 0) & (cavity_border > 0))) / float(max(1, np.count_nonzero(cavity_border)))
        if opening_width_px >= min_open_width_px and edge_ratio >= 0.04:
            return (
                "OPEN",
                f"Mouth edges visibly separated (opening={opening_width_px:.1f}px, exposed cavity={cavity_area}px).",
                cavity_mask,
            )

    # Check for visibly pressed-together closed mouth seam (sharp linear edge, no dark opening)
    edges = cv2.Canny(gray, 55, 150)
    inset_edges = (edges > 0) & (inset > 0)
    edge_px = int(np.count_nonzero(inset_edges))
    # A closed pressed mouth shows a focused thin seam (line segment) across the inset with NO recessed dark cavity
    lines = cv2.HoughLinesP(
        (inset_edges.astype(np.uint8) * 255),
        rho=1,
        theta=np.pi / 180.0,
        threshold=28,
        minLineLength=28,
        maxLineGap=8,
    )
    if lines is not None and len(lines) >= 1 and cavity_area < (min_cavity_area_px // 3):
        return (
            "CLOSED",
            f"Mouth edges visibly pressed together (seam detected, no exposed interior cavity; edge_px={edge_px}).",
            empty,
        )

    return "UNKNOWN", "Whole-packet view insufficient to verify mouth-edge separation.", empty
