"""Verification module."""

from ._imports import *

from .config import *

from .dashboard import draw_mask

# Use the standardized core implementation for robust gap analysis
from packet_cut_core import analyze_packet_gap

def draw_packet_components(
    frame,
    components,
    gap_info=None
):

    output = frame.copy()

    colors = [
        (0, 240, 100),   # Piece 1: Vibrant Green
        (255, 60, 220),  # Piece 2: Magenta
        (0, 180, 255)    # Extra pieces: Orange
    ]

    for i, item in enumerate(components[:3]):
        color = colors[i]
        output = draw_mask(
            output,
            item["mask"] * 255,
            color,
            0.28
        )
        x, y, w, h = item["bbox"]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output,
            f"Piece {i + 1}: {item['area']} px",
            (x, max(18, y - 7)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA
        )

    # Highlight verified exposed physical gap corridor
    if gap_info is not None and "gap_mask" in gap_info:
        gap_m = gap_info["gap_mask"]
        if np.any(gap_m):
            # Electric cyan overlay for confirmed physical gap
            output = draw_mask(output, gap_m * 255, (0, 255, 255), 0.50)

    return output

def gap_between_masks(
    mask_a,
    mask_b
):
    distance = cv2.distanceTransform(
        (mask_a == 0).astype(np.uint8),
        cv2.DIST_L2,
        3
    )

    values = distance[
        mask_b > 0
    ]

    return (
        float(values.min())
        if values.size
        else 0.0
    )

def get_components(
    mask,
    reference_area
):

    binary = (
        mask > 0
    ).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    min_area = max(
        MIN_PIECE_AREA_PX,
        int(reference_area * MIN_PIECE_AREA_RATIO)
    )

    items = []

    for label_id in range(1, count):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )

        if area < min_area:
            continue

        component = np.zeros_like(
            binary,
            dtype=np.uint8
        )

        component[
            labels == label_id
        ] = 255

        m = cv2.moments(
            component,
            binaryImage=True
        )

        if m["m00"] <= 0:
            continue

        items.append({

            "mask": component,

            "area": area,

            "centroid": (
                float(m["m10"] / m["m00"]),
                float(m["m01"] / m["m00"])
            ),

            "bbox": (
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_HEIGHT])
            )
        })

    items.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    return items

def save_verification_debug(output_dir, frame, packet_mask, components, gap_info, status_name, sample_idx=0):
    """
    Saves debug diagnostic package for audit and calibration:
    1. Original frame
    2. Raw packet segmentation mask
    3. Connected components visualization
    4. Piece 1 mask
    5. Piece 2 mask
    6. Gap candidate mask
    7. Validated gap overlay
    """
    debug_dir = Path(output_dir) / "verification_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)

    # 1. Original frame
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_orig.jpg"), frame)

    # 2. Raw packet segmentation mask
    if packet_mask is not None:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_pkt_mask.png"), packet_mask)

    # 3. Connected components visualization
    h, w = frame.shape[:2]
    cc_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, comp in enumerate(components):
        col = (80 * (i + 1) % 255, 120 * (i + 1) % 255, 200)
        cc_img[comp["mask"] > 0] = col
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_components.png"), cc_img)

    # 4 & 5. Piece 1 and Piece 2 individual masks
    if len(components) >= 1:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_piece1_mask.png"), components[0]["mask"])
    if len(components) >= 2:
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_piece2_mask.png"), components[1]["mask"])

    # 6. Gap candidate mask
    if gap_info is not None and "gap_mask" in gap_info and gap_info["gap_mask"] is not None:
        gap_m = (gap_info["gap_mask"] * 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_gap_candidate_mask.png"), gap_m)

    # 7. Validated gap overlay
    vis = draw_packet_components(frame, components, gap_info=gap_info)
    cv2.imwrite(str(debug_dir / f"sample_{ts}_{status_name}_gap_vis.jpg"), vis)

def signature(components):

    return [
        {
            "area": item["area"],
            "centroid": item["centroid"]
        }

        for item in sorted(
            components,
            key=lambda x: x["centroid"][0]
        )
    ]

def stable_pair(
    previous,
    current
):

    if previous is None:
        return False

    if current is None:
        return False

    if len(previous) != 2 or len(current) != 2:
        return False

    old = sorted(
        previous,
        key=lambda x: x["centroid"][0]
    )

    new = sorted(
        current,
        key=lambda x: x["centroid"][0]
    )

    for a, b in zip(old, new):

        shift = np.hypot(
            a["centroid"][0] - b["centroid"][0],
            a["centroid"][1] - b["centroid"][1]
        )

        if shift > MAX_CENTROID_SHIFT_PX:
            return False

        ratio = (
            abs(a["area"] - b["area"])
            / max(a["area"], 1)
        )

        if ratio > MAX_AREA_CHANGE_RATIO:
            return False

    return True
