"""Standalone test script for the Packet Segmentation model ONLY (models/packet-seg.onnx).

Supports:
  - Live webcam (default):
      .\\.venv\\Scripts\\python.exe test_packet_seg.py
      .\\.venv\\Scripts\\python.exe test_packet_seg.py --camera 0 --conf 0.12
  - Single image:
      .\\.venv\\Scripts\\python.exe test_packet_seg.py --image packet_cut_outputs/baseline_frame.jpg
      .\\.venv\\Scripts\\python.exe test_packet_seg.py --image packet_cut_outputs/baseline_frame.jpg --save out.jpg --no-gui
  - Video file:
      .\\.venv\\Scripts\\python.exe test_packet_seg.py --source path/to/video.mp4

Interactive keys (GUI mode):
  Q / ESC : Quit
  S       : Save annotated snapshot
  + / -   : Increase / decrease confidence threshold by 0.05
  SPACE   : Pause / resume (video mode)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from packet_cut_core import analyze_packet_gap, extract_packet_pieces

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "packet-seg.onnx"
SNAPSHOT_DIR = PROJECT_ROOT / "packet_cut_outputs" / "packet_seg_snapshots"

PIECE_COLORS = [
    (235, 160, 40),   # Piece 1: Dodger Blue / Cyan
    (60, 220, 100),   # Piece 2: Emerald Green
    (220, 120, 240),  # Piece 3: Orchid
    (60, 200, 255),   # Piece 4: Amber
    (120, 120, 255),  # Piece 5: Coral
]


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open webcam using DirectShow on Windows first, with default backend fallback."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run_packet_segmentation(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str,
    compute_gap: bool = True,
    verbose_trace: bool = False,
) -> dict[str, Any]:
    """Run ONLY packet-seg inference and piece/gap extraction on a single BGR frame."""
    t0 = time.perf_counter()
    results = model.predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    infer_ms = (time.perf_counter() - t0) * 1000.0
    result = results[0] if results else None

    seg_info = extract_packet_pieces(
        result,
        frame.shape,
        packet_ids=None,
        min_conf=conf,
        verbose=verbose_trace,
    )

    gap_info = None
    if compute_gap and seg_info["accepted_piece_count"] >= 2:
        pieces = seg_info["pieces"]
        gap_info = analyze_packet_gap(
            pieces[0]["mask"],
            pieces[1]["mask"],
            frame=frame,
        )

    return {
        "infer_ms": infer_ms,
        "seg_info": seg_info,
        "gap_info": gap_info,
    }


def render_overlay(
    frame: np.ndarray,
    output: dict[str, Any],
    model_name: str,
    conf: float,
    imgsz: int,
    fps: float | None = None,
) -> np.ndarray:
    """Draw segmentation masks, contours, boxes, optional gap info, and HUD."""
    vis = frame.copy()
    seg_info = output["seg_info"]
    gap_info = output["gap_info"]
    pieces = seg_info["pieces"]

    # Draw translucent masks, crisp contours, bounding boxes, and centroids
    for idx, piece in enumerate(pieces):
        color = PIECE_COLORS[idx % len(PIECE_COLORS)]
        mask = piece["mask"]
        mask_bool = mask > 0
        if np.any(mask_bool):
            vis[mask_bool] = (
                vis[mask_bool].astype(np.float32) * 0.58
                + np.array(color, dtype=np.float32) * 0.42
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)

        x, y, w, h = piece["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        cx, cy = int(round(piece["centroid"][0])), int(round(piece["centroid"][1]))
        cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(vis, (cx, cy), 6, color, 2)

        label = f"P{idx + 1} conf={piece['conf']:.2f} area={piece['area']}px"
        label_y = max(22, y - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, label_y - th - 6), (x + tw + 8, label_y + 4), (20, 20, 20), -1)
        cv2.putText(
            vis,
            label,
            (x + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Draw gap corridor & measurement line when 2+ pieces are present
    if gap_info is not None:
        gap_mask = gap_info.get("gap_mask")
        if gap_mask is not None and np.any(gap_mask > 0):
            gbool = gap_mask > 0
            gap_color = np.array((0, 235, 255) if gap_info["valid"] else (0, 80, 255), dtype=np.float32)
            vis[gbool] = (vis[gbool].astype(np.float32) * 0.55 + gap_color * 0.45).astype(np.uint8)

        gap_line = gap_info.get("gap_line")
        if gap_line is not None:
            p1, p2 = gap_line
            line_color = (0, 255, 255) if gap_info["valid"] else (0, 80, 255)
            cv2.line(vis, tuple(map(int, p1)), tuple(map(int, p2)), line_color, 2, cv2.LINE_AA)
            mid = ((int(p1[0]) + int(p2[0])) // 2, (int(p1[1]) + int(p2[1])) // 2)
            cv2.putText(
                vis,
                f"gap={gap_info['min_gap']:.1f}px",
                (mid[0] + 6, max(20, mid[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                line_color,
                2,
                cv2.LINE_AA,
            )

    # Top HUD bar
    hud_lines = [
        f"Model: {model_name} | conf={conf:.2f} | imgsz={imgsz} | infer={output['infer_ms']:.1f}ms"
        + (f" | FPS={fps:.1f}" if fps is not None else ""),
        (
            f"Pieces: {seg_info['accepted_piece_count']} "
            f"(raw={seg_info['yolo_instance_count']}, rejected={seg_info['rejected_piece_count']}) "
            f"| Total area: {seg_info['total_area']}px"
        ),
    ]
    if gap_info is not None:
        status = "GAP VALID" if gap_info["valid"] else "NO VALID GAP"
        hud_lines.append(
            f"{status}: min={gap_info['min_gap']:.1f}px, cov={gap_info['coverage_pct']:.0f}%, "
            f"bg={gap_info['gap_px_count']}px"
        )

    bar_h = 14 + len(hud_lines) * 22
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], bar_h), (15, 18, 24), -1)
    vis = cv2.addWeighted(overlay, 0.72, vis, 0.28, 0)

    for i, text in enumerate(hud_lines):
        color = (235, 240, 245)
        if i == 2 and gap_info is not None:
            color = (80, 240, 120) if gap_info["valid"] else (80, 140, 255)
        cv2.putText(
            vis,
            text,
            (10, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


def print_frame_summary(output: dict[str, Any], label: str = "Frame") -> None:
    """Print concise detection summary to stdout."""
    seg = output["seg_info"]
    gap = output["gap_info"]
    print(
        f"[{label}] infer={output['infer_ms']:.1f}ms | "
        f"raw={seg['yolo_instance_count']} | "
        f"accepted={seg['accepted_piece_count']} | "
        f"rejected={seg['rejected_piece_count']} | "
        f"total_area={seg['total_area']}px"
    )
    for idx, p in enumerate(seg["pieces"], start=1):
        x, y, w, h = p["bbox"]
        cx, cy = p["centroid"]
        print(
            f"  Piece #{idx}: cls={p['cls_id']} conf={p['conf']:.3f} "
            f"area={p['area']}px bbox=({x},{y},{w},{h}) centroid=({cx:.1f},{cy:.1f})"
        )
    for r in seg["rejected_pieces"]:
        print(
            f"  Rejected #{r['index']}: conf={r['conf']:.3f} area={r['area']}px "
            f"reason={r['reason']}"
        )
    if gap is not None:
        print(
            f"  Gap check: valid={gap['valid']} min_gap={gap['min_gap']:.1f}px "
            f"coverage={gap['coverage_pct']:.1f}% bg_px={gap['gap_px_count']} "
            f"reason={gap['reason']}"
        )


def save_snapshot(image: np.ndarray, prefix: str = "packet_seg") -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{prefix}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(out_path), image)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ONLY the packet segmentation model (models/packet-seg.onnx)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to packet segmentation model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run segmentation on a single image file.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Run segmentation on a video file.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index when neither --image nor --source is provided (default: 0).",
    )
    parser.add_argument("--width", type=int, default=640, help="Webcam capture width (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="Webcam capture height (default: 480).")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.12,
        help="Confidence threshold (default: 0.12).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Model input image size (default: 320, matches packet-seg.onnx).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path to save annotated image (image mode) or video (video/camera mode).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames in video/camera mode (0 = unlimited).",
    )
    parser.add_argument(
        "--no-gap",
        action="store_true",
        help="Disable gap analysis when 2+ pieces are detected.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV preview window (useful for headless testing).",
    )
    parser.add_argument(
        "--verbose-trace",
        action="store_true",
        help="Print detailed 11-stage packet_cut_core extraction trace.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: Packet segmentation model not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading packet segmentation model ONLY: {model_path}")
    t_load = time.perf_counter()
    model = YOLO(str(model_path), task="segment")
    print(f"Model handle initialized in {(time.perf_counter() - t_load) * 1000.0:.1f}ms")

    conf = float(np.clip(args.conf, 0.01, 0.99))
    compute_gap = not args.no_gap

    # ------------------------------------------------------------------
    # Mode 1: Single Image
    # ------------------------------------------------------------------
    if args.image is not None:
        img_path = args.image.expanduser().resolve()
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"ERROR: Could not read image: {img_path}", file=sys.stderr)
            return 1

        output = run_packet_segmentation(
            model=model,
            frame=frame,
            conf=conf,
            imgsz=args.imgsz,
            device=args.device,
            compute_gap=compute_gap,
            verbose_trace=args.verbose_trace,
        )
        print_frame_summary(output, label=img_path.name)

        vis = render_overlay(frame, output, model_path.name, conf, args.imgsz)
        if args.save is not None:
            save_path = args.save.expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
            print(f"Saved annotated image to: {save_path}")

        if not args.no_gui:
            win_name = "Packet Segmentation Test (Image)"
            cv2.imshow(win_name, vis)
            print("Press any key in the image window to close (or 'S' to save snapshot).")
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("s"), ord("S")):
                snap = save_snapshot(vis, prefix=img_path.stem)
                print(f"Saved snapshot: {snap}")
            cv2.destroyAllWindows()
        return 0

    # ------------------------------------------------------------------
    # Mode 2: Video File or Live Camera
    # ------------------------------------------------------------------
    is_video = args.source is not None
    if is_video:
        src_path = args.source.expanduser().resolve()
        if not src_path.exists():
            print(f"ERROR: Video file not found: {src_path}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(src_path))
        source_label = src_path.name
    else:
        cap = open_camera(args.camera, args.width, args.height)
        source_label = f"camera:{args.camera}"

    if not cap.isOpened():
        print(f"ERROR: Could not open {source_label}", file=sys.stderr)
        return 1

    writer = None
    if args.save is not None:
        save_path = args.save.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        if not fps_out or fps_out <= 1.0:
            fps_out = 20.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps_out, (w, h))
        print(f"Recording annotated video to: {save_path}")

    win_name = f"Packet Segmentation Test ({source_label})"
    print(f"Running packet segmentation on {source_label} | conf={conf:.2f} | imgsz={args.imgsz}")
    if not args.no_gui:
        print("Controls: [Q/ESC] Quit | [S] Save snapshot | [+/-] Adjust conf | [SPACE] Pause (video)")

    frame_idx = 0
    paused = False
    last_vis = None
    fps_ema = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_video:
                        print("Reached end of video.")
                    else:
                        print("WARNING: Failed to read frame from camera.")
                    break

                frame_idx += 1
                output = run_packet_segmentation(
                    model=model,
                    frame=frame,
                    conf=conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    compute_gap=compute_gap,
                    verbose_trace=args.verbose_trace,
                )

                now = time.perf_counter()
                dt = max(now - t_prev, 1e-6)
                t_prev = now
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)

                last_vis = render_overlay(
                    frame=frame,
                    output=output,
                    model_name=model_path.name,
                    conf=conf,
                    imgsz=args.imgsz,
                    fps=fps_ema,
                )

                if writer is not None:
                    writer.write(last_vis)

                if frame_idx == 1 or frame_idx % 30 == 0 or args.no_gui:
                    print_frame_summary(output, label=f"Frame {frame_idx}")

            if not args.no_gui and last_vis is not None:
                cv2.imshow(win_name, last_vis)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("s"), ord("S")):
                    snap = save_snapshot(last_vis)
                    print(f"Saved snapshot: {snap}")
                elif key in (ord("+"), ord("=")):
                    conf = min(0.95, round(conf + 0.05, 2))
                    print(f"Updated conf -> {conf:.2f}")
                elif key in (ord("-"), ord("_")):
                    conf = max(0.05, round(conf - 0.05, 2))
                    print(f"Updated conf -> {conf:.2f}")
                elif key == ord(" ") and is_video:
                    paused = not paused
                    print("Paused" if paused else "Resumed")

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    print(f"Finished after {frame_idx} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

