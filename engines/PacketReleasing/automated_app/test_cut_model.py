"""Standalone test script for the Cut Model ONLY (models/cut-seg.pt).

Supports:
  - Live webcam (default):
      .\\.venv\\Scripts\\python.exe test_cut_model.py
      .\\.venv\\Scripts\\python.exe test_cut_model.py --camera 0 --min-conf 0.70
  - Single image:
      .\\.venv\\Scripts\\python.exe test_cut_model.py --image packet_cut_outputs/baseline_frame.jpg
      .\\.venv\\Scripts\\python.exe test_cut_model.py --image packet_cut_outputs/verification_debug/sample_1794906516686_sample_1_orig.jpg --save out.jpg --no-gui
  - Video file:
      .\\.venv\\Scripts\\python.exe test_cut_model.py --source path/to/video.mp4

Interactive keys (GUI mode):
  Q / ESC : Quit
  S       : Save annotated snapshot
  + / -   : Increase / decrease cut confirmation threshold (--min-conf) by 0.05
  ] / [   : Increase / decrease raw YOLO prediction threshold (--conf) by 0.05
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

from packet_cut.cut_model import evaluate_cut_model_prediction
from packet_cut_core import unpad_letterbox_mask

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "cut-seg.pt"
SNAPSHOT_DIR = PROJECT_ROOT / "packet_cut_outputs" / "cut_model_snapshots"


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


def _names_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def extract_cut_detections(result: Any, frame_shape: tuple[int, ...], names: dict[int, str]) -> list[dict[str, Any]]:
    """Extract boxes, masks, confidences, and areas from a YOLO cut result."""
    h, w = frame_shape[:2]
    detections: list[dict[str, Any]] = []
    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return detections

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    masks_obj = getattr(result, "masks", None)
    has_polys = masks_obj is not None and getattr(masks_obj, "xy", None) is not None
    mask_data = (
        masks_obj.data.detach().cpu().numpy()
        if (masks_obj is not None and getattr(masks_obj, "data", None) is not None)
        else []
    )

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cls_id = int(classes[i])
        conf = float(confs[i])

        mask = None
        area_px = max(0, (x2 - x1) * (y2 - y1))
        if masks_obj is not None:
            inst_mask = np.zeros((h, w), dtype=np.uint8)
            if has_polys and i < len(masks_obj.xy) and len(masks_obj.xy[i]) >= 3:
                poly = masks_obj.xy[i].astype(np.int32)
                cv2.fillPoly(inst_mask, [poly], 255)
            elif i < len(mask_data):
                m_unpadded = unpad_letterbox_mask(mask_data[i], (h, w))
                inst_mask[m_unpadded >= 0.5] = 255
            if np.any(inst_mask):
                mask = inst_mask
                area_px = int(np.count_nonzero(inst_mask))

        detections.append(
            {
                "index": i + 1,
                "cls_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "mask": mask,
                "area": area_px,
            }
        )

    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def run_cut_inference(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    min_conf: float,
    cut_class_id: int,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    """Run ONLY the cut model on a single BGR frame."""
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

    names = _names_dict(getattr(result, "names", None) or getattr(model, "names", None))
    eval_info = evaluate_cut_model_prediction(
        result,
        model=model,
        cut_class_id=cut_class_id,
        min_conf=min_conf,
    )
    detections = extract_cut_detections(result, frame.shape, names)

    probs_list: list[tuple[str, float]] = []
    probs = getattr(result, "probs", None) if result is not None else None
    if probs is not None:
        values = probs.data.detach().cpu().flatten().tolist()
        probs_list = sorted(
            [(names.get(i, str(i)), float(v)) for i, v in enumerate(values)],
            key=lambda kv: kv[1],
            reverse=True,
        )

    return {
        "infer_ms": infer_ms,
        "eval": eval_info,
        "detections": detections,
        "probs": probs_list,
        "names": names,
    }


def render_overlay(
    frame: np.ndarray,
    output: dict[str, Any],
    model_name: str,
    conf: float,
    min_conf: float,
    imgsz: int,
    fps: float | None = None,
) -> np.ndarray:
    """Draw cut masks/boxes, classification probabilities (if any), and HUD."""
    vis = frame.copy()
    ev = output["eval"]
    detections = output["detections"]
    is_cut = bool(ev.get("is_cut_detected", False))

    for det in detections:
        is_positive_det = (
            ev.get("class_id") is not None
            and det["cls_id"] == ev.get("class_id")
            and det["conf"] >= min_conf
            and is_cut
        )
        color = (60, 230, 100) if is_positive_det else (0, 175, 255)

        mask = det["mask"]
        if mask is not None:
            mask_bool = mask > 0
            vis[mask_bool] = (
                vis[mask_bool].astype(np.float32) * 0.55
                + np.array(color, dtype=np.float32) * 0.45
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)

        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']} {det['conf']:.2f} ({det['area']}px)"
        label_y = max(24, y1 - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(vis, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4), (20, 20, 20), -1)
        cv2.putText(
            vis,
            label,
            (x1 + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    status_text = "CUT DETECTED" if is_cut else "NOT CUT"
    hud_lines = [
        f"Model: {model_name} | conf={conf:.2f} | min_conf={min_conf:.2f} | imgsz={imgsz} | infer={output['infer_ms']:.1f}ms"
        + (f" | FPS={fps:.1f}" if fps is not None else ""),
        (
            f"Decision: {status_text} | class={ev.get('class_name')} "
            f"| conf={ev.get('confidence', 0.0):.3f} | dets={len(detections)}"
        ),
        f"Reason: {ev.get('reason', '')}",
    ]
    if output["probs"]:
        top_probs = ", ".join(f"{name}:{val:.2f}" for name, val in output["probs"][:4])
        hud_lines.append(f"Probs: {top_probs}")

    bar_h = 14 + len(hud_lines) * 22
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], bar_h), (15, 18, 24), -1)
    vis = cv2.addWeighted(overlay, 0.74, vis, 0.26, 0)

    for i, text in enumerate(hud_lines):
        color = (235, 240, 245)
        if i == 1:
            color = (80, 240, 120) if is_cut else (80, 175, 255)
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
    """Print concise cut model result summary to stdout."""
    ev = output["eval"]
    dets = output["detections"]
    print(
        f"[{label}] infer={output['infer_ms']:.1f}ms | "
        f"is_cut={ev.get('is_cut_detected')} | "
        f"class={ev.get('class_name')} | "
        f"conf={ev.get('confidence', 0.0):.3f} | "
        f"dets={len(dets)} | "
        f"reason={ev.get('reason')}"
    )
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        print(
            f"  Det #{det['index']}: cls={det['cls_id']} ({det['class_name']}) "
            f"conf={det['conf']:.3f} area={det['area']}px bbox=({x1},{y1},{x2},{y2}) "
            f"has_mask={det['mask'] is not None}"
        )
    if output["probs"]:
        probs_str = ", ".join(f"{name}={val:.3f}" for name, val in output["probs"])
        print(f"  Probs: {probs_str}")


def save_snapshot(image: np.ndarray, prefix: str = "cut_model") -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{prefix}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(out_path), image)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ONLY the cut model (models/cut-seg.pt)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to cut model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run cut model on a single image file.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Run cut model on a video file.",
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
        default=0.25,
        help="Raw YOLO prediction confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.70,
        help="Minimum confidence required for positive CUT decision (default: 0.70).",
    )
    parser.add_argument(
        "--cut-class-id",
        type=int,
        default=0,
        help="Fallback positive cut class ID if class names are generic (default: 0).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Model input image size (default: 320).",
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
        "--no-gui",
        action="store_true",
        help="Disable OpenCV preview window (useful for headless testing).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: Cut model not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading cut model ONLY: {model_path}")
    t_load = time.perf_counter()
    model = YOLO(str(model_path))
    print(
        f"Loaded in {(time.perf_counter() - t_load) * 1000.0:.1f}ms | "
        f"task={getattr(model, 'task', 'unknown')} | classes={getattr(model, 'names', {})}"
    )

    conf = float(np.clip(args.conf, 0.01, 0.99))
    min_conf = float(np.clip(args.min_conf, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Mode 1: Single Image
    # ------------------------------------------------------------------
    if args.image is not None:
        img_path = args.image.expanduser().resolve()
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"ERROR: Could not read image: {img_path}", file=sys.stderr)
            return 1

        output = run_cut_inference(
            model=model,
            frame=frame,
            conf=conf,
            min_conf=min_conf,
            cut_class_id=args.cut_class_id,
            imgsz=args.imgsz,
            device=args.device,
        )
        print_frame_summary(output, label=img_path.name)

        vis = render_overlay(frame, output, model_path.name, conf, min_conf, args.imgsz)
        if args.save is not None:
            save_path = args.save.expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
            print(f"Saved annotated image to: {save_path}")

        if not args.no_gui:
            win_name = "Cut Model Test (Image)"
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

    win_name = f"Cut Model Test ({source_label})"
    print(
        f"Running cut model on {source_label} | conf={conf:.2f} | "
        f"min_conf={min_conf:.2f} | imgsz={args.imgsz}"
    )
    if not args.no_gui:
        print(
            "Controls: [Q/ESC] Quit | [S] Save snapshot | "
            "[+/-] min_conf | [[]/[]] raw conf | [SPACE] Pause (video)"
        )

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
                output = run_cut_inference(
                    model=model,
                    frame=frame,
                    conf=conf,
                    min_conf=min_conf,
                    cut_class_id=args.cut_class_id,
                    imgsz=args.imgsz,
                    device=args.device,
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
                    min_conf=min_conf,
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
                    min_conf = min(0.95, round(min_conf + 0.05, 2))
                    print(f"Updated min_conf -> {min_conf:.2f}")
                elif key in (ord("-"), ord("_")):
                    min_conf = max(0.05, round(min_conf - 0.05, 2))
                    print(f"Updated min_conf -> {min_conf:.2f}")
                elif key == ord("]"):
                    conf = min(0.95, round(conf + 0.05, 2))
                    print(f"Updated conf -> {conf:.2f}")
                elif key == ord("["):
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

