import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
from ultralytics import YOLO
import packet_cut_core

p = Path(r"D:\Project\Embsys\Packet-cut-final\packet_cut_outputs\verification_debug")
files = sorted([f for f in p.glob("*179457540*orig.jpg") if "final" not in f.name])
model_dir = Path(r"D:\Project\Embsys\Packet-cut-new\models")
pkt_model = YOLO(str(model_dir / "packet_seg_yolov8s_320.onnx"), task="segment")
cut_model = YOLO(str(model_dir / "final(cut_detection).pt"))

print("=" * 95)
print(f"{'SMP':<4} | {'YOLO':<4} | {'CC':<3} | {'ACC':<3} | {'REJ':<3} | {'GAP_PX':<8} | {'COV_PCT':<7} | {'OVERLAP':<7} | {'RESULT':<7} | {'REASON'}")
print("=" * 95)

valid_count = 0
stable_streak = 0
prev_sig = None
components = []
last_gap = None
last_p_info = None

for idx, f in enumerate(files):
    frame = cv2.imread(str(f))
    res = pkt_model.predict(frame, imgsz=320, conf=0.12, verbose=False)[0]
    p_info = packet_cut_core.extract_packet_pieces(res, frame.shape[:2], reference_area=55000, min_conf=0.12, verbose=False)
    last_p_info = p_info
    components = p_info["pieces"]
    rejected = p_info["rejected_pieces"]
    
    yolo_cnt = p_info["yolo_instance_count"]
    raw_cc = p_info["raw_cc_count"]
    acc_cnt = len(components)
    rej_cnt = len(rejected)
    
    min_gap = 0.0
    cov_pct = 0.0
    overlap_px = 0
    is_valid = False
    fail_reason = "Pending"
    
    if acc_cnt == 2:
        a, b = components[:2]
        gap_info = packet_cut_core.analyze_packet_gap(a["mask"], b["mask"], frame=frame, reference_area=55000)
        last_gap = gap_info
        min_gap = gap_info["min_gap"]
        cov_pct = gap_info["coverage_pct"]
        overlap_px = gap_info.get("overlap_px", 0)
        if gap_info["valid"]:
            is_valid = True
            fail_reason = "Passed"
            valid_count += 1
            curr_sig = packet_cut_core.piece_signature(components)
            if prev_sig is None or packet_cut_core.is_stable_pair(prev_sig, curr_sig):
                stable_streak += 1
            else:
                stable_streak = 1
            prev_sig = curr_sig
        else:
            fail_reason = gap_info["reason"]
            stable_streak = 0
            prev_sig = None
    elif acc_cnt == 1:
        fail_reason = "Only 1 piece accepted"
        stable_streak = 0
        prev_sig = None
    else:
        fail_reason = f"Pieces count {acc_cnt} != 2"
        stable_streak = 0
        prev_sig = None
        
    res_str = "VALID" if is_valid else "INVALID"
    print(f"#{idx+1:02d} | {yolo_cnt:<4} | {raw_cc:<3} | {acc_cnt:<3} | {rej_cnt:<3} | {min_gap:<8.1f} | {cov_pct:<7.1f} | {overlap_px:<7} | {res_str:<7} | {fail_reason}")

print("=" * 95)
frame9 = cv2.imread(str(files[7]))
cut_res = cut_model.predict(frame9, conf=0.25, verbose=False)[0]
cut_eval = packet_cut_core.evaluate_cut_model_prediction(cut_res, model=cut_model, cut_class_id=0, min_conf=0.25)
state, reason = packet_cut_core.evaluate_final_cut_decision(
    components, cut_eval, last_gap_info=last_gap,
    sample_count=len(files), valid_count=valid_count, stable_streak=stable_streak,
    min_verify_samples=8, min_valid_sample_ratio=0.70, min_streak=5, last_p_info=last_p_info
)
print(f"Cut Model: {cut_eval['class_name']} ({cut_eval['confidence']:.1%})")
print(f"Valid Samples: {valid_count}/{len(files)} ({valid_count/len(files):.1%}) | Stability Streak: {stable_streak}")
print(f"FINAL DECISION: {state}")
print(f"DETAIL: {reason}")
print("=" * 95)
