"""Workflow module."""

from ._imports import *

from .config import *
from packet_cut.cut_model import evaluate_cut_model_prediction
from packet_cut.jewel_tracking import JewelTracker

from .dashboard import create_dashboard, draw_mask, show

from .detection import blur_score, detection_present, evaluate_hand_detection, find_number_model, get_union_mask, matching_class_ids, predict, require_file

from .ocr import front_side_and_number

from .verification import analyze_packet_gap, draw_packet_components, get_components, save_verification_debug, signature, stable_pair

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        default=CAMERA_ID,
        help="Camera device index (default: 0) or path to a video file"
    )

    parser.add_argument(
        "--cpu",
        action="store_true"
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save diagnostic masks and gap verification frames"
    )

    args = parser.parse_args()

    # Automatically fallback to CPU if CUDA is not available
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = (
            int(args.device)
            if str(args.device).isdigit()
            else args.device
        )


    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    packet_path = require_file(
        PACKET_MODEL_PATH,
        "Packet segmentation model"
    )

    hand_path = require_file(
        HAND_MODEL_PATH,
        "Hand segmentation model"
    )

    scissor_path = require_file(
        SCISSOR_MODEL_PATH,
        "Scissor segmentation model"
    )

    topstrip_path = require_file(
        TOPSTRIP_MODEL_PATH,
        "TopStrip segmentation model"
    )

    number_path = find_number_model()

    print("\nLoading models...")

    packet_model = YOLO(str(packet_path))
    hand_model = YOLO(str(hand_path))
    scissor_model = YOLO(str(scissor_path))
    topstrip_model = YOLO(str(topstrip_path))
    number_model = YOLO(str(number_path))

    cut_model_path = require_file(
        CUT_MODEL_PATH,
        "Final cut detection model"
    )

    cut_model = YOLO(
        str(cut_model_path),
        task="segment"
    )
    jewel_model_path = require_file(JEWEL_MODEL_PATH, "Jewel tracking model")
    jewel_model = YOLO(str(jewel_model_path), task="segment")

    print("Packet classes:", packet_model.names)
    print("Hand classes:", hand_model.names)
    print("Scissor classes:", scissor_model.names)
    print("TopStrip classes:", topstrip_model.names)
    print("Packet number classes:", number_model.names)

    print("Inference device:", device)


    # --------------------------------------------------------
    # CLASS IDs
    # --------------------------------------------------------

    hand_ids = matching_class_ids(
        hand_model,
        ("hand",)
    )

    scissor_ids = matching_class_ids(
        scissor_model,
        ("scissor",)
    )

    topstrip_ids = matching_class_ids(
        topstrip_model,
        ("topstrip", "strip", "seal", "top")
    )

    packet_ids = matching_class_ids(
        packet_model,
        ("packet", "bag")
    )


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera_src = int(args.camera) if str(args.camera).isdigit() else args.camera

    if isinstance(camera_src, int):
        cap = cv2.VideoCapture(camera_src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_src)
    else:
        cap = cv2.VideoCapture(camera_src)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera source '{args.camera}'. "
            "Please ensure your USB webcam is plugged in and recognized by Windows, "
            "or provide a valid video file path via --camera <path>."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL
    )


    # --------------------------------------------------------
    # FRAME READER
    # --------------------------------------------------------

    def read_frame():

        ok, image = cap.read()

        if not ok:
            raise RuntimeError(
                "Could not read webcam frame"
            )

        return image


    try:

        # ====================================================
        # STEP 1 + 2: FRONT SIDE AND PACKET NUMBER
        # ====================================================

        packet_number = front_side_and_number(
            cap,
            number_model,
            device
        )

        if packet_number is None:
            return

        print(
            "Operator-confirmed packet number:",
            packet_number
        )


        # ====================================================
        # STEP 3: TURN TO BACK / TOPSTRIP INSPECTION
        # ====================================================

        while True:

            frame = read_frame()

            key = show(
                frame,
                [
                    f"PACKET NUMBER: {packet_number}",
                    "STEP 3 - Turn packet to BACK / seal side.",
                    "Press T when positioned | Q to quit"
                ],
                (0, 220, 255)
            )

            if key in (ord("q"), 27):
                return

            if key == ord("t"):
                break


        # ====================================================
        # TOPSTRIP INSPECTION
        # ====================================================

        seal_deadline = (
            time.time()
            + SEAL_INSPECTION_TIMEOUT
        )

        seal_status = "UNKNOWN_ORIENTATION"

        while time.time() < seal_deadline:

            frame = read_frame()

            packet_result = predict(
                packet_model,
                frame,
                PACKET_CONF,
                device=device
            )

            top_result = predict(
                topstrip_model,
                frame,
                TOPSTRIP_CONF,
                device=device
            )

            packet_mask = get_union_mask(
                packet_result,
                frame.shape[:2],
                packet_ids
            )

            top_mask = get_union_mask(
                top_result,
                frame.shape[:2],
                topstrip_ids
            )

            top_present = detection_present(
                top_result,
                topstrip_ids
            )

            packet_visible = (
                np.count_nonzero(packet_mask) > 500
            )

            sharpness = blur_score(frame)

            preview = draw_mask(
                frame,
                packet_mask,
                (255, 120, 0),
                0.20
            )

            preview = draw_mask(
                preview,
                top_mask,
                (255, 0, 255),
                0.35
            )

            if (
                not packet_visible
                or sharpness < BLUR_VARIANCE_MIN
            ):

                seal_status = "UNKNOWN_ORIENTATION"

                reason = (
                    "Packet not clearly visible or image blurry"
                )

            elif top_present:

                seal_status = "OPEN"

                reason = (
                    "TopStrip visible: OPEN / CUTTING_NOT_READY"
                )

            else:

                seal_status = "CLOSED"

                reason = (
                    "TopStrip not visible in valid inspection: CLOSED"
                )

            key = show(
                preview,
                [
                    f"SEAL STATUS: {seal_status}",
                    f"TopStrip: {'VISIBLE' if top_present else 'NOT DETECTED'}",
                    reason,
                    "If OPEN, reposition/close packet and inspect again.",
                    "Press C only when CLOSED | Q to quit"
                ],
                (
                    (0, 255, 0)
                    if seal_status == "CLOSED"
                    else (0, 180, 255)
                )
            )

            if key in (ord("q"), 27):
                return

            if (
                key == ord("c")
                and seal_status == "CLOSED"
            ):
                break


        if seal_status != "CLOSED":

            print(
                "Seal inspection timed out. Cutting not ready."
            )

            return


        # ====================================================
        # STEP 4: STORE INTACT PACKET REFERENCE
        # ====================================================

        frame = read_frame()

        baseline_result = predict(
            packet_model,
            frame,
            PACKET_CONF,
            device=device
        )

        reference_mask = get_union_mask(
            baseline_result,
            frame.shape[:2],
            packet_ids
        )

        reference_area = float(
            np.count_nonzero(reference_mask)
        )

        baseline_components = get_components(
            reference_mask,
            reference_area
        )

        if (
            reference_area <= 0
            or len(baseline_components) != 1
        ):

            show(
                frame,
                [
                    "BASELINE FAILED",
                    "Expected one visible connected intact packet.",
                    "Reposition packet and restart."
                ],
                (0, 0, 255)
            )

            cv2.waitKey(0)

            return

        reference_mask = (
            baseline_components[0]["mask"] * 255
        )

        reference_area = float(
            np.count_nonzero(reference_mask)
        )

        print(
            f"Stored intact reference: "
            f"area={reference_area:.0f}px, "
            f"packet={packet_number}"
        )


        # ====================================================
        # STEP 5: HAND AND SCISSOR MONITORING
        # ====================================================

        activity_started = False
        clear_start = None
        hand_consecutive_frames = 0
        scissor_consecutive_frames = 0

        print(
            "Ready. Waiting for cutting activity."
        )

        while True:

            frame = read_frame()

            hand_result = predict(
                hand_model,
                frame,
                HAND_CONF,
                device=device
            )

            scissor_result = predict(
                scissor_model,
                frame,
                SCISSOR_CONF,
                device=device
            )

            # Evaluate hand detection with seal false-positive filter & diagnostic logging
            raw_hand_present, hand_mask, hand_diags = evaluate_hand_detection(
                hand_result,
                hand_ids,
                frame.shape[:2],
                reference_mask=reference_mask
            )

            # Temporal confirmation: require 2 consecutive frames
            if raw_hand_present:
                hand_consecutive_frames += 1
            else:
                hand_consecutive_frames = 0

            hand_present = (hand_consecutive_frames >= HAND_CONFIRM_FRAMES)

            # Scissor detection with spatial proximity check to packet
            raw_scissor_present, scissor_mask, scissor_diags = evaluate_scissor_detection(
                scissor_result,
                scissor_ids,
                frame.shape[:2],
                packet_mask=reference_mask,
                min_conf=SCISSOR_CONF,
                max_dist_px=80.0
            )

            if raw_scissor_present:
                scissor_consecutive_frames += 1
            else:
                scissor_consecutive_frames = 0

            scissors_present = (scissor_consecutive_frames >= SCISSOR_CONFIRM_FRAMES)

            preview = draw_mask(
                frame,
                hand_mask,
                (0, 0, 255),
                0.30
            )

            preview = draw_mask(
                preview,
                scissor_mask,
                (0, 255, 255),
                0.30
            )

            if hand_present or scissors_present:

                activity_started = True
                clear_start = None

                status = "CUTTING_ACTIVITY"

                detail = (
                    "Hand/scissors detected. "
                    "Verification paused."
                )

                color = (0, 180, 255)

            elif activity_started:

                if clear_start is None:
                    clear_start = time.time()

                clear_elapsed = (
                    time.time() - clear_start
                )

                status = "WAITING_AFTER_CUT"

                detail = (
                    f"Scene clear: "
                    f"{clear_elapsed:.1f}/"
                    f"{CLEAR_WAIT_SECONDS:.1f}s"
                )

                color = (0, 220, 255)

                if clear_elapsed >= CLEAR_WAIT_SECONDS:
                    break

            else:

                status = "CUTTING_READY"

                detail = (
                    "Packet sealed and reference stored. "
                    "Waiting for cutting."
                )

                color = (0, 255, 0)

            # Build dashboard diagnostic extra info
            diag_info = [
                ("Packet", packet_number),
                ("Hand Confirmed", "YES" if hand_present else "NO"),
                ("Hand Streak", f"{hand_consecutive_frames}/{HAND_CONFIRM_FRAMES}"),
                ("Scissors", "YES" if scissors_present else "NO"),
            ]
            if hand_diags:
                best_hd = hand_diags[0]
                diag_info.append(("Hand Conf", f"{best_hd['conf']:.2f}"))
                diag_info.append(("Hand Area", f"{best_hd['area_px']} px"))
                diag_info.append(("Hand Pos", best_hd['pos_rel']))
                diag_info.append(("Seal Filter", "FILTERED" if best_hd['is_seal_fp'] else "PASS"))

            dashboard = create_dashboard(
                preview,
                "STEP 5 - CUTTING",
                status,
                [
                    detail,
                    "Keep hands/scissors clear after cut",
                    "Q to quit"
                ],
                extra_info=diag_info
            )
            key = show(dashboard, [])

            if key in (ord("q"), 27):
                return


        # ====================================================
        # STEP 6: POST-CUT VERIFICATION FOR 10 SECONDS
        # ====================================================

        verify_start = time.time()

        next_sample = 0.0

        sample_count = 0
        valid_count = 0

        previous_signature = None
        stable_streak = 0

        last_reason = (
            "Waiting for post-cut segmentation"
        )

        last_preview = None
        last_gap = 0.0
        components = []
        verify_hand_streak = 0
        verify_scissor_streak = 0


        while (
            time.time() - verify_start
            < CUT_VERIFY_SECONDS
        ):

            frame = read_frame()

            # Check whether hand/scissors return.
            hand_result = predict(
                hand_model,
                frame,
                HAND_CONF,
                device=device
            )

            scissor_result = predict(
                scissor_model,
                frame,
                SCISSOR_CONF,
                device=device
            )

            raw_hp, h_mask_v, h_diags_v = evaluate_hand_detection(
                hand_result,
                hand_ids,
                frame.shape[:2],
                reference_mask=reference_mask
            )

            if raw_hp:
                verify_hand_streak += 1
            else:
                verify_hand_streak = 0

            hand_present = (verify_hand_streak >= HAND_CONFIRM_FRAMES)

            raw_sp, s_mask_v, s_diags_v = evaluate_scissor_detection(
                scissor_result,
                scissor_ids,
                frame.shape[:2],
                packet_mask=reference_mask,
                min_conf=SCISSOR_CONF,
                max_dist_px=80.0
            )
            if raw_sp:
                verify_scissor_streak += 1
            else:
                verify_scissor_streak = 0
            scissors_present = (verify_scissor_streak >= SCISSOR_CONFIRM_FRAMES)


            # ----------------------------------------------
            # ACTIVITY RETURNED: PAUSE AND RESET CLEAR WAIT
            # ----------------------------------------------

            if hand_present or scissors_present:

                next_sample = 0.0
                sample_count = 0
                valid_count = 0

                stable_streak = 0
                previous_signature = None

                last_reason = (
                    "Hand/scissors returned: restart clear wait"
                )

                clear_start = None

                diag_v = [
                    ("Packet", packet_number),
                    ("Hand Present", "YES" if hand_present else "NO"),
                    ("Scissors", "YES" if scissors_present else "NO"),
                    ("Reason", "Activity detected")
                ]
                dashboard = create_dashboard(
                    frame,
                    "VERIFICATION PAUSED",
                    "PAUSED",
                    [
                        "Hand or scissors returned.",
                        last_reason,
                        "Q to quit"
                    ],
                    extra_info=diag_v
                )
                key = show(dashboard, [])

                if key in (ord("q"), 27):
                    return


                # Wait until both disappear for the full clear period.
                clear_hand_streak = 0
                clear_scissor_streak = 0
                while True:

                    frame = read_frame()

                    hr = predict(
                        hand_model,
                        frame,
                        HAND_CONF,
                        device=device
                    )

                    sr = predict(
                        scissor_model,
                        frame,
                        SCISSOR_CONF,
                        device=device
                    )

                    raw_c_hp, _, _ = evaluate_hand_detection(
                        hr,
                        hand_ids,
                        frame.shape[:2],
                        reference_mask=reference_mask
                    )
                    if raw_c_hp:
                        clear_hand_streak += 1
                    else:
                        clear_hand_streak = 0

                    hp = (clear_hand_streak >= HAND_CONFIRM_FRAMES)

                    raw_c_sp, _, _ = evaluate_scissor_detection(
                        sr,
                        scissor_ids,
                        frame.shape[:2],
                        packet_mask=reference_mask,
                        min_conf=SCISSOR_CONF,
                        max_dist_px=80.0
                    )
                    if raw_c_sp:
                        clear_scissor_streak += 1
                    else:
                        clear_scissor_streak = 0
                    sp = (clear_scissor_streak >= SCISSOR_CONFIRM_FRAMES)

                    if hp or sp:

                        clear_start = None

                        detail = (
                            "Hand/scissors still present"
                        )

                    else:

                        if clear_start is None:
                            clear_start = time.time()

                        elapsed_clear = (
                            time.time() - clear_start
                        )

                        detail = (
                            f"Clear wait: "
                            f"{elapsed_clear:.1f}/"
                            f"{CLEAR_WAIT_SECONDS:.1f}s"
                        )

                        if elapsed_clear >= CLEAR_WAIT_SECONDS:

                            verify_start = time.time()

                            break

                    dash_c = create_dashboard(
                        frame,
                        "WAITING AFTER CUT",
                        "CLEARING",
                        [
                            detail,
                            "Verification restarts after clear scene."
                        ],
                        extra_info=[
                            ("Packet", packet_number),
                            ("Hand", "YES" if hp else "NO"),
                            ("Scissors", "YES" if sp else "NO")
                        ]
                    )
                    key = show(dash_c, [])

                    if key in (ord("q"), 27):
                        return

                continue


            # ----------------------------------------------
            # RUN PACKET SEGMENTATION AT SAMPLE INTERVAL
            # ----------------------------------------------

            elapsed = (
                time.time() - verify_start
            )

            if elapsed >= next_sample:

                packet_result = predict(
                    packet_model,
                    frame,
                    PACKET_CONF,
                    device=device
                )

                p_info = extract_packet_pieces(
                    packet_result,
                    frame.shape[:2],
                    packet_ids=packet_ids,
                    reference_area=reference_area,
                    min_conf=PACKET_CONF,
                    verbose=False
                )
                components = p_info["pieces"]
                packet_mask = p_info["union_mask"]

                sample_count += 1

                total_packet_px = np.count_nonzero(packet_mask)
                last_gap_info = None

                # Rule 17 & 18: Packet missing or insufficient segmentation
                if total_packet_px < 300:
                    last_reason = "PACKET_MISSING: Packet area below minimum detection threshold"
                    stable_streak = 0
                    previous_signature = None

                elif len(components) != 2:
                    if len(components) < 2:
                        last_reason = f"No physical cut: Only {len(components)} connected piece detected"
                    else:
                        last_reason = f"VERIFICATION_UNCERTAIN: Found {len(components)} fragments (expected 2)"
                    stable_streak = 0
                    previous_signature = None

                else:
                    a, b = components[0], components[1]

                    total_ratio = (
                        a["area"] + b["area"]
                    ) / max(reference_area, 1.0)

                    if not (
                        MIN_TOTAL_AREA_RATIO
                        <= total_ratio
                        <= MAX_TOTAL_AREA_RATIO
                    ):
                        last_reason = (
                            "VERIFICATION_UNCERTAIN: Piece area mismatch: "
                            f"{total_ratio:.2f}x baseline"
                        )
                        stable_streak = 0
                        previous_signature = None

                    else:
                        # Full physical gap analysis across facing edges
                        gap_res = analyze_packet_gap(
                            a["mask"],
                            b["mask"],
                            frame=frame,
                            reference_area=reference_area,
                            min_norm_gap=MIN_NORM_GAP_RATIO,
                            min_coverage=MIN_GAP_COVERAGE_RATIO
                        )
                        last_gap_info = gap_res
                        last_gap = gap_res["min_gap"]

                        if not gap_res["valid"]:
                            last_reason = f"GAP_INVALID: {gap_res['reason']}"
                            stable_streak = 0
                            previous_signature = None

                        else:
                            current_sig = signature(components)

                            if (
                                previous_signature is None
                                or stable_pair(
                                    previous_signature,
                                    current_sig
                                )
                            ):
                                stable_streak += 1
                            else:
                                stable_streak = 1

                            previous_signature = current_sig
                            valid_count += 1
                            last_reason = (
                                f"Two detached masks with valid gap ({gap_res['min_gap']:.1f}px, "
                                f"{gap_res['coverage_pct']:.0f}% interface) [Streak: {stable_streak}/5]"
                            )

                            # Streak tracks consecutive stable frames; must complete full 10-second verification window

                # Save debug images if debug flag is enabled
                if getattr(args, "debug", False):
                    save_status = "valid" if (last_gap_info and last_gap_info.get("valid")) else "invalid"
                    save_verification_debug(
                        OUTPUT_DIR,
                        frame,
                        packet_mask,
                        components,
                        last_gap_info,
                        f"samp{sample_count}_{save_status}"
                    )

                last_preview = draw_packet_components(
                    frame,
                    components,
                    gap_info=last_gap_info
                )

                next_sample += VERIFY_SAMPLE_INTERVAL


            # ----------------------------------------------
            # LIVE VERIFICATION DISPLAY (SINGLE DASHBOARD)
            # ----------------------------------------------

            valid_ratio = (
                valid_count / max(sample_count, 1)
            )

            remaining = max(
                0.0,
                CUT_VERIFY_SECONDS
                - (time.time() - verify_start)
            )

            # Telemetry for the rich live dashboard
            min_gap_str = f"{last_gap_info['min_gap']:.1f} px" if last_gap_info else "0.0 px"
            norm_gap_str = f"{last_gap_info['norm_gap']:.3f}" if last_gap_info else "0.000"
            cov_str = f"{last_gap_info['coverage_pct']:.0f}%" if last_gap_info else "0%"
            bg_val_str = "VALID" if (last_gap_info and last_gap_info.get("valid")) else "PENDING"

            verify_telemetry = [
                ("Pieces Detected", f"{len(components)}"),
                ("Gap Width", min_gap_str),
                ("Normalized Gap", norm_gap_str),
                ("Gap Coverage", cov_str),
                ("Background", bg_val_str),
                ("Stability Timer", f"{elapsed:.1f} / {CUT_VERIFY_SECONDS:.1f}s"),
                ("Valid Samples", f"{valid_count}/{sample_count} ({valid_ratio:.0%})")
            ]

            verify_dashboard = create_dashboard(
                last_preview if last_preview is not None else frame,
                "POST-CUT VERIFICATION",
                "VERIFYING GAP" if valid_count > 0 else "EVALUATING",
                [
                    last_reason,
                    f"Remaining: {remaining:.1f}s | Stable: {stable_streak}",
                    "Q to quit"
                ],
                extra_info=verify_telemetry
            )
            key = show(verify_dashboard, [])

            if key in (ord("q"), 27):
                return

            time.sleep(0.01)


        # ====================================================
        # STEP 7: FINAL DECISION
        # ====================================================

        # Run final(cut_detection).pt confirmation inference
        cut_result = predict(cut_model, frame, 0.25, device=device)
        cut_eval = evaluate_cut_model_prediction(cut_result, model=cut_model, cut_class_id=CUT_MODEL_CLASS_ID, min_conf=0.25)
        print(f"[CUT MODEL] source={cut_model_path.resolve()} | class={cut_eval['class_name']} | confidence={cut_eval['confidence']:.1%} | cut={cut_eval['is_cut_detected']}")
        is_cut_model_confirmed = cut_eval["is_cut_detected"]

        valid_ratio = (
            valid_count / max(sample_count, 1)
        )

        total_packet_px = np.count_nonzero(packet_mask) if 'packet_mask' in locals() else 0

        # Exact evaluation across standardized 5 states (Step 7 requirement)
        final_state, final_reason = evaluate_final_cut_decision(
            components=components,
            cut_eval=cut_eval,
            last_gap_info=last_gap_info,
            sample_count=sample_count,
            valid_count=valid_count,
            stable_streak=stable_streak,
            min_verify_samples=MIN_VERIFY_SAMPLES,
            min_valid_sample_ratio=MIN_VALID_SAMPLE_RATIO,
            min_streak=5,
            last_p_info=locals().get("p_info", None)
        )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        # Save final debug package if debug mode is active
        if getattr(args, "debug", False):
            save_verification_debug(
                OUTPUT_DIR,
                frame,
                packet_mask if 'packet_mask' in locals() else None,
                components,
                last_gap_info if 'last_gap_info' in locals() else None,
                f"final_{final_state}"
            )

        while True:

            frame = read_frame()

            jewel_count = 0
            if final_state == "CUT_CONFIRMED":
                if "jewel_tracker" not in locals():
                    jewel_tracker = JewelTracker(jewel_model, device, JEWEL_CONF)
                    print(f"[JEWEL TRACKING] Model: {jewel_model_path}")
                jewel_view, visible_jewels, jewel_count = jewel_tracker.update(frame)
                cv2.putText(
                    jewel_view,
                    f"Jewels tracked: {jewel_count} | Visible: {visible_jewels} | Q quit / R next packet",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )
                last_preview = jewel_view

            final_telemetry = [
                ("Final Decision", final_state),
                ("Packet Number", packet_number),
                ("Verified Pieces", f"{len(components)}"),
                ("Valid Samples", f"{valid_count}/{sample_count} ({valid_ratio:.1%})"),
                ("Stable Streak", str(stable_streak)),
                ("Cut Model", f"{cut_eval['class_name']} ({cut_eval['confidence']:.1%})"),
                ("Model File", str(cut_model_path)),
                *( [("Jewel tracks counted", str(jewel_count)), ("Visible jewel tracks", str(visible_jewels))]
                   if final_state == "CUT_CONFIRMED" else [] ),
            ]

            final_dashboard = create_dashboard(
                last_preview if last_preview is not None else frame,
                "FINAL DECISION",
                final_state,
                [
                    final_reason,
                    "R = Restart next packet",
                    "Q = Quit verification"
                ],
                extra_info=final_telemetry
            )
            key = show(final_dashboard, [])

            if key in (ord("q"), 27):
                break

            if key == ord("r"):

                print(
                    "Restart application for next packet."
                )

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print("Camera released.")
