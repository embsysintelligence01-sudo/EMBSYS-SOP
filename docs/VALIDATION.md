# Validation report

Camera-free automated validation. Test logs are private under ignored outputs/validation/.

| Check | Result | Detail |
|---|---|---|
| cleanup-integrity | PASSED | exit=0; 6.9s; private log: outputs/validation/cleanup-integrity.log |
| storage-paths | PASSED | exit=0; 5.5s; private log: outputs/validation/storage-paths.log |
| source-security-audit | PASSED | exit=0; 13.5s; private log: outputs/validation/source-security-audit.log |
| launcher-ui | PASSED | exit=0; 0.3s; private log: outputs/validation/launcher-ui.log |
| purity-regression | FAILED | exit=1; 153.4s; private log: outputs/validation/purity-regression.log; Ran 414 tests in 150.628s; FAILED (failures=1) |
| slic-regression | FAILED | exit=1; 155.8s; private log: outputs/validation/slic-regression.log; Ran 414 tests in 153.368s; FAILED (failures=1) |
| packet-regression | PASSED | exit=0; 11.1s; private log: outputs/validation/packet-regression.log; Ran 352 tests in 8.551s; OK |
| rub-regression | PASSED | exit=0; 2.4s; private log: outputs/validation/rub-regression.log; Ran 43 tests in 2.038s; OK |
| mouth-lab-regression | FAILED | exit=1; 4.2s; private log: outputs/validation/mouth-lab-regression.log; Ran 12 tests in 2.130s; FAILED (failures=1) |
| historical-regression | FAILED | exit=1; 24.7s; private log: outputs/validation/historical-regression.log; Ran 50 tests in 22.817s; FAILED (failures=2, errors=1) |
| pickup-self-test | PASSED | exit=0; 0.4s; private log: outputs/validation/pickup-self-test.log |
| pickup-motion-self-test | PASSED | exit=0; 0.4s; private log: outputs/validation/pickup-motion-self-test.log |
| archived-pickup-self-test | PASSED | exit=0; 0.4s; private log: outputs/validation/archived-pickup-self-test.log |
| model-copy-integrity | FAILED | 47 original model/asset roles checked at retained, relocated or identical canonical paths; missing/changed: Jeweltracker/nivagar_gold_v3.onnx, Jeweltracker/rubbing_gold.onnx, Jeweltracker/nivagar_gold_v3.pt, Jeweltracker/newstone1.onnx, Jeweltracker/rojeweltrack.onnx, Jeweltracker/acid.onnx, Jeweltracker/INT8_openvino_model/best.xml, Jeweltracker/INT8_openvino_model/best.bin, Jeweltracker/new_rub_detection/models/newstone1.onnx, Jeweltracker/new_rub_detection/models/rubbedmask.onnx, Jeweltracker/int_8_GOLD_JEWEL/gold_int8.xml, Jeweltracker/int_8_GOLD_JEWEL/gold_int8.bin, Jeweltracker/weights/resnet18-f37072fd.pth, Jeweltracker/weights/siglip2-base-patch32-256_vision_encoder.sim.onnx, Jeweltracker/weights/resnet18_features.onnx, Jeweltracker/weights/edgesam/edge_sam_3x_decoder.onnx, Jeweltracker/weights/edgesam/edge_sam_3x_encoder.onnx, Jeweltracker/INT8_openvino_model_hand_seg/best.xml, Jeweltracker/INT8_openvino_model_hand_seg/best.bin, Jeweltracker/INT_8_RUBBEDMASK/rubbedmask_det_v3_int8.xml, Jeweltracker/INT_8_RUBBEDMASK/rubbedmask_det_v3_int8.bin, Jeweltracker/classification/jewelry_correction_gallery.npz, Jeweltracker/classification/operator_corrections.npz, Jeweltracker/classification/foreign_object_text_embeddings_cache.npz, Jeweltracker/classification/jewelry_text_embeddings_cache.npz, Jeweltracker/acidphintegration/acidph_openvino/acidph.bin, Jeweltracker/acidphintegration/acidph_openvino/acidph.xml, Jeweltracker/acid_test/model_training/artifacts/temporal_shared_mobilenet_v3_small_20261021_144911.onnx, Jeweltracker/acid_test/model_training/artifacts/temporal_shared_mobilenet_v3_small_20261021_144911.pt, Jeweltracker/acid_test/model_training/artifacts/temporal_shared_mobilenet_v3_small_20261102_160901.onnx, Jeweltracker/acid_test/model_training/artifacts/temporal_mobilenet_v3_20260912_170149.onnx, Jeweltracker/acid_test/model_training/artifacts/temporal_mobilenet_v3_20260912_170149.pt, Jeweltracker/acid_test/model_training/artifacts/temporal_shared_mobilenet_v3_small_20261102_160901.pt, Jeweltracker/acid_test/models/acid_detector.onnx, Packetsealing_001/packet_mouth_lab/models/mouth_classifier.joblib, Packetsealing_001/packet_mouth_lab/models/packet_bbox_classifier.joblib, Packetsealing_001/triger_motion_rohan/hand_landmarker.task, Packetsealing_001/triger_motion_rohan/newstone1.onnx, Packetsealing_001/triger_motion_rohan/int_8_GOLD_JEWEL/gold_int8.xml, Packetsealing_001/triger_motion_rohan/int_8_GOLD_JEWEL/gold_int8.bin, Packetsealing_001/model onnx/hand_landmarker.task, Packetsealing_001/model onnx/pcnumber.onnx, Packetsealing_001/model onnx/handseg model.onnx, Packetsealing_001/model onnx/bagnewmodel.onnx, Packetsealing_001/model onnx/stripmodel.onnx, Packetsealing_001/model onnx/fp_filter_resnet18_bag.onnx, Packetsealing_001/model onnx/hsv_fp_filter_strip.pt |
| required-model-paths | FAILED | 17 required assets; 0 temporal models; missing: engines/Jeweltracker/int_8_GOLD_JEWEL/gold_int8.xml, engines/Jeweltracker/INT8_openvino_model/best.xml, engines/Jeweltracker/INT_8_RUBBEDMASK/rubbedmask_det_v3_int8.xml, engines/Jeweltracker/acid_test/models/acid_detector.onnx, engines/Jeweltracker/acidphintegration/acidph_openvino/acidph.xml, engines/Jeweltracker/weights/resnet18_features.onnx, engines/Jeweltracker/weights/siglip2-base-patch32-256_vision_encoder.sim.onnx, engines/Packetsealing_001/model onnx/bagnewmodel.onnx, engines/Packetsealing_001/model onnx/fp_filter_resnet18_bag.onnx, engines/Packetsealing_001/model onnx/stripmodel.onnx, engines/Packetsealing_001/model onnx/hsv_fp_filter_strip.pt, engines/Packetsealing_001/model onnx/handseg model.onnx, engines/Packetsealing_001/model onnx/pcnumber.onnx, engines/Jeweltracker/int_8_GOLD_JEWEL/gold_int8.bin, engines/Jeweltracker/INT8_openvino_model/best.bin, engines/Jeweltracker/INT_8_RUBBEDMASK/rubbedmask_det_v3_int8.bin, engines/Jeweltracker/acidphintegration/acidph_openvino/acidph.bin |
| launcher-tests | FAILED | exit=1; 8.7s; private log: outputs/validation/launcher-tests.log; Ran 10 tests in 8.343s; FAILED (failures=1) |
| purity-imports | PASSED | exit=0; 2.2s; private log: outputs/validation/purity-imports.log |
| slic-imports | PASSED | exit=0; 2.1s; private log: outputs/validation/slic-imports.log |
| purity-ui | PASSED | exit=0; 2.1s; private log: outputs/validation/purity-ui.log |
| packet-ui | PASSED | exit=0; 2.2s; private log: outputs/validation/packet-ui.log |
| purity-models | PASSED | exit=0; 9.0s; private log: outputs/validation/purity-models.log |
| packet-models | PASSED | exit=0; 8.9s; private log: outputs/validation/packet-models.log |
| offline-ocr | PASSED | exit=0; 10.9s; private log: outputs/validation/offline-ocr.log |
| handoff-regression | PASSED | exit=0; 6.5s; private log: outputs/validation/handoff-regression.log; Ran 17 tests in 0.708s; OK |
| surveillance-export-regression | PASSED | exit=0; 5.8s; private log: outputs/validation/surveillance-export-regression.log; Ran 4 tests in 0.856s; OK |
| live-camera-and-operator-SOP | UNTESTED | Requires bench camera and operator: live ROI, full purity/sealing sequence, evidence export and physical camera release. No camera connection attempted. |
| optional-reviewer-and-legacy-hardware | UNTESTED | External recording reviewer, Qt6 standalone acid UI and legacy Hailo/pickup hardware not exercised. |

## Provisional recording trigger correction

These checks cover the recording trigger correction. Earlier integration/refactor results above retain their original scope.

| Check | Result | Detail |
|---|---|---|
| provisional-trigger-regression | PASSED | 107 tests passed, including UI, source-frame sampling, covered homes, missed landmarks, model failures, camera handoff and independent evidence outcomes. Private log outputs/validation/provisional-regression.log. |
| provisional-packet-regression | PASSED | 352 packet tests passed. Private log outputs/validation/provisional-packet.log. |
| original-standalone-trigger | PASSED | Original standalone self-test passed. Private log outputs/validation/provisional-standalone.log. |
| standalone-synthetic-start-parity | PASSED | Identical synthetic frames produced matching starts for covered homes and brief landmark misses. Both stayed idle for occupied homes and missing hands. |
| shared-replay-first-start | PASSED | First pinch start matched exactly across 112 shared real source frames, gold observations and landmarks with the existing SLIC model available to the reference. Later candidate ownership and clip boundaries differed. Private log outputs/validation/provisional-shared-frames.log. |
| provisional-offline-capture | PASSED | Full-worker replay saved a provisional UNASSIGNED cycle and a named J1 cycle from pinch triggers, reported no processing errors, and signaled capture release. Private log outputs/validation/provisional-replay.log. |
| provisional-live-batch-accuracy | UNTESTED | Live camera/operator trial remains pending. Later replay attribution differs from the standalone; complete batch accuracy and per-jewel boundary parity are unverified. Uncertain identity stays UNASSIGNED. |
