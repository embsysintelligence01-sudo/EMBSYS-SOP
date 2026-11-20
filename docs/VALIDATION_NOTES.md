# Validation interpretation


## Surveillance correction (0.2.0)

The focused handoff suites pass: 17 purity/handoff tests (including 10 new cases) and four new packet surveillance/export/concurrency tests. Full validation passes all 352 packet tests, 43 rubbing tests and 10 launcher/integration tests. Both purity runs execute 414 tests with only the same previously reproduced classification-crop assertion failure. The optional mouth lab still has its previously reproduced filter-boundary assertion failure. There are no new test failures from the surveillance changes.

Both managed workflows now use the existing SLIC environment by default. The normal environment's missing HOG/SLIC features remain a limitation only when explicitly selecting that old purity-only environment. No packages or source repositories were modified. Setup, desktop/headless launcher startup, both engine UI smoke paths, imports, model loading and offline OCR pass.

The correction separates physical workflow progress from evidence outcomes, supports zero-ID raw recording, preserves uncertainty without named matching, and waits only for safe acquisition/lifecycle ownership. See [handoff behavior and safeguards](SURVEILLANCE_HANDOFF.md) and [protected algorithm audit](SURVEILLANCE_LOGIC_AUDIT.json). Actual camera/operator surveillance and physical camera-release behavior remain untested.

Validation ran on 2026-11-20 using the original local environments. See `VALIDATION.md` for each check and its exact test counts. `AUDIT.json` records source integrity, syntax, credential/model exclusions and unexpected function changes. Private raw logs are under `outputs/validation/`.

The launcher and both existing engine windows were constructed and closed headlessly. Managed-process integration tests started the actual camera-free engine windows in sequence and verified ownership blocking and control restoration. Unit tests also cover failed process startup, nonzero exit, launcher close refusal and inherited lock ownership. These are software lifecycle checks; they cannot establish physical camera release or evidence integrity after a real operator session.

All 45 copied model/embedding assets were checked against their source hashes. Required model paths and XML/BIN pairs exist. CPU checks load the stone, rubbed-mask, acid detection, temporal acid, pH, SigLIP, gold tracking, packet segmentation/filter and ResNet feature models. Packet inference exercises SLIC and checks the existing local FFmpeg executable. Isolated PaddleOCR reads the synthetic test packet number using existing cached models. These checks establish loading/inference compatibility, not real-world detection accuracy.

Two assertion failures were reproduced using an isolated temporary copy of the untouched original Python sources, with the existing SLIC interpreter. The original project directories were not used as a writable test workspace:

- `test_classification_crop.VerifiedCropTests.test_full_frame_coordinates_are_not_offset_by_stone_origin`: the test expects the identical full input frame object; current source passes a cropped array. Neither the production crop logic nor the assertion was changed.
- `test_packet_collection.PacketCollectionTests.test_packet_only_detector_filters_and_clips`: the optional mouth-lab test expects one detection; current source retains two at the boundary. The filter threshold/comparison and assertion were preserved.

The normal purity environment provides OpenCV 5.0.0 without `HOGDescriptor`/`ximgproc`; the SLIC environment provides the extra features. Tests/features requiring them cannot all pass in the normal environment. The main purity import and camera-free UI still pass; packet sealing uses SLIC as before. No package was upgraded or replaced to hide this difference.

Historical image-based regression tests read original files through `EMBSYS_REGRESSION_ASSETS`; generated evidence is not copied into the new production tree. Only fixture lookup paths were adapted in three test files. The validator clears the runtime reviewer-directory override in regression subprocesses because those tests intentionally supply temporary filesystem layouts. Assertions and SOP algorithms are unchanged.

Untested bench work: connecting the configured camera, checking physical ROI/alignment, completing real purity and packet sealing sessions, confirming live recordings/PDF/OCR exports, and checking camera release before switching. This session exposes no `/dev/video*` device, and no live camera connection was attempted. The optional external recordings reviewer, standalone Qt6 acid UI, legacy Hailo tools and training utilities were not exercised end to end.

The combined application is implemented, but validation is not wholly green and production approval cannot be claimed. The new repository has no remote and nothing has been committed or pushed. Resolve/review the documented baseline failures and complete bench validation before publishing or deploying.


## Verified cleanup validation

The post-cleanup run includes every discovered regression suite: launcher (10 passed), purity and SLIC (414 each, same existing classification-crop failure), packet (352 passed), rubbing (43 passed), mouth lab (12, same existing filter-boundary failure), and the preserved historical snapshot (50, the same two failures and one error recorded before relocation). All three deterministic pickup self-tests pass. Failure identities match the before-cleanup/untouched-source baselines in CLEANUP_REGRESSION_COMPARISON.json; no new regression failure was introduced by cleanup.

The full run was followed by 17 handoff tests and four surveillance/export tests, cleanup/content checks, and checks against the actual prepared storage paths. The existing archive containment policy rejected a symlinked capture parent. The capture directory was restored byte-for-byte to its existing combined-project path; no archive safety rule or SOP algorithm was relaxed. Actual purity path containment now passes, and a controlled zero-ID packet recording/decode/JSON/PDF export passes through the prepared packet output location.

All original model roles are checked at retained, relocated or byte-identical canonical paths. Required production model paths remain unchanged. Camera/ROI configuration and originals remain unchanged. Generated caches are removed after validation; private test diagnostics remain under ignored outputs/validation/. See CLEANUP_REPORT.md for every retained, relocated and removed file and the storage restoration decision. Live camera/operator testing remains unperformed.
