# Non-blocking SOP surveillance handoff

This application records evidence. It does not approve a loan, authorize physical handling, release a packet or control a production machine.

Use **Packet Sealing + OCR in the purity window** to continue the same batch. The root launcher's packet button starts a separate session and remains exclusive while an engine owns the camera.

| Situation | Handoff / evidence behavior |
|---|---|
| Missing rubbing or rubbed mask | Continue; purity remains INCOMPLETE or REVIEW_REQUIRED. |
| Missed acid application/detection | Continue; no acid result is invented. |
| Invalid material classification | Continue; retain the invalid prediction and REVIEW_REQUIRED status. |
| Pending recorded-video recovery/PDF | Continue after the camera barrier; retain analysis models until their users finish. |
| Reliable nonsequential IDs, e.g. J1/J3/J8 | Retain those IDs, gallery and session linkage; never number a new batch during handoff. |
| Stale/uncertain identity | Keep the historical label with uncertainty; exclude it from named matching and pickup/placement claims. |
| No assigned IDs / missing identity model | Record packet surveillance from the first frame with UNASSIGNED status and preserve raw recording. |
| Purity gold-model failure | Retry an available packet gold backend or retain unassigned surveillance if unavailable; no absence/identity claim is fabricated. |
| Purity camera/thread still alive | Wait for actual acquisition release. A release flag cannot override a live thread. |
| Purity reset already draining a recording | Wait for lifecycle ownership to finish. No force termination. |

`packet_stage.py` remains the coordinator. It stops live purity acquisition, retains pending analysis and creates the packet window after acquisition/stone/identity threads release ownership. Manual live acid work is cancelled asynchronously; queued Qt result/finished signals are consumed before cache teardown. A running manual rub future can finish without blocking packet acquisition; its arrays and metadata are preserved as unconfirmed auxiliary evidence.

The tracking bridge copies mutable track data under the old worker's lock. Background purity analysis cannot mutate packet geometry. Identity certainty is recorded at the camera-release barrier so model-loading delay does not erase an assignment. Uncertain IDs never enter named assignment/pickup matching. A handoff with zero IDs remains in a fixed batch state and does not automatically assign replacement IDs.

Packet files use their own directory under `engines/Packetsealing_001/outputs/`. They are not placed below a purity folder that a background retention worker may archive. Unknown/uncertain recordings are retained in `.working/` in addition to available exports. Recording or camera teardown cannot be skipped to switch workflows.

The packet recorder and acquisition are closed before final combined export waits for pending purity jobs. Final purity data is published under `testing_lock` and copied under the same lock by the packet worker. Export receives its own copy and does not mutate the shared handoff data. The JSON/PDF contains independent `evidence_statuses.purity_testing` and `evidence_statuses.packet_sealing`, identity uncertainty, predictions and individual evidence fields. Sealing completion never upgrades purity verification. VERIFIED describes evidence only; strip-removal sealing remains an inference rather than physical closure approval.

Both launcher workflows reuse the already-installed SLIC environment. The original normal environment remains available as a purity-only override; its missing OpenCV features cannot support the in-process packet handoff. No library was upgraded.

Regression coverage is in:

- `engines/Jeweltracker/tests/test_packet_stage.py`
- `engines/Jeweltracker/tests/test_surveillance_handoff.py`
- `engines/Packetsealing_001/tests/test_session.py`
- `engines/Packetsealing_001/tests/test_surveillance_handoff.py`
- `engines/Packetsealing_001/tests/test_testing_handoff.py`

The final validation report distinguishes automated checks from untested physical camera/operator work. Existing baseline classification-crop and optional mouth-lab assertion failures remain documented; their algorithms and assertions were not changed to make this integration pass.

Run the focused cases with `python3 scripts/validate.py --checks handoff-regression,surveillance-export-regression`. The validator merges those results with the prior report; its overall exit code remains nonzero while documented baseline failures remain in that report. Full validation already includes these cases through each engine's normal test discovery.
