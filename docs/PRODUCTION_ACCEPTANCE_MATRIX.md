# Acceptance scope and measured performance

These are software integration results on this mini PC, using isolated synthetic
Loan IDs and recorded input. They do not establish physical SOP accuracy or
acceptable live-camera throughput. Exact logs and counts are in
`PRODUCTION_TEST_RESULTS.md` and its JSON counterpart.

| Requirement | Evidence | Remaining acceptance |
|---|---|---|
| Independent engines and model imports | Original regression suites, protected launch/import checks | Full operator procedure with each real camera interface |
| Persistent queue and multiple Loan IDs | Store/API tests and actual HTTP service recovery replay | Bank operator usability review |
| Two-START purity-to-sealing continuity | Original worker replay in the same host; headless handoff tests preserve populated identities/gallery and uncertainty | Real jewellery movement, rubbing and acid during handoff |
| Missing evidence and standalone sealing | Native replay leaves absent evidence incomplete; standalone sealing regression | Real sealed/unsealed packets |
| Packet Releasing | Original workers, source reports and common PDF parsed in replay | Cut, mouth, OCR and count accuracy with real packets |
| Duplicate START/STOP and confirmed RESET | Controller tests and native reset replay, distinct retained attempts | Physical serial buttons unavailable |
| Camera ownership | Managed-launch locks, real worker acquisition barriers, startup-stop barrier tests | Actual USB disconnect/reconnect and recovery |
| Engine/API crashes and interrupted sessions | Isolated host kill and API restart; raw evidence/result recovery | Power loss and actual mini PC reboot |
| Background evidence and PDF integrity | Native exports, bounded publication jobs, original/source hash checks and parsed PDFs | Long-duration load and retention acceptance |
| Browser, panel and debug viewer closure | Actual HTTP client disconnect, separate panel smoke test, debug viewer closure while recording | Physical monitor hot unplug |
| No desktop dependency in production | Native QCoreApplication workers with display variables removed | Boot-to-service after actual reboot |
| Missing models and exclusive acquisition | Model integrity/import checks and controller failure tests | Device-specific permissions after reboot |

Measured examples from the latest saved replay artifacts:

| Recorded workload | Startup | STOP to camera release | Peak sampled host RSS | Sampled inference rate |
|---|---|---|---|---|
| Linked purity | 3.64 s | 0.62 s | Combined pledge host 1,355 MB | Purity 6.3–8.1 FPS |
| Linked sealing, same host | 0.88 s | 1.62 s | Included above | Not exposed by this worker |
| Packet Releasing, cold | 5.44 s | 1.02 s | 1,520 MB | About 0.86–0.87 FPS after startup |
| Consecutive releasing, cached weights | 5.24 s cold / 0.65 s warm | 1.08 / 1.50 s | 1,558 MB | About 0.86–1.76 FPS after startup |

Source: `outputs/production-linked-normal-replay.json`,
`outputs/production-releasing-normal-replay.json` and
`outputs/production-releasing-cache-replay.json`. Short white-bed fixtures do not
benchmark OCR-heavy packet handling. These are host metrics, not total system
memory. CPU percentages use one core as 100%; sampled peaks were 475% for the
pledge host during initialization and 171% for cold releasing. Thread budgets
bound compatible pools; they are not a system-wide CPU quota. Release throughput
is a material limitation to review before operational rollout; no confidence,
resolution or evidence rules were weakened to increase the figures.

The three workflows may be enabled after the documented software checks pass.
Physical bench acceptance, serial commissioning and bank operational approval
remain separate. The existing release cut/mouth verification limitations remain
documented in `PRODUCTION_KNOWN_LIMITATIONS.md`.
