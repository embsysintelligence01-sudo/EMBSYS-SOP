# Engine integration report

| Engine | Processing reused | Production lifecycle | Independent UI |
|---|---|---|---|
| Purity / Jeweltracker | `engines/purity/tracker.py` Worker and original ContinuousMonitor | Qt5 core host; original camera/stone/identity threads; original recordings and analysis; attempt-scoped capture root | `./development/run_engine.sh purity` |
| Sealing / Packetsealing_001 | `engines/sealing/workflow.py` PacketWorker, TrackingBridge, PackingFlow, original classifiers/OCR/export | Same pledge host for linked purity; standalone host for separate sealing; explicit output root | `./development/run_engine.sh packet` |
| Releasing / PacketReleasing | Original `automated_app/packet_cut/desktop/worker.py`, pipeline, observers, models and report generators | Qt6 core host; actual acquisition-release signal; new pipeline each session; optional weights cache | `./development/run_engine.sh releasing --camera 0` |

The entire original working source is preserved on branch
`backup/pre-production-20261128`; a source archive also exists under ignored
`runtime/backups/pre-production-source.tar.gz`. The current work is on
`integration/production-controller-20261128`. The original source installations
were not deleted or modified. Baseline results, including original failures,
are recorded in `PRODUCTION_BASELINE.json` and private logs.

Detection algorithms, class mappings, preprocessing, confidence thresholds,
identity matching and evidence verification are unchanged in this integration.
The three worker edits are lifecycle/storage/preview switches. Independent UI
behaviour retains its previous defaults. See `configs/production_migration.json`
for exact before/after hashes and reasons. Updated regression assertions repair
outdated test assumptions about existing independent baseline capture and
cropped detector coordinates; they still reject false rubbing/contact evidence.
The installed Ultralytics reference test now receives its required directory
suffix through a temporary link, without changing model bytes or decoding.

The first purity STOP calls the original worker stop and waits for
`camera_released`. `SurveillanceHandoff` reuses PacketStage's `testing_snapshot`,
freezes identity statuses and retains original worker/gallery/feature objects.
The original worker can continue its background finalization while sealing
starts. Final committed testing data is published under a lock and event, as in
the existing PacketStage. No replacement jewel batch is created for linked
sealing, including when no identity can be assigned.

A JSON audit snapshot retains historical tracks/gallery/features/positions and
certainty. After process loss, it is not treated as a live matching model.
Recovered sealing retains historical IDs as uncertain in the report and uses the
existing unassigned recording behaviour; it cannot manufacture a live identity
claim. A confirmed RESET acknowledges the interrupted attempt before continuation.

Production rendering suppresses unnecessary live preview work while retaining
all inference and evidence annotations. Production workers use QCoreApplication,
not hidden engine windows. Debug mode starts a separate read-only instance of
the original widgets; diagnostics contain the same accepted observations.
The virtual panel and independent debugging commands do not write production
queue records. Their camera ownership uses the common lease.

Frontend availability is controlled by `configs/production.toml`. Enablement is
performed only after the corresponding integration/replay checks pass. Physical
bench acceptance is separate from software integration and remains explicitly
listed in the limitations report.
