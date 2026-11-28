# Production architecture

The production entry is `main.py`, not the preserved two-button `launcher.py`.
There is one FastAPI process, one controller, one SQLite database and at most one
live camera owner. Production uses QCoreApplication engine hosts; no QWidget or
GUI application is constructed in an engine host. Debug viewers and the virtual
hardware panel run separately and cannot own acquisition or change engine state.

```text
Browser: Loan ID / ordered queue       PyQt5 virtual hardware panel
              |                              |
              +---------- HTTP API ----------+
                              |
                      one SOP Controller
                      SQLite WAL / audit
                              |
                    exclusive camera lease
                       /               \
           pledge host (Qt5 core)    release host (Qt6 core)
           existing purity Worker    existing PacketWorker
                    |                       |
           existing TrackingBridge         |
           + sealing PacketWorker          |
                    +-----------+-----------+
                                |
                     original evidence engines
                                |
                  bounded durable report publisher
```

`core/store.py` serializes queue changes with SQLite `BEGIN IMMEDIATE`, foreign
keys and a single-live-attempt unique index. `core/controller.py` serializes
commands, manages hosts and controls admission. Application states are IDLE,
READY, STARTING, RUNNING, STOPPING, FINALIZING, INTERRUPTED and ERROR. Activity
execution status and evidence verification status are separate fields.

Purity + sealing entries are consecutive under one pledge session and cannot be
reordered. STOP never starts another activity. The first STOP releases the
original purity camera barrier and freezes identities in the same process;
original recorded-video/rub/acid/PDF work drains independently. The second START
constructs the existing sealing worker with that exact handoff context. Gold
inference uses the existing shared dispatcher. The second STOP waits for the
original combined export, including pending purity results. Individual evidence
checks are never promoted merely because an activity ends.

Reset requires confirmation. Acquisition is stopped and its previous attempt is
retained. Another attempt gets a new ID and directory. Late events from a retired
host cannot overwrite the current attempt. A process crash interrupts the audit
record; it does not approve a release, silently restart recording, or establish
an identity match.

Release sessions link to the latest actual completed pledge in the local queue
ledger for the same Loan ID. If none exists, MISSING_PLEDGE_RECORD is explicit.
Older file-based reports are not automatically invented as database records.
Release creates a new pipeline/tracker for each activity; only compatible model
weights may be retained between release sessions.

`services/engine_host.py` drives actual legacy workers from QCoreApplication.
It does not implement replacement detectors. The existing acquisition barriers
stay in place. A Linux flock matches the protected independent launchers and is
held by the actual engine host while acquisition exists. Existing processes with
open video devices are rejected at start. Raw third-party scripts cannot be
forced to obey an advisory lock; use the documented protected commands.

Backend transport and command queues are bounded. The controller admits at most
two hosts/background units by default. Torch/ORT/OpenMP budgets are configurable;
existing OpenVINO backends retain their bounded two-thread settings. No three-
engine preload occurs. A release model host can remain warm for 90 seconds,
without a camera or tracking batch, and is evicted at the host budget or expiry.
Its new worker/pipeline has independent state. Purity-to-sealing shares its
existing compatible gold backend; cross-runtime model implementations are not
interchanged merely because filenames or hashes match.

Debug mode uses the same workers and verification paths. Existing widgets are
read-only views of bounded local image/metadata snapshots. Inputs are disabled
and their signals blocked. The viewer does not initialize a worker. Killing or
closing the viewer leaves the QCoreApplication host and backend running. No live
frames, masks, predictions or debugging endpoints are served to the browser.

Common manifests and result publications use fsync plus atomic replace. One
persistent publication worker copies completed source PDFs into the attempt's
`reports/` directory, preserving original engine archives and hashes. Publication
jobs survive restarts and retain retry payloads. A recovered result remains
REVIEW_REQUIRED for the interrupted execution. In-flight native inference lost
in a crash is not magically reconstructed; raw evidence stays available to the
original recovery/review tools and no missing prediction is fabricated.
