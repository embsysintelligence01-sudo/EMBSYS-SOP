# File migration and cleanup report

The migration preserves the complete three-engine source. The earlier 892-file
baseline includes 307 first-party engine files and 585 files from the preserved
Windows environment; that environment is not an Ubuntu production dependency. No uncertain source,
active model, training tool, experiment or historical recording was deleted.
The recoverable baseline is a Git backup branch and an ignored source archive.
`configs/production_migration.json` records reviewed source changes against the
original checksum inventory; `services/source_integrity.py` verifies them and the
original backup blobs. Baseline failures remain recorded, rather than erased.

New production structure:

```text
main.py
core/         controller, SQLite queue/session/audit store, states, engine contract
backend/      FastAPI scheduling and hardware-command API
frontend/     queue-only HTML/CSS/JavaScript
hardware/     common command interface, HTTP virtual adapter, serial boundary
ui/           independent PyQt5 panel and read-only engine diagnostic viewers
services/     real-worker hosts/adapters, ownership, models/cache lifecycle,
              identity handoff, evidence/publication/recovery, diagnostics
engines/      Jeweltracker/, Packetsealing_001/, PacketReleasing/,
              existing purity/ and sealing/ implementation packages
models/       shared/, purity/, sealing/, releasing/ (private binaries)
configs/      public templates/registries and ignored local settings
runtime/      environments, Python, OCR cache, production DB/sessions/logs,
              validation records and preserved backups (all ignored)
development/  protected independent interfaces and original experiments
deployment/   provisioning, systemd units, installation and acceptance checks
tests/        controller/API/hardware/publication and original engine tests
```

Release's seven models were copied to `models/releasing`, byte-verified, then
its engine-relative directory was replaced by a compatibility link. The original
copy remains intact in runtime backups. Logical model entries were added without
repointing different preprocessing or confidence rules to a new model. Common
hashes are documented in `PACKET_RELEASING_MODEL_COMPARISON.json`; the existing
purity-to-sealing gold backend is actually shared in-process.

The old Windows `.venv-slic` in the pickup experiment was moved intact to
`runtime/preserved-windows-environment`; its original path remains a link. It is
not used for Linux production. Existing live Linux environments were copied into
runtime, with their original folders retained. Runtime evidence from previous
work was not moved or merged into new production loan sessions.

Legacy numbered trackers, training tools, mouth experiments and independent
interfaces remain preserved. The earlier `CLEANUP_REPORT.md` continues to describe
the previous cleanup; it is not a report of deletions performed here. New source
was added under the requested production folders instead of merging detectors.

New captures use Loan/session/activity/attempt directories under
`runtime/production/sessions`. Independent development retains its original
engine-local evidence paths and never writes the production SQLite ledger.
Original retention and archive verification remain authoritative for source
recordings. Common published PDFs are byte-identical copies with recorded hashes.

Models, environments, local calibration/credentials, customer records, recordings,
logs and validation fixtures are ignored by Git. No remote push or deployment to
an external service is performed by this migration.
