# Installation and operation on this Ubuntu mini PC

The backend is persistent. Browser and desktop clients are separate processes.
The user service and desktop launchers are installed on this mini PC, and user
lingering is enabled. The validated three adapters are enabled. The controller
is running on `http://127.0.0.1:8765` with an initially empty production queue.
Open **EMBSYS SOP** from the applications menu to reopen the queue and panel.

For a manual production/debug launch, first stop the installed controller:

```bash
systemctl --user stop embsys-sop.service
```

From the EMBSYS-SOP directory:

```bash
python3 main.py --mode production --hardware virtual
# Separate terminal in a desktop session:
python3 ui/virtual_panel.py
# Browser:
# http://127.0.0.1:8765
```

`main.py` uses the local controller environment automatically if needed. Only one
controller can own the same production runtime. Uvicorn uses one worker. The
browser schedules/removes never-started activities; START/STOP/confirmed RESET
come from the virtual hardware panel. No activity automatically starts after
STOP. Closing the browser or panel leaves the controller and engine running.
Do not run another manually started backend when the service is already active.

Debug uses the same queue, controller and workers:

```bash
python3 main.py --mode debug --hardware virtual
python3 ui/virtual_panel.py
```

The debug engine UI is a separate read-only process. Close it without ending
surveillance. Use the original protected interfaces for interactive development:

```bash
./development/run_engine.sh purity
./development/run_engine.sh packet
./development/run_engine.sh releasing --camera 0
```

Finish the active camera owner before independent testing. These commands do not
alter production loan records. The old `./run.sh` launcher remains a legacy
independent-development interface; it is not the persistent production queue.
After manual testing, close the manual controller and restore the service with
`systemctl --user start embsys-sop.service`.

The production ledger is `runtime/production/queue.sqlite3`. Evidence is retained
under `runtime/production/sessions/<Loan ID>/<session ID>/<activity ID>/<attempt ID>/`;
published PDFs are in each attempt's `reports/` directory. Reset creates a new
attempt and preserves the preceding directory. Legacy evidence stays in its
original engine locations and is not silently assigned to a new loan.

The software and local environments are provisioned on this PC. To reproduce
from the existing working environments, run `./deployment/install.sh`.
It installs backend packages in their separate environment and copies engine
packages without changing versions. A fresh clone needs the private model assets,
calibration, OCR cache and engine environments/wheelhouse; Git does not contain
those assets. Verify before use:

```bash
runtime/environments/controller/bin/python deployment/validate.py
```

A system-level service can be installed with:

```bash
sudo python3 deployment/install_service.py --account embsys
```

A user service is also supported:

```bash
python3 deployment/install_service.py --user
loginctl --no-ask-password enable-linger embsys
systemctl --user status embsys-sop.service
journalctl --user -u embsys-sop.service
```

Lingering lets the user controller start at boot without a desktop login. The
user account still needs device permissions. Panel/viewer clients belong in a
desktop session and are not required by the backend. Service shutdown requests
safe STOP; it does not kill an engine to switch workflows. Engines drain source
evidence on controller EOF; interrupted records remain reviewable on restart.
`KillMode=process` avoids destroying evidence workers with the API parent.

Physical serial is a prepared interface boundary, not a commissioned device:

```bash
python3 main.py --mode production --hardware serial
python3 main.py --mode debug --hardware serial
```

These fail explicitly until a validated serial adapter is supplied. Virtual
hardware is fully testable without USB buttons/LCD/LEDs.

For authenticated LAN access, supply an API token of at least 32 characters using
the ignored `configs/service.env` or process environment, plus a TLS certificate
and key. Never place actual credentials in Git:

```bash
EMBSYS_API_TOKEN=YOUR_PRIVATE_LONG_TOKEN python3 main.py \
  --mode production --hardware virtual --host 0.0.0.0 \
  --tls-cert /private/path/certificate.pem --tls-key /private/path/key.pem
```

The browser requests the token when needed; it stays in tab session storage.
The virtual panel reads the same environment token and accepts `--url` for the
server. Default binding is localhost. There is no video or predictions route.

Recorded-input acceptance uses an isolated database and synthetic test Loan IDs:

```bash
runtime/environments/controller/bin/python deployment/replay_acceptance.py --engine linked
runtime/environments/controller/bin/python deployment/replay_acceptance.py --engine releasing --scenario reset
```

A backend replay profile requires an existing video and `--runtime` under
`runtime/validation/`; it cannot use the production database. Do not infer
physical hardware validation from replay results.
