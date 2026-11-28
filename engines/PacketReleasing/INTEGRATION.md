# Packet Releasing — preparation integration

The original working automated and manual source, tests, Linux dependency lock,
seven models and current ROI/baseline files were copied from `packet-cut-system`
on 2026-11-28. The copied source is byte-identical; the source checkout and its
historical reports remain intact. No detection or evidence rule was changed.

Launch the existing automated PyQt6 window independently:

```bash
./run_development.sh --camera 0
```

This delegates to the root development runner and holds the same camera lock as
Purity Testing and Packet Sealing. Close an engine normally after saving before
starting another. Packet Releasing is not available from the production launcher.

The preserved `run_automated.sh` is the original checkout launcher; it expects a
local `.venv-ubuntu` and has no SOP ownership integration. Use `run_development.sh`
in this checkout. Do not launch raw engine entry points or backup applications
alongside a managed workflow; direct scripts do not participate in ownership.

The integrated entry point is `automated_app/app.py`. Its original manual UI,
CLI, compliance UI, inspection modules and regression tests remain available in
source for debugging; they are not production routes. The configured interpreter
is external and must remain installed. No environment or package was relocated,
installed or upgraded. Use `RELEASING_PYTHON` or `configs/local.toml` to change it.

Models remain here under `models/`. Current ROI/baselines remain under their
original `packet_cut_outputs` paths. New reports save under this engine's
`packet_cut_outputs/automated_sessions/`; baseline configurations do not transfer
a prior completed session. Old reports/videos were not copied. All of these
local assets are ignored by Git and must be provisioned separately on a clone.

Read `../../docs/THREE_ENGINE_PREPARATION.md` and
`../../docs/PACKET_RELEASING_VALIDATION.md` from the repository root for details
(the documentation is at `EMBSYS-SOP/docs/`, outside `engines/`).
