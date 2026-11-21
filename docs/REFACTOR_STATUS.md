# Functional refactor: in progress

Branch: `refactor/functional-layout-20261120`. Baseline: `/tmp/embsys-sop-functional-baseline-oauhge60/project`. Original commit: `58351968453054b1a441e5b7113c736e2cc07769`. No commit or push.

Model assets have been moved into models/ and recorded in configs/model_registry.json and MODEL_MIGRATION.json. Canonical implementation modules were moved to engines/purity, engines/sealing and shared; compatibility aliases remain. Tracker fragments use a temporary shared-globals binder to preserve existing patching and object identity.

This refactor is not complete: the full regression suite and continuous physical camera handoff are not yet validated, legacy cleanup/security audits and developer documentation still need updating for relocated files, and the existing launcher integration test comparing original packet source bytes fails after path changes. Do not treat this branch as production-ready.

Restore the complete local baseline, including ignored models and local configuration, from the baseline copy above if abandoning this refactor; avoid overwriting new evidence.
