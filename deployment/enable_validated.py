#!/usr/bin/env python3
"""Enable real workflow adapters only after recorded integration checks have passed."""

from pathlib import Path
import hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.source_integrity import verify_sources

REQUIRED = {
    "controller-api",
    "desktop-integration",
    "purity-regression",
    "sealing-regression",
    "release-regression",
    "source-model-integrity",
    "headless-handoff",
    "release-imports",
    "releasing-replay",
    "linked-replay",
    "standalone-sealing-replay",
    "reset-replay",
    "crash-replay",
    "debug-close-replay",
    "publication-jobs",
    "model-budget",
    "model-cache-replay",
    "virtual-panel",
    "http-service-recovery",
    "offline-sealing-ocr",
    "local-regression-fixtures",
}


def main():
    rows = json.loads((ROOT / "docs/PRODUCTION_TEST_RESULTS.json").read_text())
    byname = {r["check"]: r for r in rows}
    failed = [
        name
        for name in sorted(REQUIRED)
        if name not in byname or byname[name]["status"] != "PASSED"
    ]
    drift = verify_sources()
    if failed or drift:
        raise SystemExit(
            "Enablement refused. Failed/missing checks: "
            + str(failed)
            + "; source drift: "
            + str(drift)
        )
    path = ROOT / "configs/production.toml"
    text = path.read_text()
    lines = text.splitlines()
    lines = [
        (
            'enabled = ["purity", "packet", "releasing"]'
            if line.startswith("enabled =")
            else line
        )
        for line in lines
    ]
    path.write_text("\n".join(lines) + "\n")
    evidence = dict(
        software_integration="PASSED",
        enabled=["purity", "packet", "releasing"],
        required_checks=sorted(REQUIRED),
        result_sha256=hashlib.sha256(
            (ROOT / "docs/PRODUCTION_TEST_RESULTS.json").read_bytes()
        ).hexdigest(),
        source_migration_sha256=hashlib.sha256(
            (ROOT / "configs/production_migration.json").read_bytes()
        ).hexdigest(),
        physical_hardware_validation="PENDING - not inferred from replay",
        backup_branch="backup/pre-production-20261128",
    )
    (ROOT / "configs/production_acceptance.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(
        "Enabled three real adapters after software integration checks. Physical bench acceptance remains separate."
    )


if __name__ == "__main__":
    main()
