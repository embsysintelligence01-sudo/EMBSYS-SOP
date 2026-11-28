"""Original-source preservation with explicit, checksum-reviewed lifecycle-only migrations."""

from pathlib import Path
import hashlib, json, subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify_sources():
    old = json.loads((ROOT / "configs/packet_releasing_snapshot.json").read_text())
    migration = json.loads((ROOT / "configs/production_migration.json").read_text())
    approved = migration["changes"]
    errors = []
    groups = [
        ("engines/PacketReleasing/", old["files"]),
        ("", old["existing_engine_source_baseline"]),
    ]
    for prefix, entries in groups:
        for relative, before in entries.items():
            name = prefix + relative
            path = ROOT / name
            if not path.is_file():
                errors.append(name + ": missing")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual == before:
                continue
            change = approved.get(name)
            if not change or change["before"] != before or change["after"] != actual:
                errors.append(name + ": unexpected source drift")
            elif path.suffix == ".py" and (ROOT / ".git").exists():
                result = subprocess.run(
                    ["git", "show", migration["backup_branch"] + ":" + name],
                    cwd=ROOT,
                    capture_output=True,
                )
                if (
                    result.returncode
                    or hashlib.sha256(result.stdout).hexdigest() != before
                ):
                    errors.append(name + ": original baseline unavailable")
    return errors
