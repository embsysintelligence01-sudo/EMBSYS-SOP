#!/usr/bin/env python3
"""Copy existing working environments and Python runtime without changing package versions."""

from pathlib import Path
import json, os, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import interpreter


def main():
    target = ROOT / "runtime/environments"
    target.mkdir(parents=True, exist_ok=True)
    modes = {
        "purity": "purity",
        "packet": "packet",
        "ocr": "ocr",
        "releasing": "releasing",
    }
    from app.config import settings, resolve_path

    source = {
        "purity": interpreter("purity").parent.parent,
        "ocr": resolve_path(settings()["ocr_python"]).parent.parent,
        "releasing": interpreter("releasing").parent.parent,
    }
    record_path = ROOT / "runtime/environment-migration.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name, src in source.items():
        dst = target / name
        if dst.exists():
            if not (dst / ".embsys-copy-complete").exists():
                raise RuntimeError("Incomplete environment: " + str(dst))
            continue
        base = Path(
            subprocess.check_output(
                [str(src / "bin/python"), "-c", "import sys;print(sys.base_prefix)"],
                text=True,
            ).strip()
        )
        newbase = base
        if base != Path("/usr") and not base.is_relative_to(ROOT / "runtime/python"):
            newbase = ROOT / "runtime/python" / base.name
            if not newbase.exists():
                shutil.copytree(base, newbase, symlinks=True)
        shutil.copytree(
            src, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__")
        )
        if newbase != base:
            for entry in (dst / "bin").iterdir():
                if entry.is_symlink() and ("python" in entry.name):
                    link = os.readlink(entry)
                    if str(base) in link:
                        entry.unlink()
                        entry.symlink_to(link.replace(str(base), str(newbase)))
        cfg = dst / "pyvenv.cfg"
        data = (
            cfg.read_text().replace(str(src), str(dst)).replace(str(base), str(newbase))
        )
        cfg.write_text(data)
        for entry in (dst / "bin").iterdir():
            if entry.is_file() and not entry.is_symlink():
                raw = entry.read_bytes()
                if raw.startswith(b"#!") or entry.name.startswith("activate"):
                    entry.write_bytes(raw.replace(str(src).encode(), str(dst).encode()))
        subprocess.run(
            [
                str(dst / "bin/python"),
                "-c",
                "import cv2,numpy;print(cv2.__version__,numpy.__version__)",
            ],
            check=True,
        )
        (dst / ".embsys-copy-complete").write_text(str(src) + "\n")
        records[name] = dict(
            source=str(src), destination=str(dst), python_base=str(newbase)
        )
    # Keep normal engines' versions and all user configuration; change only interpreter locations.
    p = ROOT / "configs/local.toml"
    text = p.read_text()
    for key, name in [
        ("purity_python", "purity"),
        ("packet_python", "purity"),
        ("ocr_python", "ocr"),
        ("releasing_python", "releasing"),
    ]:
        lines = text.splitlines()
        value = json.dumps(str((target / name / "bin/python").relative_to(ROOT)))
        lines = [
            key + " = " + value if line.startswith(key + " ") else line
            for line in lines
        ]
        if not any(line.startswith(key + " ") for line in lines):
            lines.append(key + " = " + value)
        text = "\n".join(lines) + "\n"
    p.write_text(text)
    cache = ROOT / "runtime/model-cache/paddlex"
    original = Path.home() / ".paddlex"
    if original.exists() and not cache.exists():
        shutil.copytree(
            original,
            cache,
            symlinks=False,
            ignore=shutil.ignore_patterns("temp", "locks", "func_ret"),
        )
    (ROOT / "runtime/environment-migration.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print("Local environments provisioned; original installations untouched.")


if __name__ == "__main__":
    main()
