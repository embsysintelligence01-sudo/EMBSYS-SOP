#!/usr/bin/env python3
"""Production migration checks; no customer data, physical camera or serial hardware."""

from pathlib import Path
import argparse, json, os, re, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import interpreter, engine_environment, JEWEL, PACKET, RELEASING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", help="Comma-separated checks; merge results")
    args = parser.parse_args()
    selected = set(args.checks.split(",")) if args.checks else None
    report = ROOT / "docs/PRODUCTION_TEST_RESULTS.json"
    rows = json.loads(report.read_text()) if selected and report.exists() else []
    logs = ROOT / "outputs/production-validation"
    logs.mkdir(exist_ok=True)
    checks = [
        (
            "purity-diagnostics",
            "purity",
            ROOT,
            [
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_purity_diagnostics.py",
            ],
            [ROOT / "tests", JEWEL, PACKET, ROOT],
        ),
        (
            "linked-debug-close-replay",
            "controller",
            ROOT,
            [
                "deployment/replay_acceptance.py",
                "--engine",
                "linked",
                "--scenario",
                "debug-close",
            ],
            [ROOT],
        ),
        (
            "controller-api",
            "controller",
            ROOT,
            ["-m", "unittest", "discover", "-s", "tests", "-p", "test_production*.py"],
            [ROOT / "tests", ROOT],
        ),
        (
            "desktop-integration",
            "purity",
            ROOT,
            ["-m", "unittest", "discover", "-s", "tests"],
            [ROOT / "tests", JEWEL / "tests", JEWEL, PACKET, ROOT],
        ),
        (
            "purity-regression",
            "purity",
            ROOT,
            ["-m", "unittest", "discover", "-s", str(JEWEL / "tests")],
            [JEWEL / "tests", JEWEL, PACKET, ROOT],
        ),
        (
            "sealing-regression",
            "packet",
            PACKET,
            ["-m", "unittest", "discover", "-s", "tests"],
            [PACKET / "tests", JEWEL, PACKET, ROOT],
        ),
        (
            "release-regression",
            "releasing",
            RELEASING,
            ["-m", "unittest", "discover", "-s", "automated_app/tests"],
            [RELEASING / "automated_app/tests", RELEASING / "automated_app"],
        ),
        (
            "source-model-integrity",
            "purity",
            ROOT,
            ["scripts/check_preparation_assets.py"],
            [ROOT, JEWEL, PACKET],
        ),
        (
            "headless-handoff",
            "purity",
            ROOT,
            ["-m", "unittest", "test_headless_handoff"],
            [ROOT / "tests", JEWEL / "tests", JEWEL, PACKET, ROOT],
        ),
        (
            "release-imports",
            "releasing",
            ROOT,
            ["scripts/check_releasing.py", "imports"],
            [ROOT],
        ),
        (
            "releasing-replay",
            "controller",
            ROOT,
            ["deployment/replay_acceptance.py", "--engine", "releasing"],
            [ROOT],
        ),
        (
            "linked-replay",
            "controller",
            ROOT,
            ["deployment/replay_acceptance.py", "--engine", "linked"],
            [ROOT],
        ),
        (
            "standalone-sealing-replay",
            "controller",
            ROOT,
            ["deployment/replay_acceptance.py", "--engine", "packet"],
            [ROOT],
        ),
        (
            "reset-replay",
            "controller",
            ROOT,
            [
                "deployment/replay_acceptance.py",
                "--engine",
                "releasing",
                "--scenario",
                "reset",
            ],
            [ROOT],
        ),
        (
            "crash-replay",
            "controller",
            ROOT,
            [
                "deployment/replay_acceptance.py",
                "--engine",
                "releasing",
                "--scenario",
                "crash",
            ],
            [ROOT],
        ),
        (
            "debug-close-replay",
            "controller",
            ROOT,
            [
                "deployment/replay_acceptance.py",
                "--engine",
                "releasing",
                "--scenario",
                "debug-close",
            ],
            [ROOT],
        ),
        (
            "publication-jobs",
            "controller",
            ROOT,
            [
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_publication_jobs.py",
            ],
            [ROOT / "tests", ROOT],
        ),
        (
            "model-cache-replay",
            "controller",
            ROOT,
            [
                "deployment/replay_acceptance.py",
                "--engine",
                "releasing",
                "--scenario",
                "cache",
            ],
            [ROOT],
        ),
        (
            "http-service-recovery",
            "controller",
            ROOT,
            ["deployment/http_acceptance.py"],
            [ROOT],
        ),
        (
            "offline-sealing-ocr",
            "packet",
            ROOT,
            ["scripts/ocr_check.py"],
            [ROOT, JEWEL, PACKET],
        ),
        (
            "local-regression-fixtures",
            "purity",
            ROOT,
            [
                "-m",
                "unittest",
                "test_stone_model",
                "test_shared_gold_model",
                "test_new_rub_integration",
            ],
            [JEWEL / "tests", JEWEL, PACKET, ROOT],
        ),
        (
            "virtual-panel",
            "purity",
            ROOT,
            [
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_virtual_panel.py",
            ],
            [ROOT / "tests", ROOT],
        ),
    ]
    for name, mode, cwd, argv, paths in checks:
        if selected and name not in selected:
            continue
        python = (
            ROOT / "runtime/environments/controller/bin/python"
            if mode == "controller"
            else interpreter(mode)
        )
        env = os.environ.copy() if mode == "controller" else engine_environment(mode)
        env.update(
            QT_QPA_PLATFORM="offscreen",
            PYTHONPATH=os.pathsep.join(map(str, paths)),
            EMBSYS_REGRESSION_ASSETS=str(ROOT / "runtime/regression-assets"),
        )
        env.pop("EMBSYS_REVIEWER_DIR", None)
        start = time.monotonic()
        log = logs / (name + ".log")
        with log.open("w") as stream:
            try:
                result = subprocess.run(
                    [str(python), *argv],
                    cwd=cwd,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
                status = "PASSED" if result.returncode == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
        summary = re.findall(
            r"(?:Ran \d+ tests? in [^\n]+|OK(?: \([^\n]+\))?|FAILED \([^\n]+\))",
            log.read_text(errors="replace"),
        )
        row = dict(
            check=name,
            status=status,
            seconds=round(time.monotonic() - start, 2),
            summary="; ".join(summary[-2:]),
            log=str(log.relative_to(ROOT)),
        )
        rows = [r for r in rows if r["check"] != name]
        rows.append(row)
        report.write_text(json.dumps(rows, indent=2) + "\n")
        print(name, status, flush=True)
    lines = [
        "# Production integration test results",
        "",
        "Camera-free automated validation. Physical USB disconnect/reconnect, serial hardware, monitor hot unplug and power-cycle testing are pending.",
        "",
        "| Check | Status | Time (s) | Detail |",
        "|---|---|---:|---|",
    ]
    lines += [
        f"| {r['check']} | {r['status']} | {r['seconds']} | {r['summary']} · {r['log']} |"
        for r in rows
    ]
    (ROOT / "docs/PRODUCTION_TEST_RESULTS.md").write_text("\n".join(lines) + "\n")
    return int(any(row["status"] != "PASSED" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
