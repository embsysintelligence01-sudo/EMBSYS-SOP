#!/usr/bin/env python3
"""Repeatable validation; raw test logs are private and ignored by Git."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import JEWEL, PACKET, interpreter, engine_environment
from scripts.cleanup_layout import asset_location


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Launcher tests and static checks only')
    parser.add_argument('--checks', help='Comma-separated checks to rerun; merge with previous report')
    args = parser.parse_args()
    selected = set(args.checks.split(',')) if args.checks else None
    logs = ROOT / 'outputs/validation'
    logs.mkdir(parents=True, exist_ok=True)
    os.chmod(logs, 0o700)
    results = json.loads((ROOT/'docs/VALIDATION.json').read_text()) if selected and (ROOT/'docs/VALIDATION.json').exists() else []
    def record(name, status, detail):
        results[:] = [r for r in results if r['check'] != name]
        results.append(dict(check=name, status=status, detail=detail))
        print(f'{status}: {name}', flush=True)
    def run(name, mode, argv, cwd=ROOT, timeout=300):
        if selected and name not in selected:
            return
        env = engine_environment(mode)
        env.update(QT_QPA_PLATFORM='offscreen', PYTHONPATH=str(ROOT)+os.pathsep+str(JEWEL)+os.pathsep+str(PACKET))
        if name.endswith('-regression'):
            # Regression tests patch SCRIPT_DIR to temporary fixtures. Keep runtime
            # reviewer override out of that isolated test environment.
            env.pop('EMBSYS_REVIEWER_DIR', None)
            snapshot = json.loads((ROOT/'docs/SOURCE_SNAPSHOT.json').read_text())
            env['EMBSYS_REGRESSION_ASSETS'] = snapshot['Jeweltracker']['source']
        if name in ('handoff-regression', 'surveillance-export-regression'):
            env['PYTHONPATH'] = str(cwd/'tests') + os.pathsep + env['PYTHONPATH']
        log = logs / (name + '.log')
        started = time.monotonic()
        with log.open('w') as stream:
            os.chmod(log, 0o600)
            try:
                result = subprocess.run([str(interpreter(mode))] + argv, cwd=cwd, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                status = 'PASSED' if result.returncode == 0 else 'FAILED'
                detail = f'exit={result.returncode}; {time.monotonic()-started:.1f}s; private log: outputs/validation/{log.name}'
            except subprocess.TimeoutExpired:
                status, detail = 'FAILED', f'Test timed out after {timeout}s; private log: outputs/validation/{log.name}'
        import re
        summary = re.findall(r'(?:Ran \d+ tests? in [^\n]+|OK(?: \([^\n]+\))?|FAILED \([^\n]+\))', log.read_text(errors='replace'))
        if summary:
            detail += '; ' + '; '.join(summary[-2:])
        record(name, status, detail)
    from shared.inference.model_registry import validate, registry
    problems = validate()
    record('model-copy-integrity', 'FAILED' if problems else 'PASSED', f"{len(registry()['models'])} registered assets checked; errors: {problems}")
    record('required-model-paths', 'FAILED' if problems else 'PASSED', 'Central registry checks logical names, hashes and OpenVINO pairs.')
    run('cleanup-integrity','purity',['scripts/verify_cleanup.py'])
    run('storage-paths','packet',['scripts/check_storage_paths.py'])
    run('source-security-audit','purity',['scripts/audit.py'])
    run('launcher-tests','purity',['-m','unittest','discover','-s','tests','-v'])
    run('launcher-ui','purity',['launcher.py','--smoke-ui'])
    if not args.quick:
        run('purity-imports','purity',['scripts/engine_check.py','imports-purity'])
        run('slic-imports','packet',['scripts/engine_check.py','imports'])
        run('purity-ui','purity',['scripts/engine_check.py','purity-ui'])
        run('packet-ui','packet',[str(PACKET/'packet_app.py'),'--smoke-ui'],PACKET)
        run('purity-models','purity',['scripts/engine_check.py','purity-models'])
        run('packet-models','packet',[str(PACKET/'packet_app.py'),'--check-models'],PACKET)
        run('offline-ocr','packet',['scripts/ocr_check.py'],timeout=120)
        run('purity-regression','purity',['-m','unittest','discover','-s','tests','-v'],JEWEL,600)
        run('slic-regression','packet',['-m','unittest','discover','-s','tests','-v'],JEWEL,600)
        run('packet-regression','packet',['-m','unittest','discover','-s','tests','-v'],PACKET,600)
        run('rub-regression','purity',['-m','unittest','discover','-s','new_rub_detection/tests','-v'],JEWEL)
        run('mouth-lab-regression','packet',['-m','unittest','discover','-s','packet_mouth_lab/tests','-v'],PACKET)
        run('historical-regression','packet',['-m','unittest','discover','-s','tests','-v'],ROOT/'development/legacy/Jeweltracker/tracker_update_20261027')
        run('pickup-self-test','packet',[str(PACKET/'pickup_trigger_test.py'),'--self-test'],PACKET)
        run('pickup-motion-self-test','packet',[str(PACKET/'pickup_trigger_motion_class.py'),'--self-test'],PACKET)
        run('archived-pickup-self-test','packet',['development/run_legacy.py','pickup_snapshot.py','--self-test'])
        if selected:
            run('handoff-regression','packet',['-m','unittest','-v','test_packet_stage','test_surveillance_handoff'],JEWEL)
            run('surveillance-export-regression','packet',['-m','unittest','-v','test_surveillance_handoff'],PACKET)
    record('live-camera-and-operator-SOP', 'UNTESTED', 'Requires bench camera and operator: live ROI, full purity/sealing sequence, evidence export and physical camera release. No camera connection attempted.')
    if args.quick:
        record('isolated-OCR-inference','UNTESTED','Full validation includes isolated OCR with local cached models.')
    record('optional-reviewer-and-legacy-hardware','UNTESTED','External recording reviewer, Qt6 standalone acid UI and legacy Hailo/pickup hardware not exercised.')
    (ROOT/'docs/VALIDATION.json').write_text(json.dumps(results,indent=2)+'\n')
    report=['# Validation report','','Camera-free automated validation. Test logs are private under ignored outputs/validation/.','','| Check | Result | Detail |','|---|---|---|']
    report += [f"| {r['check']} | {r['status']} | {r['detail']} |" for r in results]
    (ROOT/'docs/VALIDATION.md').write_text('\n'.join(report)+'\n')
    return int(any(r['status']=='FAILED' for r in results))

if __name__ == '__main__':
    raise SystemExit(main())
