#!/usr/bin/env python3
"""Validate preparation without opening a camera or modifying engine algorithms."""
from pathlib import Path
import hashlib
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import RELEASING, JEWEL, PACKET, interpreter, engine_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', help='Comma-separated checks to rerun and merge into the report')
    selected_arg = parser.parse_args().checks
    selected = set(selected_arg.split(',')) if selected_arg else None
    logs = ROOT/'outputs/validation/packet-releasing'
    logs.mkdir(parents=True, exist_ok=True)
    results_path = ROOT/'docs/PACKET_RELEASING_VALIDATION.json'
    results = json.loads(results_path.read_text()) if selected and results_path.exists() else []
    def run(name, workflow, argv, cwd=ROOT, timeout=300, pythonpath=None):
        if selected and name not in selected:
            return
        results[:] = [row for row in results if row['check'] != name]
        env = engine_environment(workflow)
        env['QT_QPA_PLATFORM'] = 'offscreen'
        if pythonpath:
            env['PYTHONPATH'] = os.pathsep.join(map(str, pythonpath))
        Path(env.get('YOLO_CONFIG_DIR', ROOT/'runtime/PacketReleasing/ultralytics')).mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with (logs/(name+'.log')).open('w') as stream:
            try:
                result = subprocess.run([str(interpreter(workflow))]+argv, cwd=cwd, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                status = 'PASSED' if result.returncode == 0 else 'FAILED'
                detail = f'exit {result.returncode}'
            except subprocess.TimeoutExpired:
                status, detail = 'FAILED', 'Timeout'
        row = dict(check=name, status=status, seconds=round(time.monotonic()-started, 2),
                   detail=detail, log=str((logs/(name+'.log')).relative_to(ROOT)))
        results.append(row)
        print(name, status, flush=True)
    # Record complete installed-version inventories without credentials/index URLs.
    inventory = "import importlib.metadata as m,json; print(json.dumps(sorted({d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name']})))"
    for name in ('purity', 'packet', 'releasing'):
        rows = json.loads(subprocess.check_output([str(interpreter(name)), '-c', inventory], text=True))
        (ROOT/'configs'/(name+'-environment.txt')).write_text('\n'.join(rows)+'\n')
    run('launcher-regression', 'purity', ['-m','unittest','discover','-s','tests','-p','test_launcher.py','-v'])
    run('new-integration', 'purity', ['-m','unittest','discover','-s','tests','-p','test_releasing_integration.py','-v'])
    run('historical-integration-checks', 'purity', ['-m','unittest','discover','-s','tests','-p','test_integration.py','-v'])
    run('asset-integrity', 'purity', [str(ROOT/'scripts/check_preparation_assets.py')])
    run('purity-dependencies', 'purity', ['-m','pip','check'])
    run('sealing-dependencies', 'packet', ['-m','pip','check'])
    run('releasing-dependencies', 'releasing', ['-m','pip','check'])
    run('releasing-protected-entrypoint', 'releasing', [str(ROOT/'development/run_engine.py'), 'releasing', '--', '--help'])
    run('releasing-imports', 'releasing', [str(ROOT/'scripts/check_releasing.py'),'imports'])
    run('releasing-ui', 'releasing', [str(ROOT/'scripts/check_releasing.py'),'ui'])
    run('releasing-models', 'releasing', [str(ROOT/'scripts/check_releasing.py'),'models'])
    run('releasing-regression', 'releasing', ['-m','unittest','discover','-s','automated_app/tests'], RELEASING,
        pythonpath=[RELEASING/'automated_app/tests', RELEASING/'automated_app'])
    run('purity-imports', 'purity', [str(ROOT/'scripts/engine_check.py'),'imports-purity'])
    run('sealing-imports', 'packet', [str(ROOT/'scripts/engine_check.py'),'imports'])
    run('purity-sealing-handoff', 'packet', ['-m','unittest','-v','test_packet_stage','test_surveillance_handoff'], JEWEL,
        pythonpath=[JEWEL/'tests',ROOT,JEWEL,PACKET])
    registry = json.loads((ROOT/'configs/model_registry.json').read_text())['models']
    comparisons = []
    for path in sorted((RELEASING/'models').glob('*')):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        identical = [name for name,entry in registry.items() if entry['sha256'] == digest]
        comparisons.append(dict(model=path.name, sha256=digest, identical_registered_models=identical))
    (ROOT/'docs/PACKET_RELEASING_MODEL_COMPARISON.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    results[:] = [row for row in results if row['check'] not in ('live-camera-full-operator-flow', 'production-enablement')]
    results.append(dict(check='live-camera-full-operator-flow', status='NOT_TESTED', detail='Bench operation still required; preparation never opens a camera.'))
    results.append(dict(check='production-enablement', status='DISABLED', detail='Production command rejects releasing; launcher UI and purity-to-sealing handoff unchanged.'))
    (ROOT/'docs/PACKET_RELEASING_VALIDATION.json').write_text(json.dumps(results,indent=2)+'\n')
    lines = ['# Packet Releasing preparation validation', '', 'Date: 2026-11-28. Camera-free checks; source algorithms unchanged.',
             '', 'Historical integration assertions remain failing against pre-existing working-tree differences. See THREE_ENGINE_PREPARATION.md. Production stays disabled.', '', '| Check | Result | Detail |', '|---|---|---|']
    for r in results:
        lines.append(f"| {r['check']} | {r['status']} | {r.get('seconds','')} {r['detail']}; {r.get('log','')} |")
    (ROOT/'docs/PACKET_RELEASING_VALIDATION.md').write_text('\n'.join(lines)+'\n')
    return int(any(row['status']=='FAILED' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
