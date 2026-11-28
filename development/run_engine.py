#!/usr/bin/env python3
"""Independent PyQt engine windows under the SOP ownership lock. No queue/handoff changes."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import command, engine_environment
from scripts.prepare_runtime import maintenance_lock


from app.ownership import camera_owners


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=('purity', 'packet', 'releasing'))
    parser.add_argument('--describe', action='store_true', help='Print launch paths without opening a camera')
    values = list(sys.argv[1:] if argv is None else argv)
    forwarded = []
    if '--' in values:
        index = values.index('--')
        values, forwarded = values[:index], values[index+1:]
    args, remaining = parser.parse_known_args(values)
    remaining += forwarded
    argv, cwd = command(args.engine, development=True)
    if args.describe:
        print('Development only' if args.engine == 'releasing' else 'Existing engine')
        print('Working directory:', cwd)
        print('Command:', argv + remaining)
        return 0
    try:
        with maintenance_lock() as lock_fd:
            owners = camera_owners()
            if owners:
                raise RuntimeError('Camera already open by process(es): ' + ', '.join(map(str, owners)))
            environment = engine_environment(args.engine)
            if args.engine == 'releasing':
                Path(environment['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
            child = subprocess.Popen(argv + remaining, cwd=cwd, env=environment,
                                     pass_fds=(lock_fd,), start_new_session=True)
            # Inherited lock remains held if this runner exits unexpectedly.
            while True:
                try:
                    return child.wait()
                except KeyboardInterrupt:
                    print('Close the engine normally so evidence export can finish.', file=sys.stderr)
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
