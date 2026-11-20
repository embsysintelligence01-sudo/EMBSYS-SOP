#!/usr/bin/env python3
"""Run preserved developer snapshots with the original asset/import context."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import JEWEL, PACKET, engine_environment
from scripts.prepare_runtime import maintenance_lock

REGISTRY = {
    'jewel_tracker_2.py': ('legacy/Jeweltracker/jewel_tracker_2.py', JEWEL/'jewel_tracker_2.py'),
    'jewel_tracker_6.py': ('legacy/Jeweltracker/jewel_tracker_6.py', JEWEL/'jewel_tracker_6.py'),
    'jewel_tracker_similarity.py': ('legacy/Jeweltracker/jewel_tracker_similarity.py', JEWEL/'jewel_tracker_similarity.py'),
    'pickup_snapshot.py': ('legacy/Packetsealing_001/triger_motion_rohan/pickup_trigger_test.py', PACKET/'pickup_snapshot.py'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('script',nargs='?',choices=REGISTRY)
    args, remaining=parser.parse_known_args()
    if args.list:
        print('\n'.join(REGISTRY))
        return 0
    if not args.script:
        parser.error('Select a development script or --list')
    source, context=REGISTRY[args.script]
    source=Path(__file__).resolve().parent/source
    os.environ.update(engine_environment('packet'))
    sys.path[:0]=[str(source.parent),str(context.parent),str(JEWEL),str(PACKET)]
    sys.argv=[str(context)]+remaining
    os.chdir(context.parent)
    # This virtual __file__ preserves model/camera paths without changing the
    # relocated algorithm source. Snapshots are developer tools, not production SOPs.
    with maintenance_lock():
        exec(compile(source.read_bytes(),str(context),'exec'),
             {'__name__':'__main__','__file__':str(context),'__package__':None})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
