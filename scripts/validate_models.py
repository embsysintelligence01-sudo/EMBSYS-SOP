#!/usr/bin/env python3
"""Offline logical-name/checksum/model-pair validation; optional engine inference."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shared.inference.model_registry import validate, registry

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['purity','packet'])
    args=parser.parse_args()
    errors=validate(args.stage)
    print(json.dumps({'assets':len(registry()['models']),'stage':args.stage,'errors':errors},indent=2))
    raise SystemExit(bool(errors))
