"""Put the out-of-scope view gate (scripts/train_view_gate.py) into the release bundle."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.bundle import sha256, verify_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, default=ROOT / 'models/release')
    ap.add_argument('--head', type=Path, default=ROOT / 'outputs/view_gate_v1/gate_head.json')
    args = ap.parse_args()
    manifest = verify_bundle(args.bundle)
    dst = args.bundle / 'view_gate.json'
    shutil.copy(args.head, dst)
    manifest['view_gate'] = 'view_gate.json'
    manifest['files']['view_gate.json'] = sha256(dst)
    manifest['model_id'] = manifest['model_id'].split('+')[0] + '+landmarks_v1+lt_v1+art_v1+gate_v1'
    (args.bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    verify_bundle(args.bundle)
    print(json.dumps({'threshold': json.loads(dst.read_text())['threshold'], 'model_id': manifest['model_id']}))


if __name__ == '__main__':
    main()
