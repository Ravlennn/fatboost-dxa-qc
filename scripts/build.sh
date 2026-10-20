#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
test -f models/release/manifest.json || { echo 'Missing models/release. Run scripts/prepare_release.py or unpack supplied model bundle.' >&2; exit 2; }
exec docker build --pull -t "${DXAQC_IMAGE:-dxaqc:local}" "$@" .
