#!/usr/bin/env bash
# Run on the server from the repo root after unpacking the data archive:
#   tar -xf dxaqc_h100_data.tar && bash h100_kit/verify_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
M=h100_kit/dist/data_sha256.txt
[ -f "$M" ] || { echo "no $M — unpack dxaqc_h100_data.tar in the repo root first"; exit 2; }
if command -v sha256sum >/dev/null; then sha256sum -c --quiet "$M"; else shasum -a 256 -c --quiet "$M"; fi
echo "data OK: $(wc -l < "$M") files verified"
n=$(find -L "data/interim/train/Исследования" -type f | wc -l); echo "DICOM files: $n (expected 499)"
[ "$n" -eq 499 ] || { echo "WARNING: unexpected DICOM count"; exit 3; }
