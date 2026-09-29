#!/usr/bin/env bash
# Run on the Mac. Builds a self-contained folder for a server WITHOUT git access:
# code of the current commit + data + model bundle + ImageNet weights + instructions.
#   bash h100_kit/build_portable_kit.sh [dest=~/Downloads/dxaqc_h100_kit]
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-$HOME/Downloads/dxaqc_h100_kit}
[ -z "$(git status --porcelain -- src scripts tests h100_kit docs)" ] || { echo "commit your changes first"; exit 2; }
[ -f h100_kit/dist/dxaqc_h100_data.tar ] || bash h100_kit/make_data_archive.sh
mkdir -p "$DEST"
P="$DEST/dxaqc"; [ -e "$P" ] && { echo "$P exists — remove it first"; exit 2; }
mkdir -p "$P"
git archive HEAD | tar -x -C "$P"                         # code, scripts, docs, tests, uv.lock (no .git)
git rev-parse HEAD > "$P/COMMIT"
tar -xf h100_kit/dist/dxaqc_h100_data.tar -C "$P"         # data + small outputs, with sha256 manifest
mkdir -p "$P/models"
cp -R models/release "$P/models/release"                  # model bundle for CLI / measurements
cp -R models/torch_home "$P/models/torch_home"            # ImageNet weights already downloaded
cp h100_kit/START_HERE.md "$DEST/START_HERE.md"
( cd "$P" && bash h100_kit/verify_data.sh >/dev/null && echo "data verified" )
( cd "$DEST" && find dxaqc -type f ! -name '.DS_Store' -print0 | sort -z | xargs -0 shasum -a 256 > KIT_SHA256.txt )
du -sh "$DEST"; echo "Kit ready: $DEST (commit $(cat "$P/COMMIT" | cut -c1-8))"
