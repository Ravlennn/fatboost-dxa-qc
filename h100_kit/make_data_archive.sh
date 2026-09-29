#!/usr/bin/env bash
# Run on the Mac (repo root or anywhere): packs everything the H100 runs need into one tar
# with a SHA-256 manifest. Symlinks are dereferenced. Output: h100_kit/dist/dxaqc_h100_data.tar
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=h100_kit/dist; mkdir -p "$OUT"
LIST=$(grep -v '^#' h100_kit/data_files.txt | sed '/^\s*$/d')
PRESENT=(); while IFS= read -r p; do [ -e "$p" ] && PRESENT+=("$p") || echo "skip (absent): $p"; done <<< "$LIST"
# checksums of every file that goes into the archive
: > "$OUT/data_sha256.txt"
for p in "${PRESENT[@]}"; do find -L "$p" -type f ! -name '.DS_Store' -print0 | sort -z | xargs -0 shasum -a 256 >> "$OUT/data_sha256.txt"; done
tar -chf "$OUT/dxaqc_h100_data.tar" --exclude .DS_Store "${PRESENT[@]}" "$OUT/data_sha256.txt"
shasum -a 256 "$OUT/dxaqc_h100_data.tar" > "$OUT/dxaqc_h100_data.tar.sha256"
echo "files: $(wc -l < "$OUT/data_sha256.txt")"; ls -lh "$OUT/dxaqc_h100_data.tar"; cat "$OUT/dxaqc_h100_data.tar.sha256"
