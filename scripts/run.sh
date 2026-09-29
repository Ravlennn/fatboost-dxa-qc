#!/bin/sh
# Usage: scripts/run.sh /absolute/input/directory-or-zip /absolute/results.csv [CLI options]
set -eu
if [ "$#" -lt 2 ]; then echo "Usage: $0 INPUT OUTPUT.csv|xlsx [OPTIONS]" >&2; exit 2; fi
input=$1
output=$2
shift 2
input_parent=$(cd "$(dirname "$input")" && pwd -P)
input_name=$(basename "$input")
mkdir -p "$(dirname "$output")"
output_parent=$(cd "$(dirname "$output")" && pwd -P)
output_name=$(basename "$output")
exec docker run --rm --network none --read-only --cpus=2 --memory=4g \
  --tmpfs /tmp:rw,nosuid,nodev,size=3g --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$input_parent/$input_name,dst=/input/$input_name,readonly" \
  --mount "type=bind,src=$output_parent,dst=/output" \
  "${DXAQC_IMAGE:-dxaqc:local}" --input "/input/$input_name" --output "/output/$output_name" "$@"
