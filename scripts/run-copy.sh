#!/bin/sh
# Portable alternative for Docker contexts without host bind-mount support.
set -eu
if [ "$#" -lt 2 ]; then echo "Usage: $0 INPUT OUTPUT.csv|xlsx [OPTIONS]" >&2; exit 2; fi
input=$1
output=$2
shift 2
case "$output" in *.csv) extension=csv;; *.xlsx) extension=xlsx;; *) echo 'Output must be .csv or .xlsx' >&2; exit 2;; esac
case "$input" in *.zip) source_name=input.zip;; *) source_name=input;; esac
test -e "$input" || { echo 'Input does not exist' >&2; exit 2; }
for destination in "$output" "${output%.*}.log" "${output%.*}.summary.json"; do
  test ! -e "$destination" || { echo "Destination exists; choose a new output name: $destination" >&2; exit 2; }
done
mkdir -p "$(dirname "$output")"
container=$(docker create --network none --cpus=2 --memory=4g --user 0:0 \
  --cap-drop ALL --security-opt no-new-privileges \
  "${DXAQC_IMAGE:-dxaqc:local}" -i "/tmp/$source_name" -o "/tmp/result/report.$extension" "$@")
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT HUP INT TERM
docker cp "$input" "$container:/tmp/$source_name"
docker start -a "$container" || true
code=$(docker inspect --format '{{.State.ExitCode}}' "$container")
# Copy only generated reports; failure diagnostics have already been printed.
if docker cp "$container:/tmp/result/report.$extension" "$output"; then
  docker cp "$container:/tmp/result/report.log" "${output%.*}.log"
  docker cp "$container:/tmp/result/report.summary.json" "${output%.*}.summary.json"
fi
exit "$code"
