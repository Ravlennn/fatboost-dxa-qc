#!/bin/sh
# Smoke test of a built image: version, doctor, web UI and API, a small batch through the CLI and the API.
# Usage: scripts/smoke_image.sh /absolute/test.zip [expected quality_prob values, 4 decimals, any order]
#   e.g. scripts/smoke_image.sh "$PWD/data/test.zip" 0.8904 0.9945 0.9861   (the organisers' «Для теста»)
# Image: $DXAQC_IMAGE (default dxaqc:local). Needs docker, curl and python3 on the host. Exit code 0 = all checks passed.
set -eu
if [ "$#" -lt 1 ]; then echo "Usage: $0 /absolute/test.zip [EXPECTED_PROB ...]" >&2; exit 2; fi
input=$1
shift
image=${DXAQC_IMAGE:-dxaqc:local}
port=${SMOKE_PORT:-18080}
# under $HOME: colima (and some Docker setups) can bind-mount only the home folder, not /tmp or /var/folders
work=$(mktemp -d "${SMOKE_TMP:-$HOME}/.dxaqc-smoke.XXXXXX")
name=dxaqc-smoke-$$
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "ok   $*"; }

# 1–2. version and model files
version=$(docker run --rm "$image" --version) || fail "--version"
ok "version $version"
docker run --rm "$image" doctor --model-dir /app/models/release >"$work/doctor.json" || fail "doctor"
ok "doctor $(cat "$work/doctor.json")"

# 3. HTTP server (the same settings as scripts/run-ui.sh)
docker run -d --name "$name" -p "127.0.0.1:$port:8080" --read-only --cpus=2 --memory=4g \
  --tmpfs /tmp:rw,nosuid,nodev,size=3g "$image" serve --host 0.0.0.0 --port 8080 >/dev/null
i=0
until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
  i=$((i + 1)); [ "$i" -lt 180 ] || fail "server did not start in 180 s"; sleep 1
done
ok "server up after ~${i} s (models are warmed up before it listens)"

# 4. pages, their assets and favicons
for page in / /journal/; do
  curl -fsS "http://127.0.0.1:$port$page" >"$work/page.html" || fail "GET $page"
  for ref in $(grep -o '\(src\|href\)="[^"]*"' "$work/page.html" | sed 's/^[a-z]*="//; s/"$//' | grep '^/'); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port$ref")
    [ "$code" = 200 ] || fail "GET $ref from $page: HTTP $code"
  done
  ok "GET $page and every asset it links: 200"
done

# 5–6. the batch API on the test package
curl -fsS -X POST -H 'Content-Type: application/zip' --data-binary "@$input" \
  "http://127.0.0.1:$port/v1/batch" >"$work/api.json" || fail "POST /v1/batch"
python3 - "$work/api.json" "$@" <<'PY' || fail "API rows"
import json, sys
rows = json.load(open(sys.argv[1]))['rows']
bad = [r for r in rows if r.get('processing_status') != 'Success']
assert rows and not bad, f'{len(bad)} of {len(rows)} rows are not Success: {bad[:2]}'
got = sorted(round(float(r['quality_prob']), 4) for r in rows)
want = sorted(round(float(v), 4) for v in sys.argv[2:])
assert not want or got == want, f'quality_prob {got} != expected {want}'
print(f'ok   API: {len(rows)} rows Success, quality_prob {got}' + (' = expected' if want else ''))
PY

# 7. the CLI as the organisers run it: CSV, XLSX and DICOM SR, no network
mkdir -p "$work/out"
in_dir=$(cd "$(dirname "$input")" && pwd -P)
in_name=$(basename "$input")
for out in result.csv result.xlsx; do
  docker run --rm --network none --read-only --cpus=2 --memory=4g --tmpfs /tmp:rw,nosuid,nodev,size=3g \
    --user "$(id -u):$(id -g)" --mount "type=bind,src=$in_dir/$in_name,dst=/input/$in_name,readonly" \
    --mount "type=bind,src=$work/out,dst=/output" "$image" \
    --input "/input/$in_name" --output "/output/$out" --sr "/output/sr-${out%.*}.zip" >"$work/cli.log" 2>&1 \
    || { tail -5 "$work/cli.log" >&2; fail "CLI -> $out"; }
done
[ -s "$work/out/result.csv" ] && [ -s "$work/out/result.xlsx" ] && [ -s "$work/out/sr-result.zip" ] || fail "CLI outputs missing"
python3 - "$work/out/result.csv" <<'PY' || fail "CSV rows"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')))
assert rows and all(r['processing_status'] == 'Success' for r in rows), 'Failure rows in CSV'
print(f'ok   CLI: CSV with {len(rows)} rows Success, XLSX and DICOM SR written')
PY
echo "PASS: $image"
