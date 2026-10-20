#!/bin/sh
# Web interface in the container: http://127.0.0.1:${PORT:-8080}/
# The port is published on the host's loopback only; studies stay on this machine.
set -eu
port=${PORT:-8080}
echo "DXA QC: http://127.0.0.1:$port/  (интерфейс «Журнал»: http://127.0.0.1:$port/journal/)"
exec docker run --rm -p "127.0.0.1:$port:8080" --read-only --cpus=2 --memory=4g \
  --tmpfs /tmp:rw,nosuid,nodev,size=3g --user "$(id -u):$(id -g)" \
  "${DXAQC_IMAGE:-dxaqc:local}" serve --host 0.0.0.0 --port 8080
