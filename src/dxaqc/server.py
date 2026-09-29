"""Small local batch HTTP adapter; single worker bounds memory/CPU use."""
from __future__ import annotations

import json
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .api import analyze

MAX_UPLOAD = 512 * 1024 ** 2


def handler_for(predictor):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(60)

        def reply(self, code, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'status': 'ok', 'model_id': getattr(predictor, 'model_id', None)})
            else:
                self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/v1/batch':
                self.reply(404, {'error': 'not_found'})
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/zip':
                    self.reply(415, {'error': 'Expected application/zip body'})
                    return
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_UPLOAD:
                    self.reply(413, {'error': 'Provide Content-Length between 1 and 536870912 bytes'})
                    return
                with tempfile.TemporaryDirectory(prefix='dxaqc-http-') as directory:
                    path = Path(directory) / 'input.zip'
                    with path.open('wb') as output:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError('Incomplete request body')
                            output.write(chunk)
                            remaining -= len(chunk)
                    rows, stats = analyze(path, predictor=predictor)
                self.reply(200, {'rows': rows, 'summary': stats})
            except (ValueError, OSError, EOFError, zipfile.BadZipFile) as error:
                self.reply(400, {'error': str(error)})
            except Exception:
                self.reply(500, {'error': 'Batch processing failed; check server logs'})

    return Handler


def serve(predictor, host='127.0.0.1', port=8080):
    with HTTPServer((host, port), handler_for(predictor)) as server:
        print(f'DXA QC API http://{host}:{server.server_port}; POST /v1/batch (application/zip)', flush=True)
        server.serve_forever()
