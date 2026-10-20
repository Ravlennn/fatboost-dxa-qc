"""Local HTTP server: the batch API (POST /v1/batch) and the web interface (GET /).

A single model lock means one batch runs at a time, whether it came from the API or the page.
"""
from __future__ import annotations

import json
import tempfile
import zipfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .api import analyze
from .webui import JobStore, handle_ui

MAX_UPLOAD = 512 * 1024 ** 2


def handler_for(predictor, store=None, model_lock=None):
    model_lock = model_lock or threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # polling would flood the console
            pass

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

        def do_PUT(self):
            if store is None or not handle_ui(self, store):
                self.reply(404, {'error': 'not_found'})

        def do_GET(self):
            if store is not None and handle_ui(self, store):
                return
            if self.path == '/health':
                self.reply(200, {'status': 'ok', 'model_id': getattr(predictor, 'model_id', None)})
            else:
                self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if store is not None and handle_ui(self, store):
                return
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
                    with model_lock:
                        rows, stats = analyze(path, predictor=predictor)
                self.reply(200, {'rows': rows, 'summary': stats})
            except (ValueError, OSError, EOFError, zipfile.BadZipFile) as error:
                self.reply(400, {'error': str(error)})
            except Exception:
                self.reply(500, {'error': 'Batch processing failed; check server logs'})

    return Handler


def serve(predictor, host='127.0.0.1', port=8080):
    if hasattr(predictor, 'warm_up'):
        print('DXA QC: загрузка моделей…', flush=True)
        predictor.warm_up()
    lock = threading.Lock()
    store = JobStore(predictor, model_lock=lock)
    with ThreadingHTTPServer((host, port), handler_for(predictor, store, lock)) as server:
        server.daemon_threads = True
        print(f'DXA QC: интерфейс http://{host}:{server.server_port}/  ·  API POST /v1/batch (application/zip)',
              flush=True)
        try:
            server.serve_forever()
        finally:
            import shutil
            shutil.rmtree(store.root, ignore_errors=True)   # uploaded studies never outlive the server
