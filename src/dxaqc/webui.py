"""Local web interface: upload, live batch progress, per-image measurements, report download.

Everything runs in this process on this machine: the browser talks to 127.0.0.1 only, uploaded
studies live in a temporary folder that is removed when the server stops, and the page itself
is served from the package (no CDN, no external request). One worker processes one batch at a
time, so memory and CPU stay bounded like the CLI.
"""
from __future__ import annotations

import io
import json
import logging
import math
import mimetypes
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .api import analyze
from .inputs import _safe_name
from .report import write_report

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / 'webui_dist'
# the «Журнал» UI (ui-journal/), served at /journal/ next to the classic one at /
JOURNAL = Path(__file__).parent / 'webui_journal_dist'
MAX_REQUEST = 512 * 1024 ** 2
KEEP_JOBS = 5
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('font/woff2', '.woff2')


def _json_safe(value):
    """Plain JSON types only: numpy scalars/arrays converted, non-finite floats dropped to None."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return None if value.size > 4096 else _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _item(index, row, img, region, pred):
    """What the page needs for one file: the report row plus the measurements it draws."""
    item = {'i': index, 'path': row['path_to_study'], 'status': row['processing_status'],
            'region': row.get('anatomical_region') or None, 'quality': row.get('quality_class'),
            'violations': [c for c in (row.get('violation_type') or '').split(';') if c],
            'seconds': row.get('time_of_processing'), 'error': row.get('error_message') or '',
            'study_uid': row.get('study_uid') or '', 'image_uid': row.get('image_uid') or '',
            'has_image': img is not None}
    if img is not None:
        item['width'], item['height'] = int(img.pixels.shape[1]), int(img.pixels.shape[0])
        item['mm_per_px'] = img.meta.get('spacing_mm')
    if pred is not None:
        lm = pred.details.get('landmarks') or {}
        lt = lm.get('lesser_trochanter') or {}
        item.update({
            'scores': {k: v for k, v in pred.scores.items() if isinstance(v, (int, float, np.floating))},
            'points': lm.get('points') or {},
            'measurements': lm.get('measurements') or {},
            'lt': {'visible': lt.get('visible'), 'prominence_mm': lt.get('lt_prominence_mm'),
                   'apex': (lt.get('points') or {}).get('lt_apex'), 'edge': lt.get('edge')} if lt else None,
            'warnings': [w for w in pred.details.get('warnings', []) if isinstance(w, str)],
            'artifacts': _artifact_boxes(lm.get('artifacts')),
        })
    return _json_safe(item)


def _artifact_boxes(found) -> list[dict]:
    """Wires and metal the spine artifact detector found, as boxes in image pixels for the film."""
    boxes = []
    for kind, key in (('wire', 'wires'), ('metal', 'metal')):
        for a in (found or {}).get(key) or []:
            x, y, w, h = (int(v) for v in a['bbox'])
            boxes.append({'type': kind, 'x': x, 'y': y, 'w': w, 'h': h})
    return boxes


def _reviewed_row(row: dict, quality: int, violations: list[str]) -> dict:
    """A report row carrying the specialist's confirmed decision. When the specialist changed the
    verdict, quality_prob becomes certain (1 or 0); otherwise the model's probability stays."""
    out = {**row, 'quality_class': quality, 'violation_type': ';'.join(violations)}
    if row.get('quality_class') != quality:
        out['quality_prob'] = 1.0 if quality else 0.0
    return out


class Job:
    def __init__(self, root: Path):
        self.id = uuid.uuid4().hex[:12]
        self.dir = root / self.id
        (self.dir / 'input').mkdir(parents=True)
        (self.dir / 'img').mkdir()
        self.state = 'receiving'       # receiving -> queued -> running -> done | error
        self.name = ''
        self.files_received = 0
        self.paths: list[str] = []
        self.items: list[dict] = []
        self.rows: list[dict] = []
        self.summary: dict | None = None
        self.reviews: dict[int, dict] = {}
        self.error = ''
        self.started = self.finished = None
        self.lock = threading.Lock()

    def view(self, since: int = 0) -> dict:
        with self.lock:
            return {'id': self.id, 'state': self.state, 'name': self.name, 'total': len(self.paths) or None,
                    'done': len(self.items), 'paths': self.paths if since == 0 else None,
                    'items': self.items[since:], 'summary': self.summary, 'error': self.error,
                    'elapsed': round((self.finished or time.time()) - self.started, 1) if self.started else 0}


class JobStore:
    def __init__(self, predictor, model_lock=None):
        self.predictor = predictor
        self.model_lock = model_lock or threading.Lock()
        self.root = Path(tempfile.mkdtemp(prefix='dxaqc-ui-'))
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.queue: queue.Queue[Job] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True, name='dxaqc-ui-worker').start()

    def create(self) -> Job:
        job = Job(self.root)
        self.jobs[job.id] = job
        self.order.append(job.id)
        while len(self.order) > KEEP_JOBS:                  # bounded disk use: oldest batch goes
            old = self.jobs.pop(self.order.pop(0), None)
            if old is not None and old.state in ('done', 'error'):
                shutil.rmtree(old.dir, ignore_errors=True)
        return job

    def add_file(self, job: Job, name: str, body) -> None:
        rel = _safe_name(name or 'upload.dcm')
        if rel is None:
            raise ValueError(f'Unsafe file name: {name}')
        target = job.dir / 'input' / Path(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as out:
            shutil.copyfileobj(body, out, length=1024 * 1024)
        with job.lock:
            job.files_received += 1
            job.name = job.name or rel.parts[0]

    def start(self, job: Job) -> None:
        with job.lock:
            if job.state != 'receiving':
                return
            job.state = 'queued'
        self.queue.put(job)

    TZ_CODES = {'lumbar_spine': ('spine_scan_range', 'spine_axis_tilt', 'spine_artifact'),
                'hip_left': ('hip_positioning', 'hip_roi_margins'), 'hip_right': ('hip_positioning', 'hip_roi_margins')}

    def rescore(self, job: Job, index: int, points: dict) -> dict:
        """Run the model again on this image with the specialist's points; nothing is stored."""
        with job.lock:
            item = next((it for it in job.items if it['i'] == index), None)
        if item is None or not item.get('has_image') or not item.get('region'):
            raise ValueError('Для этого файла коррекция недоступна')
        if set(points) != set(item.get('points') or {}):
            raise ValueError('Нужен полный набор точек для этой области')
        pixels = np.asarray(Image.open(job.dir / 'img' / f'{index}.png').convert('L'))
        clean = {k: [float(v[0]), float(v[1])] for k, v in points.items()}
        try:
            with self.model_lock:
                pred = self.predictor.predict(pixels, item['region'], mm_per_px=item.get('mm_per_px'), points=clean)
        except TypeError as error:
            raise ValueError('Эта модель не поддерживает пересчёт по исправленным точкам') from error
        row = {'path_to_study': item['path'], 'processing_status': 'Success', 'anatomical_region': item['region'],
               'quality_class': int(pred.quality_class), 'violation_type': ';'.join(pred.violations),
               'time_of_processing': item.get('seconds'), 'study_uid': item.get('study_uid'),
               'image_uid': item.get('image_uid')}
        img = type('Img', (), {'pixels': pixels, 'meta': {'spacing_mm': item.get('mm_per_px')}})()
        return _item(index, row, img, item['region'], pred)

    def review(self, job: Job, index: int, points: dict | None, overrides: dict) -> dict:
        """The specialist confirms the image: corrected points (optional) and per-criterion decisions.
        The confirmed verdict replaces the model's in the report; the correction is kept for export."""
        with job.lock:
            pos = next((k for k, it in enumerate(job.items) if it['i'] == index), None)
            if pos is None:
                raise ValueError('Файл не найден в пакете')
            before = dict(job.items[pos])
        if before.get('status') != 'Success' or before.get('region') not in self.TZ_CODES:
            raise ValueError('Подтвердить можно только обработанный снимок')
        allowed = self.TZ_CODES[before['region']]
        if set(overrides) - set(allowed) or not all(isinstance(v, bool) for v in overrides.values()):
            raise ValueError('Неизвестный критерий для этой области')
        after = self.rescore(job, index, points) if points else dict(before)
        model_codes = [c for c in after['violations'] if c in allowed]
        final = sorted({c for c in model_codes if overrides.get(c, True)} | {c for c, v in overrides.items() if v},
                       key=allowed.index)
        original = before.get('review', {}).get('model') or {'quality': before['quality'], 'violations': before['violations'],
                                                              'points': before.get('points')}
        record = {'state': 'confirmed', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'corrected_points': bool(points), 'overrides': overrides, 'model': original}
        after.update({'violations': final, 'quality': int(bool(final)), 'review': record})
        with job.lock:
            job.items[pos] = after
            job.reviews[index] = {'path': before['path'], 'study_uid': before.get('study_uid'),
                                  'image_uid': before.get('image_uid'), 'region': before['region'],
                                  'points': after.get('points'), **record,
                                  'final': {'quality': after['quality'], 'violations': final}}
            if index < len(job.rows):
                job.rows[index] = _reviewed_row(job.rows[index], after['quality'], final)
        return after

    def _worker(self):
        while True:
            job = self.queue.get()
            with job.lock:
                job.state, job.started = 'running', time.time()
            inputs = [p for p in (job.dir / 'input').iterdir()]
            source = inputs[0] if len(inputs) == 1 and inputs[0].suffix.lower() == '.zip' else job.dir / 'input'

            def on_start(paths):
                with job.lock:
                    job.paths = list(paths)

            def on_result(index, total, row, img, region, pred):
                if img is not None:
                    Image.fromarray(img.pixels).save(job.dir / 'img' / f'{index}.png', optimize=False)
                item = _item(index, row, img, region, pred)
                with job.lock:
                    job.items.append(item)

            try:
                with self.model_lock:
                    rows, stats = analyze(source, predictor=self.predictor, on_start=on_start, on_result=on_result,
                                          sr_dir=job.dir / 'sr')
                with job.lock:
                    job.rows = rows
                    for index, rv in job.reviews.items():   # decisions confirmed while the batch was running
                        if index < len(rows):
                            rows[index] = _reviewed_row(rows[index], rv['final']['quality'], rv['final']['violations'])
                    # archive members that could not be extracted come back as extra Failure rows
                    for row in rows[len(job.items):]:
                        job.items.append(_item(len(job.items), row, None, None, None))
                    job.summary = _json_safe({k: v for k, v in stats.items() if k != 'zip_skipped'})
                    job.state = 'done'
            except Exception as error:  # noqa: BLE001 — surfaced to the page, the server keeps running
                log.exception('batch failed')
                with job.lock:
                    job.state, job.error = 'error', f'{type(error).__name__}: {error}'
            finally:
                with job.lock:
                    job.finished = time.time()


def handle_ui(handler, store: JobStore) -> bool:
    """Route one request; returns False when the path is not the UI's."""
    method, raw = handler.command, handler.path
    path, _, query = raw.partition('?')
    params = dict(p.split('=', 1) for p in query.split('&') if '=' in p)
    parts = [p for p in path.split('/') if p]
    if method == 'GET' and (path == '/' or path == '/index.html'):
        return _static(handler, 'index.html')
    if method == 'GET' and parts[:1] == ['assets']:
        return _static(handler, '/'.join(parts))
    if method == 'GET' and path == '/favicon.svg':   # the classic page links it from the root
        return _static(handler, 'favicon.svg')
    if method == 'GET' and path == '/journal':
        handler.send_response(301)
        handler.send_header('Location', '/journal/')
        handler.send_header('Content-Length', '0')
        handler.end_headers()
        return True
    if method == 'GET' and parts[:1] == ['journal']:
        return _static(handler, '/'.join(parts[1:]) or 'index.html', JOURNAL)
    if parts[:2] != ['api', 'jobs']:
        return False
    try:
        if method == 'POST' and len(parts) == 2:
            job = store.create()
            return _json(handler, 201, {'id': job.id})
        job = store.jobs.get(parts[2]) if len(parts) > 2 else None
        if job is None:
            return _json(handler, 404, {'error': 'Задача не найдена — возможно, сервер перезапускался.'})
        tail = parts[3:]
        if method == 'PUT' and tail == ['files']:
            length = int(handler.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_REQUEST:
                return _json(handler, 413, {'error': 'Файл пустой или больше 512 МБ.'})
            from urllib.parse import unquote
            store.add_file(job, unquote(params.get('name', '')), _Limited(handler.rfile, length))
            return _json(handler, 200, {'received': job.files_received})
        if method == 'POST' and tail == ['start']:
            store.start(job)
            return _json(handler, 202, job.view())
        if method == 'GET' and not tail:
            return _json(handler, 200, job.view(int(params.get('since', '0') or 0)))
        if method == 'GET' and len(tail) == 2 and tail[0] == 'image':
            f = job.dir / 'img' / f'{int(tail[1].removesuffix(".png"))}.png'
            if not f.is_file():
                return _json(handler, 404, {'error': 'no image'})
            return _bytes(handler, f.read_bytes(), 'image/png', cache=True)
        if method == 'POST' and len(tail) == 3 and tail[0] == 'items' and tail[2] in ('rescore', 'review'):
            length = int(handler.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024 * 1024:
                return _json(handler, 400, {'error': 'Пустой запрос'})
            body = json.loads(handler.rfile.read(length) or b'{}')
            index = int(tail[1])
            if tail[2] == 'rescore':
                return _json(handler, 200, store.rescore(job, index, body.get('points') or {}))
            return _json(handler, 200, store.review(job, index, body.get('points'), body.get('overrides') or {}))
        if method == 'GET' and tail == ['sr.zip']:
            if job.state != 'done':
                return _json(handler, 409, {'error': 'DICOM SR будет доступен после обработки пакета.'})
            import zipfile
            buf = io.BytesIO()
            from .sr import reviewed_sr
            with job.lock:
                reviewed = {row.get('_sr_file'): (row, job.reviews[i]) for i, row in enumerate(job.rows)
                            if i in job.reviews and row.get('_sr_file')}
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in sorted((job.dir / 'sr').glob('*.dcm')):
                    if f.name not in reviewed:
                        zf.write(f, f.name)
                        continue
                    row, rv = reviewed[f.name]      # the specialist's decision replaces the model's
                    ds = reviewed_sr(f, rv['final']['quality'], rv['final']['violations'], row.get('quality_prob'),
                                     rv['corrected_points'], rv['at'])
                    out = io.BytesIO()
                    ds.save_as(out, enforce_file_format=True)
                    zf.writestr(f.name, out.getvalue())
            stem = PurePosixPath(job.name or 'dxaqc').stem or 'dxaqc'
            return _bytes(handler, buf.getvalue(), 'application/zip', download=f'{stem}_SR.zip')
        if method == 'GET' and tail == ['corrections.json']:
            with job.lock:
                data = json.dumps({'batch': job.name, 'reviews': [job.reviews[k] for k in sorted(job.reviews)]},
                                  ensure_ascii=False, indent=1, default=float).encode('utf-8')
            stem = PurePosixPath(job.name or 'dxaqc').stem or 'dxaqc'
            return _bytes(handler, data, 'application/json; charset=utf-8', download=f'{stem}_corrections.json')
        if method == 'GET' and tail and tail[0] in ('report.csv', 'report.xlsx'):
            if job.state != 'done':
                return _json(handler, 409, {'error': 'Отчёт будет доступен после обработки пакета.'})
            suffix = Path(tail[0]).suffix
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / f'report{suffix}'
                write_report(job.rows, out, strict_columns=True)
                data = out.read_bytes()
            mime = 'text/csv; charset=utf-8' if suffix == '.csv' else \
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            stem = PurePosixPath(job.name or 'dxaqc').stem or 'dxaqc'
            return _bytes(handler, data, mime, download=f'{stem}_qc{suffix}')
        return _json(handler, 404, {'error': 'not_found'})
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        return _json(handler, 400, {'error': str(error)})


class _Limited(io.RawIOBase):
    """Read exactly `n` bytes of the request body, no more."""

    def __init__(self, raw, n):
        self.raw, self.left = raw, n

    def readable(self):
        return True

    def readinto(self, b):
        if self.left <= 0:
            return 0
        data = self.raw.read(min(len(b), self.left))
        if not data:
            raise ValueError('Загрузка прервалась')
        b[:len(data)] = data
        self.left -= len(data)
        return len(data)


def _json(handler, code, value):
    body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
    handler.send_response(code)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(body)
    return True


def _bytes(handler, data, mime, cache=False, download=None):
    handler.send_response(200)
    handler.send_header('Content-Type', mime)
    handler.send_header('Content-Length', str(len(data)))
    handler.send_header('Cache-Control', 'private, max-age=3600' if cache else 'no-store')
    if download:
        from urllib.parse import quote
        handler.send_header('Content-Disposition', f"attachment; filename*=UTF-8''{quote(download)}")
    handler.end_headers()
    handler.wfile.write(data)
    return True


def _static(handler, rel, root=STATIC):
    target = (root / rel).resolve()
    if root.resolve() not in target.parents and target != (root / 'index.html').resolve():
        return _json(handler, 404, {'error': 'not_found'})
    if not target.is_file():
        if rel == 'index.html':
            src = 'ui-journal' if root == JOURNAL else 'ui'
            return _bytes(handler, f'<!doctype html><meta charset=utf-8><p>Web UI is not built: cd {src} && bun run build'.encode(),
                          'text/html; charset=utf-8')
        return _json(handler, 404, {'error': 'not_found'})
    mime = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
    immutable = rel.startswith('assets/')
    handler.send_response(200)
    handler.send_header('Content-Type', mime + ('; charset=utf-8' if mime.startswith('text/') else ''))
    data = target.read_bytes()
    handler.send_header('Content-Length', str(len(data)))
    handler.send_header('Cache-Control', 'public, max-age=31536000, immutable' if immutable else 'no-cache')
    handler.end_headers()
    handler.wfile.write(data)
    return True
