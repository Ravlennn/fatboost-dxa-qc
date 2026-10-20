"""The web interface's job API: upload, live progress, image and report downloads."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_release_cli import SyntheticPredictor, synthetic_dicom, zip_bytes  # noqa: E402
from dxaqc.predictor import Prediction  # noqa: E402
from dxaqc.server import handler_for  # noqa: E402
from dxaqc.webui import JobStore  # noqa: E402


class PointAwarePredictor:
    """Hip whose ROI verdict follows the ROI box: top edge above y=5 means the margin is too small."""
    name, model_id = 'point_aware_test', 'synthetic'
    POINTS = {'roi_top': [20, 3], 'roi_bottom': [20, 28], 'roi_left': [5, 15], 'roi_right': [35, 15],
              'neck_center': [20, 12], 'neck_head_side': [26, 9], 'neck_troch_side': [14, 15]}

    def _pred(self, points):
        bad = points['roi_top'][1] < 5
        return Prediction(int(bad), ['hip_roi_margins'] if bad else [], {'quality': 0.9 if bad else 0.1},
                          {'warnings': [], 'landmarks': {'points': points, 'measurements': {'roi_top_mm': points['roi_top'][1] * 0.6}}})

    def predict_image(self, pixels, mm_per_px=None):
        return 'hip_right', self._pred(dict(self.POINTS))

    def predict(self, pixels, region, mm_per_px=None, points=None):
        return self._pred(points or dict(self.POINTS))


class WebUIJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = JobStore(SyntheticPredictor())
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.store.predictor, self.store))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def call(self, method, path, data=None):
        req = urllib.request.Request(self.base + path, data=data, method=method)
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read()
            return r.status, r.headers, body

    def test_both_interfaces_are_served(self):
        # classic UI at /, the «Журнал» UI at /journal/ — each with its own assets
        from dxaqc.webui import JOURNAL, STATIC
        for prefix, root in (('', STATIC), ('/journal', JOURNAL)):
            if not (root / 'index.html').is_file():
                continue
            status, headers, body = self.call('GET', prefix + '/')
            self.assertEqual(status, 200)
            self.assertIn('text/html', headers['Content-Type'])
            asset = next((root / 'assets').glob('*.js')).name
            self.assertIn(f'{prefix}/assets/{asset}', body.decode())
            status, _, _ = self.call('GET', f'{prefix}/assets/{asset}')
            self.assertEqual(status, 200)
            # every favicon the page links must exist (the classic page links /favicon.svg from the root)
            for href in re.findall(r'href="([^"]*favicon[^"]*)"', body.decode()):
                status, _, _ = self.call('GET', href)
                self.assertEqual(status, 200, href)
        with self.assertRaises(urllib.error.HTTPError):
            self.call('GET', '/journal/../webui.py')

    def wait_done(self, job_id):
        for _ in range(200):
            _, _, body = self.call('GET', f'/api/jobs/{job_id}')
            view = json.loads(body)
            if view['state'] in ('done', 'error'):
                return view
            time.sleep(0.05)
        self.fail('job did not finish')

    def test_zip_upload_runs_and_serves_image_and_report(self):
        good = synthetic_dicom(Path(self.tmp.name) / 'a.dcm').read_bytes()
        archive = zip_bytes([('study/a.dcm', good), ('study/broken.dcm', b'not a dicom')])
        _, _, body = self.call('POST', '/api/jobs')
        job = json.loads(body)['id']
        self.call('PUT', f'/api/jobs/{job}/files?name=batch.zip', archive)
        self.call('POST', f'/api/jobs/{job}/start')
        view = self.wait_done(job)
        self.assertEqual(view['state'], 'done')
        self.assertEqual(view['total'], 2)
        self.assertEqual(sorted(i['status'] for i in view['items']), ['Failure', 'Success'])
        ok = next(i for i in view['items'] if i['status'] == 'Success')
        status, headers, png = self.call('GET', f"/api/jobs/{job}/image/{ok['i']}")
        self.assertEqual(headers['Content-Type'], 'image/png')
        self.assertTrue(png.startswith(b'\x89PNG'))
        _, headers, csv = self.call('GET', f'/api/jobs/{job}/report.csv')
        self.assertIn('attachment', headers['Content-Disposition'])
        header = csv.decode('utf-8').splitlines()[0]
        self.assertEqual(header.split(','), ['path_to_study', 'study_uid', 'image_uid', 'anatomical_region',
                                             'quality_class', 'violation_type', 'processing_status',
                                             'time_of_processing', 'quality_prob'])
        _, headers, sr = self.call('GET', f'/api/jobs/{job}/sr.zip')
        self.assertIn('_SR.zip', headers['Content-Disposition'])
        import io, zipfile
        self.assertEqual(len(zipfile.ZipFile(io.BytesIO(sr)).namelist()), 1)   # the broken file has no SR
        # incremental polling returns only the new items
        _, _, body = self.call('GET', f'/api/jobs/{job}?since=2')
        self.assertEqual(json.loads(body)['items'], [])

    def test_loose_files_and_unsafe_names(self):
        good = synthetic_dicom(Path(self.tmp.name) / 'a.dcm').read_bytes()
        _, _, body = self.call('POST', '/api/jobs')
        job = json.loads(body)['id']
        self.call('PUT', f'/api/jobs/{job}/files?name=folder%2Fimg1.dcm', good)
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.call('PUT', f'/api/jobs/{job}/files?name=..%2Fescape.dcm', good)
        self.assertEqual(err.exception.code, 400)
        self.call('POST', f'/api/jobs/{job}/start')
        view = self.wait_done(job)
        self.assertEqual([i['path'] for i in view['items']], ['folder/img1.dcm'])

    def test_specialist_correction_rescores_confirms_and_reaches_the_report(self):
        self.server.shutdown(); self.server.server_close()
        self.store = JobStore(PointAwarePredictor())
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.store.predictor, self.store))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        good = synthetic_dicom(Path(self.tmp.name) / 'a.dcm').read_bytes()
        _, _, body = self.call('POST', '/api/jobs')
        job = json.loads(body)['id']
        self.call('PUT', f'/api/jobs/{job}/files?name=hip.dcm', good)
        self.call('POST', f'/api/jobs/{job}/start')
        view = self.wait_done(job)
        item = view['items'][0]
        self.assertEqual(item['violations'], ['hip_roi_margins'])       # model: ROI top too close
        moved = dict(item['points'], roi_top=[20, 8])
        _, _, body = self.call('POST', f'/api/jobs/{job}/items/0/rescore', json.dumps({'points': moved}).encode())
        self.assertEqual(json.loads(body)['violations'], [])            # preview only
        self.assertEqual(json.loads(self.call('GET', f'/api/jobs/{job}')[2])['items'][0]['violations'], ['hip_roi_margins'])
        # confirm the moved points, and mark rotation as violated by hand
        payload = {'points': moved, 'overrides': {'hip_positioning': True}}
        _, _, body = self.call('POST', f'/api/jobs/{job}/items/0/review', json.dumps(payload).encode())
        final = json.loads(body)
        self.assertEqual((final['quality'], final['violations']), (1, ['hip_positioning']))
        self.assertEqual(final['review']['model']['violations'], ['hip_roi_margins'])
        csv = self.call('GET', f'/api/jobs/{job}/report.csv')[2].decode('utf-8').splitlines()[1].split(',')
        self.assertEqual((csv[3], csv[4], csv[5]), ('Проксимальный отдел бедра', '1', 'Некорректная укладка'))
        corr = json.loads(self.call('GET', f'/api/jobs/{job}/corrections.json')[2])
        self.assertEqual(len(corr['reviews']), 1)
        self.assertEqual(corr['reviews'][0]['points']['roi_top'], [20.0, 8.0])
        import io, zipfile, pydicom
        zf = zipfile.ZipFile(io.BytesIO(self.call('GET', f'/api/jobs/{job}/sr.zip')[2]))
        sr = pydicom.dcmread(io.BytesIO(zf.read(zf.namelist()[0])))
        content = {i.ConceptNameCodeSequence[0].CodeValue: i.get('TextValue') for i in sr.ContentSequence}
        self.assertEqual(sr.VerificationFlag, 'VERIFIED')
        self.assertEqual(content['VIOLATIONS'], 'Некорректная укладка')
        self.assertIn('Некорректная область интереса', content['MODEL_VERDICT'])
        self.assertIn('исправлены', content['REVIEW'])
        with self.assertRaises(urllib.error.HTTPError) as err:          # unknown criterion for a hip
            self.call('POST', f'/api/jobs/{job}/items/0/review', json.dumps({'overrides': {'spine_artifact': True}}).encode())
        self.assertEqual(err.exception.code, 400)

    def test_report_waits_for_the_batch_and_unknown_job_is_404(self):
        _, _, body = self.call('POST', '/api/jobs')
        job = json.loads(body)['id']
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.call('GET', f'/api/jobs/{job}/report.csv')
        self.assertEqual(err.exception.code, 409)
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.call('GET', '/api/jobs/nope')
        self.assertEqual(err.exception.code, 404)


class ArtifactBoxes(unittest.TestCase):
    def test_detector_boxes_reach_the_page_in_image_pixels(self):
        from dxaqc.webui import _item
        found = {'wires': [{'bbox': [3, 4, 60, 18], 'length_px': 65.5}],
                 'metal': [{'bbox': [146, 177, 4, 3], 'area_px': 6}]}
        pred = Prediction(1, ['spine_artifact'], {'quality': 0.9}, {'landmarks': {'artifacts': found}})
        item = _item(0, {'path_to_study': 'a.dcm', 'processing_status': 'Success'}, None, 'lumbar_spine', pred)
        self.assertEqual(item['artifacts'], [{'type': 'wire', 'x': 3, 'y': 4, 'w': 60, 'h': 18},
                                             {'type': 'metal', 'x': 146, 'y': 177, 'w': 4, 'h': 3}])

    def test_no_detector_output_gives_an_empty_list(self):
        from dxaqc.webui import _item
        item = _item(0, {'path_to_study': 'a.dcm', 'processing_status': 'Success'}, None, 'hip_left', Prediction(0))
        self.assertEqual(item['artifacts'], [])


if __name__ == '__main__':
    unittest.main()
