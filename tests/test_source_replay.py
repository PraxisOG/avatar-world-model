"""Integration guards using the pinned local source; no downloads during tests."""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from egodex_source import load_source
from fetch_egodex_sample import RangeFile
from source_replay import load_replay

SELECTION = json.loads((ROOT / 'config/source_replay.json').read_text())
SOURCE = ROOT / SELECTION['source_hdf5']


@unittest.skipUnless(SOURCE.exists(), 'Prepare the local source with ./run-replay --prepare')
class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selection = copy.deepcopy(SELECTION)
        for key in ('source_hdf5', 'source_mp4'):
            source = ROOT / SELECTION[key]
            target = self.root / source.name
            shutil.copyfile(source, target)
            shutil.copyfile(str(source) + '.provenance.json', str(target) + '.provenance.json')
            self.selection[key] = target.name

    def load(self):
        return load_source(self.root, self.selection, self.root / 'out')

    def repin_modified_fixture(self):
        path = self.root / self.selection['source_hdf5']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = Path(str(path) + '.provenance.json')
        record = json.loads(sidecar.read_text())
        record['sha256'] = digest
        sidecar.write_text(json.dumps(record))
        self.selection['expected_source_sha256']['hdf5'] = digest

    def test_preserves_source_timing_and_unused_tracking(self):
        source, confidence, times, images, report = self.load()
        np.testing.assert_allclose(times, np.arange(12, 32) / 30, atol=1e-12)
        self.assertEqual(len(images), 20)
        self.assertFalse(report['full_phase1_gate_passed'])
        self.assertEqual(report['session_id'], report['split_group'])
        with h5py.File(SOURCE) as original, np.load(self.root / 'out/original-tracking.npz') as saved:
            for name in original['transforms']:
                np.testing.assert_array_equal(saved['transform_' + name], original['transforms'][name][12:32])
            for name in original['confidences']:
                np.testing.assert_array_equal(confidence[name], original['confidences'][name][12:32])
            np.testing.assert_array_equal(saved['source_frame_ids'], np.arange(12, 32))
            np.testing.assert_allclose(saved['source_pts'] / 15360, times, atol=1e-12)
        self.assertIn('camera', source)
        self.assertIn('leftHand', source)
        self.assertEqual(confidence['rightForearm'].min(), 0)

    def test_rejects_nontraining_source_and_checksum_corruption(self):
        path = self.root / self.selection['source_hdf5']
        sidecar = Path(str(path) + '.provenance.json')
        record = json.loads(sidecar.read_text())
        record['official_split'] = 'test'
        sidecar.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'training-eligible'):
            self.load()
        record['official_split'] = 'train'
        sidecar.write_text(json.dumps(record))
        with path.open('ab') as handle:
            handle.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.load()

    def test_rejects_untracked_and_nonfinite_fingers(self):
        path = self.root / self.selection['source_hdf5']
        for confidence in (0., float('nan')):
            with self.subTest(confidence=confidence):
                with h5py.File(path, 'r+') as handle:
                    handle['confidences/rightIndexFingerTip'][12] = confidence
                self.repin_modified_fixture()
                with self.assertRaisesRegex(ValueError, 'source confidence'):
                    self.load()


@unittest.skipUnless((ROOT / 'artifacts/source-replay/replay-validation.json').exists(), 'Prepare replay first')
class ReplayTests(unittest.TestCase):
    def test_saved_replay_alignment_and_corruption_detection(self):
        output = ROOT / 'artifacts/source-replay'
        model, config, frames, replay = load_replay(output)
        self.assertEqual(frames.shape, (20, 29))
        np.testing.assert_array_equal(replay['source_frame_ids'], np.arange(12, 32))
        for index, row in enumerate(replay['metrics']):
            self.assertEqual(row['source_frame_id'], replay['source_frame_ids'][index])
            self.assertEqual(row['source_timestamp'], replay['source_timestamps'][index])
        # Guard the first-frame lower-bound optimizer stall and discontinuity regression.
        self.assertLess(max(replay['metrics'][0]['normalized_fingertip_errors_m']), .01)
        self.assertLess(np.max(np.abs(np.diff(frames, axis=0))), .1)
        self.assertTrue(np.all(frames >= model.jnt_range[:, 0] - 1e-9))
        self.assertTrue(np.all(frames <= model.jnt_range[:, 1] + 1e-9))
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            report = replay['report']
            for name in ('replay-validation.json', *report['artifact_sha256']):
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output / name, target / name)
            (target / 'replay-trajectory.npz').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'artifact checksum mismatch'):
                load_replay(target)


class RangeTests(unittest.TestCase):
    def test_exact_range_and_server_ignoring_range(self):
        class Handler(BaseHTTPRequestHandler):
            ignore_range = False
            def log_message(self, *args):
                pass
            def do_HEAD(self):
                self.send_response(200)
                self.send_header('Content-Length', '10')
                self.end_headers()
            def do_GET(self):
                start, end = map(int, self.headers['Range'].split('=')[1].split('-'))
                self.send_response(200 if self.ignore_range else 206)
                self.send_header('Content-Range', f'bytes {start}-{end}/10')
                self.end_headers()
                self.wfile.write(b'0123456789'[start:end+1])
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            remote = RangeFile(f'http://127.0.0.1:{server.server_port}', budget=5)
            remote.seek(3)
            self.assertEqual(remote.read(3), b'345')
            with self.assertRaisesRegex(ValueError, 'oversized'):
                remote.read(3)
            Handler.ignore_range = True
            with self.assertRaisesRegex(ValueError, 'ignored the exact byte range'):
                remote.read(1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
