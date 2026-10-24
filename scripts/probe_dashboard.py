"""Local monitoring UI: bounded latest-frame buffer, explicit scripted playback."""

import base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import queue
import threading
import time

import mujoco
from PIL import Image

from arm_assembly import diagnostics, set_state
from probe_validation import graphics_info, make_camera


class Timeline:
    def __init__(self, frame_count):
        self.frame_count = frame_count
        self.step = 0
        self.cycle = 0
        self.paused = False
        self.speed = 1.0

    def advance(self):
        self.step += 1
        if self.step == self.frame_count:
            self.step = 0
            self.cycle += 1

    def apply(self, command):
        action = command.get('action')
        if action == 'play':
            self.paused = False
        elif action == 'pause':
            self.paused = True
        elif action == 'step':
            self.paused = True
            self.advance()
        elif action == 'restart':
            self.paused = True
            self.step = 0
            self.cycle = 0
        elif action == 'speed':
            speed = command.get('value')
            if type(speed) not in (int, float) or speed not in (.25, .5, 1, 2):
                raise ValueError('Speed must be 0.25, 0.5, 1, or 2')
            self.speed = float(speed)
        else:
            raise ValueError('Unknown playback action')


class Dashboard:
    def __init__(self, port):
        self.lock = threading.Lock()
        self.commands = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.snapshot = {'status': 'starting', 'heartbeat': time.time(), 'image': None}
        owner = self
        page = (Path(__file__).resolve().parents[1] / 'ui/probe.html').read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, payload, content_type='application/json'):
                if not isinstance(payload, bytes):
                    payload = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def allowed_host(self):
                port = owner.server.server_address[1]
                return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

            def do_GET(self):
                if not self.allowed_host():
                    return self.send(403, {'error': 'Use the localhost address'})
                if self.path == '/':
                    return self.send(200, page, 'text/html; charset=utf-8')
                if self.path == '/api/frame':
                    with owner.lock:
                        current = owner.snapshot.copy()
                    return self.send(200, current)
                self.send(404, {'error': 'Not found'})

            def do_POST(self):
                port = owner.server.server_address[1]
                origin = self.headers.get('Origin')
                if not self.allowed_host() or (origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}')):
                    return self.send(403, {'error': 'Local origin required'})
                if self.path != '/api/control':
                    return self.send(404, {'error': 'Not found'})
                if self.headers.get('Content-Type') != 'application/json':
                    return self.send(415, {'error': 'Expected application/json'})
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 512:
                        raise ValueError('Invalid command length')
                    command = json.loads(self.rfile.read(length))
                    if not isinstance(command, dict):
                        raise ValueError('Command must be an object')
                    Timeline(2).apply(command)  # Validate before enqueueing.
                    owner.commands.put_nowait(command)
                except (ValueError, UnicodeError) as error:
                    return self.send(400, {'error': str(error)})
                except queue.Full:
                    return self.send(429, {'error': 'Command queue full; retry shortly'})
                self.send(202, {'accepted': True})

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def run(self, model, config, frames, stages, replay=None):
        timeline = Timeline(len(frames))
        if replay is not None:
            timeline.speed = .25
        data = mujoco.MjData(model)
        camera = make_camera(config)
        events = deque([f"Recorded-source replay: {replay['recording']}" if replay else
                        'Synthetic assembly probe started; source-recording replay is not loaded.'], maxlen=20)
        last_signature = None
        image = None
        sequence = 0
        deadline = time.monotonic() + 1 / config['sample_rate_hz']
        performance = deque(maxlen=60)
        with mujoco.Renderer(model, config['render_height'], config['render_width']) as renderer:
            graphics = graphics_info()
            if config['expected_renderer'] not in graphics['renderer']:
                raise RuntimeError(f"Expected {config['expected_renderer']}; got {graphics['renderer']}")
            while not self.stop.is_set():
                changed = False
                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    timeline.apply(command)
                    changed = True
                    events.append(f"{command['action']}" + (f" {command['value']}x" if command['action'] == 'speed' else ''))
                now = time.monotonic()
                if changed:
                    deadline = now + 1 / (config['sample_rate_hz'] * timeline.speed)
                elif not timeline.paused and now >= deadline:
                    timeline.advance()
                    deadline = now + 1 / (config['sample_rate_hz'] * timeline.speed)
                signature = (timeline.cycle, timeline.step)
                if signature != last_signature:
                    start = time.perf_counter()
                    set_state(model, data, frames[timeline.step], timeline.step, config['sample_rate_hz'])
                    metrics = diagnostics(model, data)
                    renderer.update_scene(data, camera=camera)
                    buffer = io.BytesIO()
                    Image.fromarray(renderer.render()).save(buffer, format='JPEG', quality=88)
                    image = base64.b64encode(buffer.getvalue()).decode()
                    performance.append(time.perf_counter() - start)
                    last_signature = signature
                    sequence += 1
                warning = None
                if metrics['max_penetration_m'] > config['penetration_tolerance_m']:
                    warning = 'Collision penetration exceeds the assembly probe tolerance.'
                current = {
                    'status': 'paused' if timeline.paused else 'live',
                    'mode': 'recorded source replay' if replay else 'synthetic kinematic assembly probe',
                    'stage': stages[timeline.step], 'step': timeline.step, 'cycle': timeline.cycle,
                    'steps': len(frames), 'time': timeline.step / config['sample_rate_hz'],
                    'source_time': float(replay['source_timestamps'][timeline.step]) if replay else None,
                    'source_frame_id': int(replay['source_frame_ids'][timeline.step]) if replay else None,
                    'source_recording': replay['recording'] if replay else None,
                    'reference_image': replay['reference_images'][timeline.step] if replay else None,
                    'speed': timeline.speed, 'sequence': sequence,
                    'heartbeat': time.time(), 'image': image, 'pov_available': False,
                    'recording': False, 'metrics': metrics,
                    'source_tracking_metrics': replay['metrics'][timeline.step] if replay else None,
                    'render_capacity_fps': len(performance) / sum(performance) if performance else None,
                    'renderer': graphics['renderer'], 'events': list(events), 'warning': warning,
                }
                with self.lock:
                    self.snapshot = current
                self.stop.wait(.005)
