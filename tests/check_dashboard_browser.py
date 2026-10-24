"""Exercise the running probe in Chromium; requires the optional Playwright tool."""

import hashlib
import json
import os
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get('PROBE_URL', 'http://127.0.0.1:8765')


def main():
    trajectory = ROOT / 'artifacts/arm-probe/trajectory.npz'
    before = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(URL)
        page.wait_for_function("document.querySelector('#external').naturalWidth > 0")

        def state():
            return page.request.get(URL + '/api/frame').json()

        def wait_until(predicate):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = state()
                if predicate(current):
                    return current
                page.wait_for_timeout(30)
            raise AssertionError('Playback state did not reach expected condition')

        page.get_by_role('button', name='Pause', exact=True).click()
        paused = wait_until(lambda s: s['status'] == 'paused')
        page.wait_for_timeout(250)
        assert state()['step'] == paused['step']
        page.get_by_label('Speed', exact=True).select_option('2')
        speed = wait_until(lambda s: s['speed'] == 2)
        assert speed['step'] == paused['step']
        page.get_by_role('button', name='Restart', exact=True).click()
        restarted = wait_until(lambda s: s['step'] == 0 and s['status'] == 'paused')
        assert restarted['time'] == 0
        page.get_by_role('button', name='Single step', exact=True).click()
        stepped = wait_until(lambda s: s['step'] == 1)
        assert stepped['status'] == 'paused' and abs(stepped['time'] - 1 / 30) < 1e-12
        page.get_by_role('button', name='Play', exact=True).click()
        moving = wait_until(lambda s: s['step'] > 35)
        assert moving['image'] != restarted['image']
        page.get_by_role('button', name='Pause', exact=True).click()
        wait_until(lambda s: s['status'] == 'paused')
        page.wait_for_timeout(200)
        assert page.get_by_text('The head camera comes next').is_visible()
        assert page.get_by_text('Dataset recording: off', exact=False).is_visible()
        page.screenshot(path=str(ROOT / 'artifacts/arm-probe/dashboard.png'), full_page=True)
        page.route('**/api/frame', lambda route: route.abort())
        page.wait_for_function("document.querySelector('#status').textContent.includes('Disconnected')")
        assert page.get_by_role('button', name='Play', exact=True).is_disabled()
        page.unroute('**/api/frame')
        page.wait_for_function("document.querySelector('#status').textContent === 'Paused'")
        page.get_by_label('Speed', exact=True).select_option('1')
        wait_until(lambda s: s['speed'] == 1)
        page.get_by_role('button', name='Play', exact=True).click()
        active = wait_until(lambda s: s['status'] == 'live')
        page.close()
        # Closing the only viewer must not stop the simulation producer.
        request = browser.new_context().request
        time.sleep(.3)
        remaining = request.get(URL + '/api/frame').json()
        assert remaining['sequence'] > active['sequence']
        assert not errors, errors
        browser.close()
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == before
    print(json.dumps({'passed': True, 'browser': 'system Chromium via Playwright',
        'checks': ['live image changes', 'pause stable', 'single-step timestamp',
                   'restart zero', 'speed preserves sample index', 'head POV unavailable label',
                   'stale/disconnected state and recovery', 'browser close leaves simulation running',
                   'saved trajectory unchanged', 'no JavaScript errors']}, indent=2))


if __name__ == '__main__':
    main()
