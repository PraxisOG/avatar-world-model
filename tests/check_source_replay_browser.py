"""Verify recorded-source synchronization and playback in the running dashboard."""
import hashlib
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8765'


def main():
    trajectory = ROOT / 'artifacts/source-replay/replay-trajectory.npz'
    before = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(URL)
        page.wait_for_function("document.querySelector('#reference').naturalWidth > 0")

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
        page.get_by_label('Speed', exact=True).select_option('0.5')
        assert wait_until(lambda s: s['speed'] == .5)['step'] == paused['step']
        page.get_by_role('button', name='Restart', exact=True).click()
        first = wait_until(lambda s: s['step'] == 0 and s['status'] == 'paused')
        assert first['source_frame_id'] == 12 and first['source_time'] == .4
        page.get_by_role('button', name='Single step', exact=True).click()
        step = wait_until(lambda s: s['step'] == 1)
        assert step['source_frame_id'] == 13 and abs(step['source_time'] - 13/30) < 1e-12
        assert step['source_tracking_metrics']['source_frame_id'] == 13
        page.wait_for_function("document.querySelector('#timestamp').textContent.includes('Source frame 13')")
        pair = page.evaluate("[document.querySelector('#external').src, document.querySelector('#reference').src]")
        assert pair == ['data:image/jpeg;base64,' + step[key] for key in ('image', 'reference_image')]
        assert page.get_by_text('Recorded source POV', exact=True).is_visible()
        assert page.get_by_text('Simulated head POV is unavailable.', exact=False).is_visible()
        assert not page.get_by_text('The head camera comes next', exact=True).is_visible()
        page.get_by_role('button', name='Play', exact=True).click()
        moving = wait_until(lambda s: s['step'] >= 10)
        assert moving['image'] != first['image'] and moving['reference_image'] != first['reference_image']
        page.get_by_role('button', name='Pause', exact=True).click()
        wait_until(lambda s: s['status'] == 'paused')
        page.wait_for_timeout(250)
        page.screenshot(path=str(ROOT / 'artifacts/source-replay/dashboard.png'), full_page=True)
        page.get_by_label('Speed', exact=True).select_option('0.25')
        wait_until(lambda s: s['speed'] == .25)
        page.get_by_role('button', name='Play', exact=True).click()
        active = wait_until(lambda s: s['status'] == 'live')
        page.close()
        request = browser.new_context().request
        time.sleep(.3)
        assert request.get(URL + '/api/frame').json()['sequence'] > active['sequence']
        assert not errors, errors
        browser.close()
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == before
    print(json.dumps({'passed': True, 'checks': ['original PTS and source frame indices',
        'synchronized decoded robot/reference images', 'pause/step/restart/speed',
        'both images change during playback', 'head POV explicitly unavailable',
        'saved trajectory unchanged', 'browser close leaves producer running', 'no JavaScript errors']}, indent=2))


if __name__ == '__main__':
    main()
