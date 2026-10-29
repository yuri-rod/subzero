import json
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient

from subzero.worker.api import create_app
from subzero.worker.config import Config
from subzero.worker.jobs import KINDS, Runner


@pytest.mark.parametrize('provider', ['ollama', 'openai'])
def test_configured_provider_completes_a_persistent_translation_job(tmp_path, monkeypatch, provider):
    video = tmp_path / 'episode.mkv'
    video.write_bytes(b'video fixture')
    video.with_suffix('.en.srt').write_text('1\n00:00:01,000 --> 00:00:02,000\nHello.\n')
    requests = []

    def http_request(client, method, url, **kwargs):
        requests.append((method, url))
        if url == 'http://jellyfin/Items/episode/PlaybackInfo':
            body = {'MediaSources': [{'Id': 'source', 'Name': 'Episode', 'Path': str(video),
                                     'RunTimeTicks': 30_000_000, 'MediaStreams': []}]}
        elif url == 'http://jellyfin/Items':
            body = {'Items': [{'Id': 'episode', 'Name': 'Episode', 'Type': 'Movie'}]}
        elif url == 'http://jellyfin/Items/episode/Refresh':
            body = {}
        elif url == 'http://ollama/api/show':
            body = {}
        elif url == 'http://ollama/api/generate':
            assert provider == 'ollama'
            body = {'response': '[{"id":1,"text":"Hola."}]', 'done': True, 'done_reason': 'stop'}
        else:
            raise AssertionError(f'Unexpected HTTP request: {method} {url}')
        return httpx.Response(200, json=body)

    def openai_request(req, timeout):
        assert provider == 'openai'
        assert req.full_url == 'http://local-api/v1/chat/completions'
        assert req.get_header('Authorization') == 'Bearer provider-secret'
        assert timeout == 17
        assert json.loads(req.data)['model'] == 'local-translator'
        requests.append(('POST', req.full_url))
        response = BytesIO(json.dumps({'choices': [{'message': {'content': '1. Hola.'},
                                                   'finish_reason': 'stop'}]}).encode())
        response.status = 200
        return response

    monkeypatch.setattr(httpx.Client, 'request', http_request)
    monkeypatch.setattr('urllib.request.urlopen', openai_request)
    cfg = Config.load({'JELLYFIN_URL': 'http://jellyfin', 'JELLYFIN_API_KEY': 'jellyfin-key',
                       'BEARER_TOKEN': 'token', 'DB_PATH': str(tmp_path / 'jobs.db'),
                       'SYNC_CACHE': str(tmp_path / 'cache'), 'SEMANTIC_QA': '0',
                       'TRANSLATION_PROVIDER': provider, 'OLLAMA_URL': 'http://ollama',
                       'OLLAMA_MODEL': 'generic:7b', 'OPENAI_BASE_URL': 'http://local-api/v1',
                       'OPENAI_MODEL': 'local-translator', 'OPENAI_API_KEY': 'provider-secret',
                       'OPENAI_TIMEOUT': '17'})
    app = create_app(cfg, runner=False)
    client = TestClient(app)
    response = client.post('/jobs', headers={'Authorization': 'Bearer token'},
                           json={'itemId': 'episode', 'kind': 'translate',
                                 'targetLang': 'es', 'sourceId': 'en'})
    assert response.status_code == 200
    job_id = response.json()['id']
    finished = Runner(app.state.store, {kind: app.state.service.run for kind in KINDS}).run_once()
    assert finished.id == job_id
    assert finished.state == 'done', finished.message
    assert video.with_suffix('.es.srt').read_text() == '1\n00:00:01,000 --> 00:00:02,000\nHola.\n'
    assert app.state.store.get(job_id).result_path == str(video.with_suffix('.es.srt'))
    assert 'provider-secret' not in repr(cfg)
    assert 'provider-secret' not in client.get('/jobs', headers={'Authorization': 'Bearer token'}).text
    if provider == 'openai':
        assert all('ollama' not in url for _, url in requests)
        assert app.state.service.ollama.provider == 'openai'


def test_worker_registers_configured_pressure_guard_and_reports_it_until_shutdown(tmp_path, monkeypatch):
    from subzero import compute
    from subzero.worker.api import start_runner
    snapshot = {'timestamp': '2026-10-29T00:00:00Z', 'memory_total_bytes': 8192,
                'memory_available_bytes': 1024, 'swap_used_bytes': 0,
                'swap_total_bytes': 2048, 'vram_free_bytes': None}
    monkeypatch.setattr(compute, 'sample_resources', lambda: snapshot)
    monkeypatch.setattr('subzero.worker.api.free_vram_mb', lambda: None)
    previous = compute.PressureWatchdog(min_memory_available_bytes=8192, retry_seconds=13)
    monkeypatch.setattr(compute, '_pressure_watchdog', previous)
    cfg = Config.load({'JELLYFIN_URL': 'http://jellyfin', 'JELLYFIN_API_KEY': 'key',
                       'BEARER_TOKEN': 'token', 'DB_PATH': str(tmp_path / 'jobs.db'),
                       'SYNC_CACHE': str(tmp_path / 'cache'), 'AUTO_ENABLED': '0',
                       'IDLE_SHUTDOWN_MINUTES': '0', 'COMPUTE_MIN_MEMORY_AVAILABLE_BYTES': '4096',
                       'COMPUTE_PRESSURE_INTERVAL': '60', 'COMPUTE_PRESSURE_RETRY_SECONDS': '7'})
    app = create_app(cfg, runner=False)
    start_runner(app, app.state.store, app.state.service, None, cfg, app.state.service.jellyfin, None)
    try:
        body = TestClient(app).get('/health', headers={'Authorization': 'Bearer token'}).json()
        assert body['resourcePressure']['enabled'] is True
        assert body['resourcePressure']['blocked'] is True
        assert body['resourcePressure']['snapshot'] == snapshot
        with pytest.raises(compute.ComputePressureError) as error:
            compute.check_compute_pressure()
        assert error.value.retry_after == 7
    finally:
        for handler in app.router.on_shutdown:
            handler()
        app.state.runner_thread.join(timeout=2)
    with pytest.raises(compute.ComputePressureError) as restored:
        compute.check_compute_pressure()
    assert restored.value.retry_after == 13
    assert not app.state.runner_thread.is_alive()


def test_shutdown_keeps_pressure_guard_until_active_handler_exits(tmp_path, monkeypatch):
    from threading import Event, Thread
    from types import SimpleNamespace
    from subzero import compute
    from subzero.worker.api import start_runner

    available, clock = [8192], [0.0]
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': available[0]})
    monkeypatch.setattr(compute.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(compute, '_pressure_watchdog', None)
    cfg = Config('http://jellyfin', 'key', 'token', db_path=str(tmp_path / 'jobs.db'),
                 sync_cache=str(tmp_path / 'cache'), auto_enabled=False, idle_shutdown_minutes=0,
                 compute_min_memory_available_bytes=4096, compute_pressure_interval=1)
    app = create_app(cfg, runner=False)
    job = app.state.store.enqueue('episode', 'translate', 'es')
    entered, release, stopped = Event(), Event(), Event()
    computations = []

    def handle(job, progress):
        entered.set()
        assert release.wait(5)
        with compute.compute_phase('ollama'):
            computations.append('admitted')
        return None

    def shutdown():
        for handler in app.router.on_shutdown:
            handler()
        stopped.set()

    start_runner(app, app.state.store, SimpleNamespace(run=handle), None, cfg, None, None)
    try:
        assert entered.wait(2)
        Thread(target=shutdown, daemon=True).start()
        assert stopped.wait(.5), 'Shutdown must not wait for the active handler'
        assert app.state.runner_thread.is_alive()
        available[0], clock[0] = 1024, 2.0
    finally:
        app.state.stop.set()
        release.set()
        app.state.runner_thread.join(timeout=2)
    assert not app.state.runner_thread.is_alive()
    assert computations == []
    deferred = app.state.store.get(job.id)
    assert (deferred.state, deferred.phase, deferred.attempts) == ('queued', 'resource_pressure', 0)
    compute.check_compute_pressure()


def test_openai_worker_cache_reuses_blocks_and_separates_endpoints(tmp_path, monkeypatch):
    from subzero.worker.srt import Cue
    replies = {'http://first/v1/chat/completions': 'Hola.',
               'http://second/v1/chat/completions': 'Buenos días.'}

    def request(req, timeout):
        text = replies.pop(req.full_url)
        response = BytesIO(json.dumps({'choices': [{'message': {'content': f'1. {text}'}}]}).encode())
        response.status = 200
        return response

    monkeypatch.setattr('urllib.request.urlopen', request)
    cfg = Config('http://jellyfin', 'key', 'token', translation_provider='openai',
                 db_path=str(tmp_path / 'jobs.db'), sync_cache=str(tmp_path / 'cache'),
                 openai_base_url='http://first/v1', openai_api_key='provider-secret')
    app = create_app(cfg, runner=False)
    job = app.state.store.enqueue('episode', 'embedded_translate', 'es')
    app.state.store.start(job.id)
    flow = app.state.service.sync_flow
    source = [Cue(1, 1.0, 2.0, 'Hello.')]
    first = flow._translate_cues_cached(job, 'video-key', source, 'es', lambda *_: None, 'en')
    reused = flow._translate_cues_cached(job, 'video-key', source, 'es', lambda *_: None, 'en')
    assert first == reused
    assert first[0].text == 'Hola.'
    app.state.service.ollama.base_url = 'http://second/v1'
    second = flow._translate_cues_cached(job, 'video-key', source, 'es', lambda *_: None, 'en')
    assert second[0].text == 'Buenos días.'
    caches = list((tmp_path / 'cache' / 'translations').rglob('*.json'))
    assert len(caches) == 2
    assert all('provider-secret' not in path.read_text() for path in caches)
