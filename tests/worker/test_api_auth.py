import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from subzero.worker.api import create_app
from subzero.worker.config import Config


@pytest.fixture
def client(tmp_path):
    cfg = Config.load({
        "JELLYFIN_URL": "http://127.0.0.1:8096",
        "JELLYFIN_API_KEY": "jf",
        "BEARER_TOKEN": "segredo",
        "OPENSUBTITLES_API_KEY": "os",
        "DB_PATH": str(tmp_path / "jobs.db"),
    })
    return TestClient(create_app(cfg, runner=False))


def test_health_needs_a_token(client):
    assert client.get("/health").status_code == 401


def test_resume_requires_auth_and_only_accepts_paused_job(client):
    store = client.app.state.store
    job = store.enqueue('episode', 'repair', 'pt-BR')
    headers = {'Authorization': 'Bearer segredo'}
    path = f'/jobs/{job.id}/resume'
    assert client.post(path).status_code == 401
    assert client.post(path, headers=headers).status_code == 409
    store.start(job.id)
    store.pause(job.id, 'DeepL Free quota')
    assert client.get('/health', headers=headers).json()['paused'] == 1
    response = client.post(path, headers=headers)
    assert response.status_code == 200
    assert response.json()['state'] == 'queued'
    assert client.post('/jobs/missing/resume', headers=headers).status_code == 404


def test_health_rejects_the_wrong_token(client):
    assert client.get("/health", headers={"Authorization": "Bearer errado"}).status_code == 401


def test_health_reports_the_worker(client):
    r = client.get("/health", headers={"Authorization": "Bearer segredo"})
    assert r.status_code == 200
    body = r.json()
    assert body["version"]
    assert "gpu" in body
    assert body["model"] == "large-v3"
    assert body["translation_model"] == "subzero/hy-mt2:7b"
    assert body['translation_provider'] == 'ollama'


def test_cpu_provider_selection_and_health_do_not_start_models(tmp_path, monkeypatch):
    from subzero.worker.libretranslate import LibreTranslate

    def no_model(*args, **kwargs):
        raise AssertionError('App startup and health must not start translation models')

    monkeypatch.setattr('subzero.worker.api.Ollama', no_model)
    monkeypatch.setattr('subzero.worker.api.resolve_deepl_key', no_model)
    monkeypatch.setattr(LibreTranslate, 'ensure_available', no_model)
    cfg = Config('http://localhost', 'key', 'token', db_path=str(tmp_path / 'jobs.db'),
                 sync_cache=str(tmp_path / 'cache'), translation_provider='libretranslate',
                 libretranslate_runtime=str(tmp_path / 'runtime'))
    opensubs = type('OpenSubs', (), {'login': lambda self: None})()
    client = TestClient(create_app(cfg, runner=False, opensubs=opensubs))
    response = client.get('/health', headers={'Authorization': 'Bearer token'})
    assert response.status_code == 200
    assert response.json()['translation_provider'] == 'libretranslate'
    assert response.json()['translation_model'] == LibreTranslate.model


def test_deepl_selection_and_health_use_only_resolved_credentials(tmp_path, monkeypatch):
    resolved = []
    constructed = []
    key = 'resolved-free-test-key:fx'

    def no_model(*args, **kwargs):
        raise AssertionError('DeepL startup and health must not run translation or load local models')

    def key_from_config(cfg):
        resolved.append(cfg)
        return key

    class DeepLFree:
        provider = 'deepl-free'
        model = 'deepl-free/en:pt-BR'

        def __init__(self, api_key, *, usage_path):
            constructed.append((api_key, usage_path))

        ensure_available = no_model
        translate_block = no_model

    module = ModuleType('subzero.worker.deepl')
    module.DeepLFree = DeepLFree
    monkeypatch.setitem(sys.modules, 'subzero.worker.deepl', module)
    monkeypatch.setattr('subzero.worker.api.Ollama', no_model)
    monkeypatch.setattr('subzero.worker.api.resolve_deepl_key', key_from_config)
    cfg = Config('http://localhost', 'key', 'token', db_path=str(tmp_path / 'jobs.db'),
                 sync_cache=str(tmp_path / 'cache'), translation_provider='deepl-free')
    opensubs = type('OpenSubs', (), {'login': lambda self: None})()
    app = create_app(cfg, runner=False, opensubs=opensubs)
    response = TestClient(app).get('/health', headers={'Authorization': 'Bearer token'})
    assert response.status_code == 200
    assert resolved == [cfg]
    assert constructed == [(key, tmp_path / 'cache' / 'deepl-free-usage.json')]
    assert isinstance(app.state.service.ollama, DeepLFree)
    assert response.json()['translation_provider'] == 'deepl-free'
    assert response.json()['translation_model'] == DeepLFree.model
    assert key not in response.text


def test_deepl_real_client_startup_does_not_request_usage_or_translation(tmp_path, monkeypatch):
    from subzero.worker.deepl import DeepLFree

    def no_request(*args, **kwargs):
        raise AssertionError('Provider HTTP requests do not belong in startup or health')

    monkeypatch.setattr(DeepLFree, '_request', no_request)
    cfg = Config('http://localhost', 'key', 'token', db_path=str(tmp_path / 'jobs.db'),
                 sync_cache=str(tmp_path / 'cache'), translation_provider='deepl-free',
                 deepl_api_key='explicit-free-test-key:fx')
    opensubs = type('OpenSubs', (), {'login': lambda self: None})()
    app = create_app(cfg, runner=False, opensubs=opensubs)
    response = TestClient(app).get('/health', headers={'Authorization': 'Bearer token'})
    assert response.status_code == 200
    assert response.json()['translation_model'] == DeepLFree.model
    translator = app.state.service.ollama
    assert isinstance(translator, DeepLFree)
    assert translator.usage_path == tmp_path / 'cache' / 'deepl-free-usage.json'
    assert not translator.usage_path.exists()
    assert cfg.deepl_api_key not in response.text
    assert cfg.deepl_api_key not in str(translator.cache_settings)


def test_config_defaults():
    cfg = Config.load({"JELLYFIN_URL": "http://x", "JELLYFIN_API_KEY": "k", "BEARER_TOKEN": "t"})
    assert cfg.auto_langs == ["pt-BR"]
    assert cfg.ollama_model == "subzero/hy-mt2:7b"


def test_config_demands_the_essentials():
    with pytest.raises(ValueError):
        Config.load({"JELLYFIN_URL": "http://x"})


def test_shutdown_endpoint(client):
    mock_server = type("MockServer", (), {"should_exit": False})()
    client.app.state.server = mock_server
    assert client.post("/shutdown").status_code == 401
    r = client.post("/shutdown", headers={"Authorization": "Bearer segredo"})
    assert r.status_code == 200
    assert r.json() == {"status": "shutting_down"}
    import time
    time.sleep(0.6)
    assert mock_server.should_exit is True
