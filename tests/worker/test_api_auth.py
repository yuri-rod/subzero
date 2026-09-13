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


def test_config_defaults():
    cfg = Config.load({"JELLYFIN_URL": "http://x", "JELLYFIN_API_KEY": "k", "BEARER_TOKEN": "t"})
    assert cfg.daily_download_budget == 15
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
