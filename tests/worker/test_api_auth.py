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


def test_config_defaults():
    cfg = Config.load({"JELLYFIN_URL": "http://x", "JELLYFIN_API_KEY": "k", "BEARER_TOKEN": "t"})
    assert cfg.daily_download_budget == 15
    assert cfg.auto_langs == ["pt-BR"]
    assert cfg.ollama_model == "gemma3:12b"


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

