from types import SimpleNamespace

import pytest

from subzero.worker.qbittorrent import QBitTorrentClient, QBitTorrentError, SeenStore


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers, params=None):
        self.calls.append((url, headers, params))
        return self.responses.pop(0)


def make_torrent(**overrides):
    torrent = {"hash": "a" * 40, "name": "movie", "progress": 1.0, "category": "movies",
               "save_path": "/Volumes/HD-3/FILMES", "content_path": "/Volumes/HD-3/FILMES/movie/movie.mkv",
               "completion_on": 1}
    torrent.update(overrides)
    return torrent


def test_client_uses_api_key_header():
    http = FakeHTTP([FakeResponse(200, [])])
    client = QBitTorrentClient("http://127.0.0.1:8585", "secret", http=http)
    assert client.torrents() == []
    assert http.calls[0][1]["X-API-Key"] == "secret"


def test_client_rejected_key_raises():
    http = FakeHTTP([FakeResponse(403, "")])
    client = QBitTorrentClient("http://127.0.0.1:8585", "bad", http=http)
    with pytest.raises(QBitTorrentError, match="rejected"):
        client.torrents()


def test_completed_filters_progress_and_category():
    torrents = [
        make_torrent(hash="1" * 40, progress=1.0, category="movies"),
        make_torrent(hash="2" * 40, progress=0.5, category="movies"),
        make_torrent(hash="3" * 40, progress=1.0, category="games"),
        make_torrent(hash="4" * 40, progress=1.0, category="tv shows"),
    ]
    http = FakeHTTP([FakeResponse(200, torrents)])
    client = QBitTorrentClient("http://x", "key", http=http)
    completed = client.completed({"movies", "tv shows"})
    assert {t["hash"] for t in completed} == {"1" * 40, "4" * 40}


def test_seen_store_roundtrip(tmp_path):
    path = tmp_path / "qbt-seen.json"
    store = SeenStore(str(path))
    assert "abc" not in store
    store.add("abc")
    store.save()

    reloaded = SeenStore(str(path))
    assert "abc" in reloaded


def test_seen_store_ignores_corrupt_state(tmp_path):
    path = tmp_path / "qbt-seen.json"
    path.write_text("{not json", encoding="utf-8")
    store = SeenStore(str(path))
    assert "abc" not in store
