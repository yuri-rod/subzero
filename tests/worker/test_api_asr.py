import pytest
from fastapi.testclient import TestClient

from subzero.worker.api import create_app
from subzero.worker.config import Config
from subzero.worker.jellyfin import JellyfinError, Media
from subzero.worker.srt import Cue

AUTH = {"Authorization": "Bearer segredo"}


class FakeJellyfin:
    def media(self, item_id):
        raise JellyfinError("nao usado aqui")

    def refresh(self, item_id):
        pass

    def all_items(self):
        return []


def build(tmp_path, compat=True, max_mb=1024):
    cfg = Config.load({"JELLYFIN_URL": "http://jf", "JELLYFIN_API_KEY": "k",
                       "BEARER_TOKEN": "segredo", "DB_PATH": str(tmp_path / "jobs.db"),
                       "ASR_COMPAT": "1" if compat else "0", "ASR_MAX_MB": str(max_mb)})
    return TestClient(create_app(cfg, runner=False, jellyfin=FakeJellyfin(), opensubs=object()))


@pytest.fixture
def fake_transcribe(monkeypatch):
    seen = {}

    def fake(path, holder, progress):
        seen["bytes"] = open(path, "rb").read()
        return [Cue(1, 0.0, 1.5, "hello there")], "en"

    monkeypatch.setattr("subzero.worker.api.transcribe", fake)
    return seen


def test_asr_is_absent_unless_the_compat_flag_is_on(tmp_path):
    client = build(tmp_path, compat=False)

    assert client.post("/asr", headers=AUTH, files={"audio_file": ("a.wav", b"x")}).status_code == 404


def test_asr_still_needs_the_bearer(tmp_path, fake_transcribe):
    client = build(tmp_path)

    assert client.post("/asr", files={"audio_file": ("a.wav", b"x")}).status_code == 401


def test_asr_returns_srt_bazarr_can_read(tmp_path, fake_transcribe):
    client = build(tmp_path)

    r = client.post("/asr?task=transcribe&language=en&output=srt", headers=AUTH,
                    files={"audio_file": ("a.wav", b"RIFFfake")})

    assert r.status_code == 200
    assert "00:00:00,000 --> 00:00:01,500" in r.text
    assert "hello there" in r.text
    assert fake_transcribe["bytes"] == b"RIFFfake"


def test_asr_can_return_plain_text(tmp_path, fake_transcribe):
    client = build(tmp_path)

    r = client.post("/asr?output=txt", headers=AUTH, files={"audio_file": ("a.wav", b"x")})

    assert r.text.strip() == "hello there"
    assert "-->" not in r.text


def test_detect_language_answers_the_shape_bazarr_expects(tmp_path, fake_transcribe):
    client = build(tmp_path)

    body = client.post("/detect-language", headers=AUTH,
                       files={"audio_file": ("a.wav", b"x")}).json()

    assert body == {"detected_language": "en", "language_code": "en"}


def test_asr_refuses_an_upload_over_the_cap(tmp_path, fake_transcribe):
    client = build(tmp_path, max_mb=1)

    r = client.post("/asr", headers=AUTH, files={"audio_file": ("a.wav", b"x" * (2 * 1024 * 1024))})

    assert r.status_code == 413


def test_asr_refuses_an_empty_upload(tmp_path, fake_transcribe):
    client = build(tmp_path)

    assert client.post("/asr", headers=AUTH,
                       files={"audio_file": ("a.wav", b"")}).status_code == 400
