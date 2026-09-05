import pytest

from subzero.worker.jellyfin import EmbeddedSub, Media
from subzero.worker.jobs import Job
from subzero.worker.service import Service, three_letter


def media(**kw):
    base = dict(item_id="abc", name="Filme", path="/media/Filme.mkv", container="mkv",
                duration=120, audio_lang="eng", source_id="src",
                embedded=[EmbeddedSub(index=2, lang="eng", codec="subrip", title="English", sub_index=0)],
                sidecars=[])
    base.update(kw)
    return Media(**base)


class FakeJellyfin:
    def __init__(self, media):
        self._media = media
        self.refreshed = []

    def media(self, item_id):
        return self._media

    def refresh(self, item_id):
        self.refreshed.append(item_id)


class FakeOpenSubs:
    def __init__(self, text="1\n00:00:01,000 --> 00:00:02,000\nOla\n"):
        self.text = text
        self.downloaded = []

    def search(self, query=None, moviehash=None, langs=None):
        return []

    def download(self, file_id):
        self.downloaded.append(file_id)
        return self.text


def job(kind, target="pt-BR", source=None):
    return Job(id="j1", item_id="abc", kind=kind, target_lang=target, source_id=source,
               origin="manual", state="running", phase="", percent=0, message="",
               result_path=None, attempts=0, next_attempt=0, created=0, updated=0)


@pytest.fixture
def service(tmp_path):
    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    jf = FakeJellyfin(media(path=str(video)))
    svc = Service(jellyfin=jf, opensubs=FakeOpenSubs(), holder=None, ollama=None)
    return svc


def test_three_letter_maps_the_ui_codes():
    assert three_letter("pt-BR") == "por"
    assert three_letter("en") == "eng"
    assert three_letter("ja") == "jpn"


def test_embedded_handler_extracts_and_refreshes(service, tmp_path, monkeypatch):
    calls = {}

    def fake_extract(media, sub_index, lang, progress=None, bare=False):
        calls["sub_index"] = sub_index
        path = str(tmp_path / f"Filme.{lang}.srt")
        open(path, "w", encoding="utf-8").write("1\n00:00:01,000 --> 00:00:02,000\nOi\n")
        return path

    monkeypatch.setattr("subzero.worker.service.extract_embedded", fake_extract)

    out = service.run(job("embedded", target="en", source="2"), lambda p, n: None)

    assert calls["sub_index"] == 0
    assert out.endswith("Filme.en.srt")
    assert service.jellyfin.refreshed == ["abc"]


def test_embedded_handler_needs_a_stream(service):
    with pytest.raises(RuntimeError):
        service.run(job("embedded", source=None), lambda p, n: None)


def test_opensubtitles_handler_writes_the_downloaded_text(service, tmp_path):
    out = service.run(job("opensubtitles", target="pt-BR", source="99"), lambda p, n: None)

    assert service.opensubs.downloaded == [99]
    assert out.endswith("Filme.pt-BR.srt")
    assert "Ola" in open(out, encoding="utf-8").read()


def test_whisper_handler_translates_when_the_language_differs(service, tmp_path, monkeypatch):
    from subzero.worker.srt import Cue

    monkeypatch.setattr("subzero.worker.service.extract_audio", lambda path, duration=0, progress=None: "audio.wav")
    monkeypatch.setattr("subzero.worker.service.transcribe",
                        lambda audio, holder, progress: ([Cue(1, 0, 1, "hello")], "en"))
    monkeypatch.setattr("subzero.worker.service.translate",
                        lambda cues, lang, ollama, progress: [Cue(1, 0, 1, "ola")])

    out = service.run(job("whisper", target="pt-BR"), lambda p, n: None)

    assert "ola" in open(out, encoding="utf-8").read()


def test_whisper_handler_skips_translation_when_it_already_matches(service, monkeypatch):
    from subzero.worker.srt import Cue

    monkeypatch.setattr("subzero.worker.service.extract_audio", lambda path, duration=0, progress=None: "audio.wav")
    monkeypatch.setattr("subzero.worker.service.transcribe",
                        lambda audio, holder, progress: ([Cue(1, 0, 1, "hello")], "en"))

    def boom(*args, **kwargs):
        raise AssertionError("nao devia traduzir")

    monkeypatch.setattr("subzero.worker.service.translate", boom)

    out = service.run(job("whisper", target="en"), lambda p, n: None)
    assert "hello" in open(out, encoding="utf-8").read()


def test_translate_handler_reads_an_existing_sidecar(service, tmp_path, monkeypatch):
    from subzero.worker.srt import Cue

    source = tmp_path / "Filme.en.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    monkeypatch.setattr("subzero.worker.service.translate",
                        lambda cues, lang, ollama, progress: [Cue(1, 0, 1, "ola")])

    out = service.run(job("translate", target="pt-BR", source="en"), lambda p, n: None)

    assert out.endswith("Filme.pt-BR.srt")
    assert "ola" in open(out, encoding="utf-8").read()


def test_translate_handler_without_a_source_fails(service):
    with pytest.raises(RuntimeError):
        service.run(job("translate", target="pt-BR", source="fr"), lambda p, n: None)


def test_whisper_handler_pushes_the_cues_by_the_audio_offset(service, monkeypatch):
    from subzero.worker.srt import Cue

    monkeypatch.setattr("subzero.worker.service.audio_start_offset", lambda path: 1.4)
    monkeypatch.setattr("subzero.worker.service.extract_audio", lambda path, duration=0, progress=None: "audio.wav")
    monkeypatch.setattr("subzero.worker.service.transcribe",
                        lambda audio, holder, progress: ([Cue(1, 0.0, 1.0, "hello")], "en"))

    out = service.run(job("whisper", target="en"), lambda p, n: None)

    # sem o desvio a legenda entraria em 00:00:00,000 e ficaria 1,4s adiantada
    assert "00:00:01,400 --> 00:00:02,400" in open(out, encoding="utf-8").read()


def test_whisper_handler_leaves_timings_alone_without_an_offset(service, monkeypatch):
    from subzero.worker.srt import Cue

    monkeypatch.setattr("subzero.worker.service.audio_start_offset", lambda path: 0.0)
    monkeypatch.setattr("subzero.worker.service.extract_audio", lambda path, duration=0, progress=None: "audio.wav")
    monkeypatch.setattr("subzero.worker.service.transcribe",
                        lambda audio, holder, progress: ([Cue(1, 0.0, 1.0, "hello")], "en"))

    out = service.run(job("whisper", target="en"), lambda p, n: None)

    assert "00:00:00,000 --> 00:00:01,000" in open(out, encoding="utf-8").read()


def test_downloaded_subtitles_lose_the_hearing_impaired_marks(service, monkeypatch):
    monkeypatch.setattr(service.opensubs, "download",
                        lambda file_id: "1\n00:00:01,000 --> 00:00:02,000\n[DOOR SLAMS]\n\n"
                                        "2\n00:00:03,000 --> 00:00:04,000\nJOHN: Corre!\n",
                        raising=False)

    out = service.run(job("opensubtitles", target="pt-BR", source="7"), lambda p, n: None)

    body = open(out, encoding="utf-8").read()
    assert "Corre!" in body
    assert "DOOR SLAMS" not in body and "JOHN:" not in body


def test_whisper_releases_the_translation_model_first(service, monkeypatch):
    from subzero.worker.srt import Cue
    freed = []
    service.ollama = type("O", (), {"release": lambda self: freed.append(True)})()
    monkeypatch.setattr("subzero.worker.service.audio_start_offset", lambda path: 0.0)
    monkeypatch.setattr("subzero.worker.service.extract_audio",
                        lambda path, duration=0, progress=None: "a.wav")
    monkeypatch.setattr("subzero.worker.service.transcribe",
                        lambda audio, holder, progress: ([Cue(1, 0, 1, "hello")], "en"))

    service.run(job("whisper", target="en"), lambda p, n: None)

    assert freed == [True]
