import threading
import time
from pathlib import Path

import pytest

from subzero.worker.jellyfin import Media
from subzero.worker.srt import Cue, parse
from subzero.worker.tracks import (ModelHolder, Ollama, audio_start_offset, deliver, extract_audio, extract_embedded,
                              last_lines, run_ffmpeg, shift, sidecar_path, transcribe,
                              translate)


def media(path="F:\\FILMES\\Filme.mkv", audio="eng"):
    return Media(item_id="abc", name="Filme", path=path, container="mkv", duration=120,
                 audio_lang=audio, source_id="src")


class FakeRun:
    def __init__(self, code=0, stderr=""):
        self.code = code
        self.stderr = stderr
        self.cmd = None

    def __call__(self, cmd, **kwargs):
        self.cmd = cmd
        return type("R", (), {"returncode": self.code, "stderr": self.stderr, "stdout": ""})()


class FakePopen:
    def __init__(self, lines, code=0, stderr="", captured=None):
        self.lines = lines
        self.code = code
        self._stderr = stderr
        self.captured = captured

    def __call__(self, cmd, **kwargs):
        if self.captured is not None:
            self.captured.append(cmd)
        holder = self

        class Proc:
            stdout = iter(holder.lines)
            stderr = type("E", (), {"read": lambda self: holder._stderr})()

            def wait(self, timeout=None):
                return holder.code

        return Proc()


def test_sidecar_path_keeps_the_windows_folder():
    assert sidecar_path("F:\\FILMES\\Filme (2020).mkv", "pt-BR") == "F:\\FILMES\\Filme (2020).pt-BR.srt"


def test_sidecar_path_handles_posix_too():
    assert sidecar_path("/media/Filme.mkv", "en") == "/media/Filme.en.srt"


def test_extract_embedded_maps_the_subtitle_relative_stream(tmp_path):
    cmds = []
    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Filme.eng.srt").write_text("1\n")

    out = extract_embedded(media(path=str(video)), sub_index=3, lang="eng",
                           popen=FakePopen([], captured=cmds))

    assert out == str(tmp_path / "Filme.eng.srt")
    # sem -nostdin o ffmpeg trava lendo stdin quando o worker roda como servico
    assert cmds[0][:3] == ["ffmpeg", "-nostdin", "-y"]
    # 0:s:3 e a terceira legenda de dentro do arquivo, o Index do Jellyfin nao serve aqui
    assert "0:s:3" in cmds[0]
    assert out in cmds[0]


def test_extract_embedded_refuses_an_empty_track(tmp_path):
    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Filme.eng.srt").write_text("")

    with pytest.raises(RuntimeError, match="nenhuma legenda"):
        extract_embedded(media(path=str(video)), sub_index=9, lang="eng", popen=FakePopen([]))

    assert not (tmp_path / "Filme.eng.srt").exists()


def test_extract_embedded_reports_the_ffmpeg_error():
    with pytest.raises(RuntimeError, match="Stream map"):
        extract_embedded(media(), sub_index=9, lang="eng",
                         popen=FakePopen([], code=1, stderr="Stream map '0:9' matches no streams"))


def test_ffmpeg_error_keeps_the_end_not_the_banner():
    banner = "\n".join(f"ffmpeg banner line {i}" for i in range(40))
    with pytest.raises(RuntimeError) as err:
        extract_embedded(media(), sub_index=9, lang="eng",
                         popen=FakePopen([], code=1, stderr=banner + "\nInvalid stream specifier"))

    assert "Invalid stream specifier" in str(err.value)
    assert "banner line 0" not in str(err.value)


class FakeSegment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class FakeHolder:
    def __init__(self, segments, language="ja", duration=None):
        self.segments = segments
        self.language = language
        self.duration = duration if duration is not None else max(10, max((seg.end for seg in segments), default=0))
        self.unloaded = False
        self.lock = threading.Lock()

    def load(self):
        holder = self

        class Model:
            def transcribe(self, path, **kwargs):
                info = type("Info", (), {"language": holder.language, "duration": holder.duration})()
                return iter(holder.segments), info

        return Model()

    def unload(self):
        self.unloaded = True


def test_transcribe_builds_cues_and_reports_the_language(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    holder = FakeHolder([FakeSegment(0, 1.5, " Ola "), FakeSegment(2, 3, "mundo")])
    seen = []

    cues, lang = transcribe(str(audio), holder, progress=lambda phase, pct: seen.append(pct))

    assert [c.text for c in cues] == ["Ola", "mundo"]
    assert cues[0].end == 1.5
    assert lang == "ja"
    assert holder.unloaded is True
    assert seen


def test_transcribe_without_speech_fails(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        transcribe(str(audio), FakeHolder([]), progress=lambda phase, pct: None)


def test_transcribe_clamps_audio_end_and_skips_late_segments():
    holder = FakeHolder([FakeSegment(8.88, 9.72, "One."), FakeSegment(9.72, 10.72, "Two."),
                         FakeSegment(10.1, 11.0, "Outside audio.")], duration=10)
    cues, _ = transcribe("audio.wav", holder, progress=lambda *args: None)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(8.88, 9.72, "One."), (9.72, 10.0, "Two.")]


@pytest.mark.parametrize("duration", [0, float("inf"), float("nan")])
def test_transcribe_without_finite_duration_keeps_valid_timestamps(duration):
    holder = FakeHolder([FakeSegment(8, 11, "A valid caption.")], duration=duration)
    cues, _ = transcribe("audio.wav", holder, progress=lambda *args: None)
    assert [(cue.start, cue.end) for cue in cues] == [(8, 11)]


def test_transcribe_rejects_a_repetition_loop(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    segments = [FakeSegment(i, i + 1, "obrigado por assistir") for i in range(20)]
    with pytest.raises(RuntimeError, match="travou repetindo"):
        transcribe(str(audio), FakeHolder(segments), progress=lambda phase, pct: None)


def test_transcribe_allows_a_few_repeated_lines(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    segments = [FakeSegment(i, i + 1, f"linha {i}") for i in range(16)]
    segments += [FakeSegment(20 + i, 21 + i, "obrigado") for i in range(4)]
    cues, _ = transcribe(str(audio), FakeHolder(segments), progress=lambda phase, pct: None)
    assert len(cues) == 20


class FakeOllama:
    def __init__(self, replies):
        self.replies = list(replies)
        self.blocks = 0

    def translate_block(self, cues, target_lang):
        self.blocks += 1
        return self.replies.pop(0)


def test_translate_keeps_index_and_timing():
    cues = [Cue(1, 0, 1, "hello"), Cue(2, 1, 2, "world")]
    out = translate(cues, "pt-BR", FakeOllama([["ola", "mundo"]]), progress=lambda p, n: None)

    assert [c.text for c in out] == ["ola", "mundo"]
    assert [(c.start, c.end) for c in out] == [(0, 1), (1, 2)]


def test_translate_fails_when_the_model_loses_lines():
    cues = [Cue(1, 0, 1, "hello"), Cue(2, 1, 2, "world")]
    with pytest.raises(RuntimeError, match="preserv"):
        translate(cues, "pt-BR", FakeOllama([[]] * 6), progress=lambda p, n: None)


def test_translate_works_block_by_block():
    cues = [Cue(i, i, i + 1, f"line {i}") for i in range(45)]
    fake = FakeOllama([[f"linha {i}" for i in range(20)],
                       [f"linha {i}" for i in range(20)],
                       [f"linha {i}" for i in range(5)]])

    translate(cues, "pt-BR", fake, progress=lambda p, n: None)

    assert fake.blocks == 3


def test_translate_stops_at_the_first_unrecoverable_block():
    cues = [Cue(i, i, i + 1, f"line {i}") for i in range(45)]
    fake = FakeOllama([[]] * 6)

    with pytest.raises(RuntimeError, match="preserv"):
        translate(cues, "pt-BR", fake, progress=lambda p, n: None)
    assert fake.blocks == 6


def test_ollama_prompt_numbers_the_lines():
    sent = {}

    class HTTP:
        def request(self, method, url, **kwargs):
            sent["url"] = url
            sent["json"] = kwargs["json"]
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": "1. ola\n2. mundo", "done": True, "done_reason": "stop"}})()

    out = Ollama("http://ollama", "gemma3:12b", http=HTTP()).translate_block(
        [Cue(1, 0, 1, "hello"), Cue(2, 1, 2, "world")], "pt-BR")

    assert out == ["ola", "mundo"]
    assert "pt-BR" in sent["json"]["prompt"]
    assert sent["json"]["stream"] is False


class FakeJellyfin:
    def __init__(self):
        self.refreshed = []

    def refresh(self, item_id):
        self.refreshed.append(item_id)


def test_deliver_writes_the_sidecar_and_refreshes(tmp_path):
    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    jf = FakeJellyfin()

    path = deliver(media(path=str(video)), [Cue(1, 0, 1, "Oi")], "pt-BR", jf)

    assert path == str(tmp_path / "Filme.pt-BR.srt")
    assert parse(open(path, encoding="utf-8").read())[0].text == "Oi"
    assert jf.refreshed == ["abc"]


@pytest.mark.parametrize('suffix', ['', '[EZTVx.to]', '[TGx]'])
def test_deliver_prunes_literal_stem_sidecars_and_preserves_neighbors(tmp_path, suffix):
    from types import SimpleNamespace

    video = tmp_path / f'Movie{suffix}.mkv'
    video.write_bytes(b'video')
    episode = media(path=str(video))
    episode.embedded = [SimpleNamespace(lang='eng')]
    stale = [video.with_suffix('.srt'), video.with_suffix('.en.srt')]
    preserved = [video.with_suffix('.fr.srt'), tmp_path / f'{video.stem}.en.srt.bak',
                 tmp_path / f'{video.stem}Xen.srt', tmp_path / f'{video.stem}.extended.en.srt']
    for path in stale + preserved:
        path.write_text('existing subtitle')
    target = deliver(episode, [Cue(1, 0, 1, 'Oi')], 'pt-BR', FakeJellyfin())
    assert Path(target).exists()
    assert all(not path.exists() for path in stale)
    assert all(path.read_text() == 'existing subtitle' for path in preserved)


def test_deliver_refuses_an_empty_subtitle(tmp_path):
    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        deliver(media(path=str(video)), [], "pt-BR", FakeJellyfin())


def test_run_ffmpeg_turns_out_time_into_percent():
    from subzero.worker.tracks import run_ffmpeg

    seen = []
    run_ffmpeg(["ffmpeg"], duration=100, progress=lambda phase, pct: seen.append(pct),
               phase="extraindo", popen=FakePopen(["out_time_ms=25000000\n", "out_time_ms=90000000\n"]))

    assert seen == [25, 90]


def test_run_ffmpeg_asks_ffmpeg_for_progress():
    from subzero.worker.tracks import run_ffmpeg

    cmds = []
    run_ffmpeg(["ffmpeg", "-i", "x.mkv"], duration=10, progress=lambda p, n: None,
               phase="extraindo", popen=FakePopen([], captured=cmds))

    assert cmds[0][-3:] == ["-progress", "pipe:1", "-nostats"]


def test_run_ffmpeg_raises_with_the_tail_of_stderr():
    from subzero.worker.tracks import run_ffmpeg

    with pytest.raises(RuntimeError, match="Invalid stream"):
        run_ffmpeg(["ffmpeg"], duration=10, progress=lambda p, n: None, phase="extraindo",
                   popen=FakePopen([], code=1, stderr="banner\nInvalid stream specifier"))


class FakeProbe:
    """ffprobe stub: devolve o JSON de streams que o teste quiser."""

    def __init__(self, payload, code=0):
        self.payload = payload
        self.code = code
        self.cmd = None

    def __call__(self, cmd, **kwargs):
        self.cmd = cmd
        out = payload_json(self.payload)
        return type("R", (), {"returncode": self.code, "stdout": out, "stderr": ""})()


def payload_json(streams):
    import json
    return json.dumps({"streams": streams})


def test_audio_offset_reads_the_gap_between_audio_and_video():
    probe = FakeProbe([{"codec_type": "video", "start_time": "0.000000"},
                       {"codec_type": "audio", "start_time": "1.400000"}])

    assert audio_start_offset("F:\\x.mkv", run=probe) == pytest.approx(1.4)


def test_audio_offset_is_zero_when_both_streams_start_together():
    probe = FakeProbe([{"codec_type": "video", "start_time": "0.000000"},
                       {"codec_type": "audio", "start_time": "0.000000"}])

    assert audio_start_offset("F:\\x.mkv", run=probe) == 0.0


def test_audio_offset_ignores_a_gap_too_small_to_hear():
    probe = FakeProbe([{"codec_type": "video", "start_time": "0.000000"},
                       {"codec_type": "audio", "start_time": "0.010000"}])

    assert audio_start_offset("F:\\x.mkv", run=probe) == 0.0


def test_audio_offset_never_goes_negative():
    probe = FakeProbe([{"codec_type": "video", "start_time": "2.000000"},
                       {"codec_type": "audio", "start_time": "0.000000"}])

    assert audio_start_offset("F:\\x.mkv", run=probe) == 0.0


def test_audio_offset_falls_back_to_zero_when_ffprobe_fails():
    probe = FakeProbe([], code=1)

    assert audio_start_offset("F:\\x.mkv", run=probe) == 0.0


def test_audio_offset_falls_back_to_zero_without_ffprobe():
    def missing(cmd, **kwargs):
        raise OSError("ffprobe nao esta no PATH")

    assert audio_start_offset("F:\\x.mkv", run=missing) == 0.0


def test_audio_offset_survives_a_stream_without_start_time():
    probe = FakeProbe([{"codec_type": "video"}, {"codec_type": "audio", "start_time": "N/A"}])

    assert audio_start_offset("F:\\x.mkv", run=probe) == 0.0


def test_shift_moves_every_cue_forward():
    cues = [Cue(1, 0.0, 1.5, "um"), Cue(2, 10.0, 12.0, "dois")]

    moved = shift(cues, 1.4)

    assert [(c.start, c.end) for c in moved] == [(1.4, 2.9), (11.4, 13.4)]
    assert [c.index for c in moved] == [1, 2]
    assert [c.text for c in moved] == ["um", "dois"]


def test_shift_by_zero_returns_the_same_timings():
    cues = [Cue(1, 0.0, 1.5, "um")]

    assert [(c.start, c.end) for c in shift(cues, 0.0)] == [(0.0, 1.5)]


def test_transcribe_never_runs_two_models_at_once():
    """A fila e o HTTP chamam o mesmo holder: dois loads simultaneos estouram a VRAM."""
    overlap = []
    live = []

    class SlowHolder(FakeHolder):
        def load(self):
            live.append(1)
            overlap.append(len(live))
            time.sleep(0.05)
            model = super().load()
            live.pop()
            return model

    holder = SlowHolder([type("S", (), {"start": 0.0, "end": 1.0, "text": "oi"})()])
    threads = [threading.Thread(target=transcribe, args=("a.wav", holder, lambda p, n: None))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max(overlap) == 1


def test_run_ffmpeg_drains_stderr_while_stdout_is_read():
    """Um stderr grande nao pode travar o processo: tem de ser lido em paralelo."""
    started = threading.Event()
    released = threading.Event()

    class BlockingProc:
        """Simula o ffmpeg: so termina o stdout depois que alguem leu o stderr."""

        def __init__(self):
            self.stdout = self._stdout()
            self.stderr = self

        def read(self):
            started.set()
            released.set()
            return "x" * 200_000 + "\nerro de verdade no fim"

        def _stdout(self):
            yield "out_time_ms=1000000\n"
            # o ffmpeg real bloqueia aqui ate o pai drenar o stderr
            assert released.wait(5), "stderr nunca foi drenado: deadlock"
            yield "out_time_ms=2000000\n"

        def wait(self, timeout=None):
            return 1

    proc = BlockingProc()
    seen = []
    with pytest.raises(RuntimeError) as err:
        run_ffmpeg(["ffmpeg"], 10, lambda p, n: seen.append(n), "fase", popen=lambda *a, **k: proc)

    assert started.is_set()
    assert "erro de verdade no fim" in str(err.value)
    assert seen == [10, 20]


def test_last_lines_keeps_the_error_when_one_line_is_enormous():
    huge = "x" * 200_000
    text = f"ffmpeg version 9.0\n{huge}\nStream map error: matches no streams"

    assert "matches no streams" in last_lines(text)


def test_sidecar_path_can_drop_the_language_tag():
    assert sidecar_path("F:\\FILMES\\Filme.mkv", "pt-BR") == "F:\\FILMES\\Filme.pt-BR.srt"
    assert sidecar_path("F:\\FILMES\\Filme.mkv", "pt-BR", bare=True) == "F:\\FILMES\\Filme.srt"


def test_ollama_asks_the_model_to_be_released_after_the_job():
    """Segurar o gemma carregado deixa o whisper do proximo job sem VRAM."""
    class Recorder:
        def __init__(self):
            self.sent = None

        def request(self, method, url, **kwargs):
            self.sent = kwargs.get("json")
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": "1. ola", "done": True, "done_reason": "stop"}})()

    http = Recorder()
    Ollama("http://o", "gemma3:12b", http=http).translate_block([Cue(1, 0, 1, "hi")], "pt-BR")

    assert http.sent["keep_alive"] == "2m"


def test_model_holder_compute_type_defaults():
    assert ModelHolder("tiny", device="cuda").compute_type == "float16"
    assert ModelHolder("tiny", device="cpu").compute_type == "int8"
    assert ModelHolder("tiny", device="cpu", compute_type="float32").compute_type == "float32"


def test_model_holder_requires_cached_model(tmp_path, monkeypatch):
    import faster_whisper
    import faster_whisper.utils

    for filename in ("model.bin", "config.json", "tokenizer.json"):
        (tmp_path / filename).write_text("cached")

    def cached_snapshot(name, **kwargs):
        if not kwargs.get("local_files_only"):
            raise AssertionError("runtime attempted to permit remote model lookup")
        return str(tmp_path)

    class CachedModel:
        def __init__(self, name, **kwargs):
            if not kwargs.get("local_files_only"):
                raise AssertionError("runtime attempted to permit remote model lookup")
            self.name = name

    monkeypatch.setattr(faster_whisper, "WhisperModel", CachedModel)
    monkeypatch.setattr(faster_whisper.utils, "download_model", cached_snapshot)
    holder = ModelHolder("large-v3-turbo", device="cpu")
    assert holder.load().name == str(tmp_path)


def test_model_holder_reports_missing_local_cache(monkeypatch):
    import faster_whisper
    import faster_whisper.utils
    from huggingface_hub.errors import LocalEntryNotFoundError

    def missing_model(*args, **kwargs):
        raise LocalEntryNotFoundError("snapshot not found")

    monkeypatch.setattr(faster_whisper, "WhisperModel", missing_model)
    monkeypatch.setattr(faster_whisper.utils, "download_model", missing_model)
    with pytest.raises(RuntimeError, match="not cached locally"):
        ModelHolder("large-v3-turbo", device="cpu").load()


def test_model_holder_rejects_cache_missing_tokenizer(tmp_path, monkeypatch):
    import faster_whisper

    (tmp_path / "model.bin").write_bytes(b"cached model")
    (tmp_path / "config.json").write_text("{}")

    def remote_tokenizer(*args, **kwargs):
        raise AssertionError("Whisper would fetch a tokenizer for an incomplete cache")

    monkeypatch.setattr(faster_whisper, "WhisperModel", remote_tokenizer)
    with pytest.raises(RuntimeError, match="tokenizer.json"):
        ModelHolder(str(tmp_path), device="cpu").load()


def test_audio_extractions_for_same_video_use_separate_files(tmp_path, monkeypatch):
    from subzero.worker import tracks

    monkeypatch.setattr(tracks.tempfile, "tempdir", str(tmp_path))
    first = Path(extract_audio("episode.mkv", popen=FakePopen([])))
    second = Path(extract_audio("episode.mkv", popen=FakePopen([])))
    assert first != second
    assert first.is_file() and second.is_file()


def test_audio_extraction_maps_the_reference_stream(tmp_path, monkeypatch):
    from subzero.worker import tracks

    monkeypatch.setattr(tracks.tempfile, 'tempdir', str(tmp_path))
    commands = []
    monkeypatch.setattr(tracks, 'run_ffmpeg', lambda cmd, *args, **kwargs: commands.append(cmd))
    audio = Path(extract_audio('episode.mkv', audio_index=3))
    assert commands == [['ffmpeg', '-nostdin', '-y', '-i', 'episode.mkv', '-map', '0:3',
                         '-vn', '-ac', '1', '-ar', '16000', '-f', 'wav', str(audio)]]


@pytest.mark.parametrize('audio_index', [True, 1.0, '1', -1])
def test_audio_extraction_rejects_invalid_stream_before_creating_files(tmp_path, monkeypatch, audio_index):
    from subzero.worker import tracks

    monkeypatch.setattr(tracks.tempfile, 'tempdir', str(tmp_path))
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid stream must be rejected before running FFmpeg')
    monkeypatch.setattr(tracks, 'run_ffmpeg', unexpected)
    with pytest.raises(ValueError, match='audio stream index'):
        extract_audio('episode.mkv', audio_index=audio_index)
    assert not list(tmp_path.glob('*.wav'))


def test_failed_audio_extraction_removes_partial_wav(tmp_path, monkeypatch):
    from subzero.worker import tracks

    monkeypatch.setattr(tracks.tempfile, "tempdir", str(tmp_path))

    def fail(cmd, *args, **kwargs):
        Path(cmd[-1]).write_bytes(b"partial")
        raise RuntimeError("invalid audio stream")

    monkeypatch.setattr(tracks, "run_ffmpeg", fail)
    with pytest.raises(RuntimeError, match="invalid audio stream"):
        extract_audio("episode.mkv")
    assert not list(tmp_path.glob("*.wav"))
def test_strict_translation_never_keeps_untranslated_fallback_lines():
    cues=[Cue(1,0,1,'hello'),Cue(2,1,2,'world')]
    with pytest.raises(RuntimeError,match='bloco'):
        translate(cues,'pt-BR',FakeOllama([[]] * 6),lambda *args:None,strict=True)


def test_numbered_translation_rejects_reordered_or_duplicate_lines():
    assert Ollama._lines('2. mundo\n1. ola') == []
    assert Ollama._lines('1. ola\n1. mundo') == []


def test_ollama_preflight_checks_model_without_loading_it():
    sent = []

    class HTTP:
        def request(self, method, url, **kwargs):
            sent.append((method, url, kwargs))
            return type("R", (), {"status_code": 200})()

    Ollama("http://localhost:11434", "gemma3:4b", http=HTTP()).ensure_available()
    assert sent == [("POST", "http://localhost:11434/api/show",
                     {"json": {"model": "gemma3:4b"}, "timeout": 5})]


def test_ollama_preflight_reports_missing_model_without_response_body():
    class HTTP:
        def request(self, *args, **kwargs):
            return type("R", (), {"status_code": 404, "text": "private server response"})()

    with pytest.raises(RuntimeError, match="model.*installed") as caught:
        Ollama("http://localhost", "missing", http=HTTP()).ensure_available()
    assert "private" not in str(caught.value)


def test_ollama_preflight_reports_transport_failure_without_url():
    import httpx

    class HTTP:
        def request(self, *args, **kwargs):
            raise httpx.ConnectError("http://token:secret@localhost")

    with pytest.raises(RuntimeError, match="unavailable") as caught:
        Ollama("http://localhost", "model", http=HTTP()).ensure_available()
    assert "secret" not in str(caught.value)


def test_ollama_release_uses_short_timeout():
    import httpx
    sent = {}

    class HTTP:
        def request(self, *args, **kwargs):
            sent.update(kwargs)
            raise httpx.ReadTimeout("no response")

    Ollama("http://localhost", "model", http=HTTP()).release()
    assert sent["timeout"] == 5
    assert sent["json"]["keep_alive"] == 0


def test_ollama_generates_structured_cues_with_bounded_options():
    sent = {}

    class HTTP:
        def request(self, method, url, **kwargs):
            sent.update(kwargs["json"])
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": '[{"id":1,"text":"Ola"}]', "done": True, "done_reason": "stop"}})()

    client = Ollama("http://localhost", "model", http=HTTP(), num_ctx=2048, num_predict=1024)
    assert client.translate_block([Cue(1, 1, 2, "Hello")], "pt-BR") == ["Ola"]
    assert sent["format"]["type"] == "array"
    assert sent["options"] == {"temperature": 0, "num_ctx": 2048, "num_predict": 1024}


def test_worker_translation_splits_failed_blocks_and_preserves_names():
    fake = FakeOllama([[], [], ["Ola"], ["Jeff"]])
    cues = [Cue(7, 1, 2, "Hello"), Cue(8, 3, 4, "Jeff")]
    translated = translate(cues, "pt-BR", fake, lambda *args:None)
    assert [(c.index, c.start, c.end, c.text) for c in translated] == [(7, 1, 2, "Ola"), (8, 3, 4, "Jeff")]


def test_ollama_response_limit_is_checked_before_json_parsing():
    class HTTP:
        def request(self, *args, **kwargs):
            return type("R", (), {"status_code": 200, "content": b"x" * 131073})()

    with pytest.raises(RuntimeError, match="size limit"):
        Ollama("http://localhost", "model", http=HTTP()).translate_block([Cue(1, 1, 2, "Hello")], "pt-BR")


def test_equal_count_reordered_output_never_becomes_a_translation():
    class HTTP:
        def request(self, *args, **kwargs):
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": "2. mundo\n1. ola", "done": True, "done_reason": "stop"}})()

    cues = [Cue(1, 1, 2, "Hello"), Cue(2, 3, 4, "World")]
    with pytest.raises(RuntimeError, match="preserv"):
        translate(cues, "pt-BR", Ollama("http://localhost", "model", http=HTTP()), lambda *args:None)


def test_worker_translategemma_receives_source_language_through_translation():
    sent = {}

    class HTTP:
        def request(self, method, url, **kwargs):
            sent.update(kwargs["json"])
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": "Ola", "done": True, "done_reason": "stop"}})()

    cues = [Cue(7, 1, 2, "Hello")]
    client = Ollama("http://localhost", "translategemma:4b", http=HTTP())
    translated = translate(cues, "pt-BR", client, lambda *args:None, strict=True, source_lang="eng")
    assert [(c.index, c.start, c.end, c.text) for c in translated] == [(7, 1, 2, "Ola")]
    assert "English (en) to Brazilian Portuguese (pt-BR)" in sent["prompt"]


def test_worker_native_translation_keeps_each_cue_with_its_own_response():
    prompts = []
    replies = iter(["Olá, Jeff.", "Até mais."])

    class HTTP:
        def request(self, method, url, **kwargs):
            payload = kwargs["json"]
            prompts.append(payload["prompt"].split("\n\n\n", 1)[1])
            reply = next(replies)
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": reply, "done": True, "done_reason": "stop"}})()

    cues = [Cue(7, 1.2, 3.4, "Hello, Jeff."), Cue(8, 5.6, 7.8, "Goodbye.")]
    client = Ollama("http://localhost", "translategemma:12b", http=HTTP())
    translated = translate(cues, "pt-BR", client, lambda *args: None, source_lang="en")
    assert [(c.index, c.start, c.end, c.text) for c in translated] == [
        (7, 1.2, 3.4, "Olá, Jeff."), (8, 5.6, 7.8, "Até mais."),
    ]
    assert prompts == ["Hello, Jeff.", "Goodbye."]


@pytest.mark.parametrize("model", ["translategemma:4b", "kaelri/hy-mt2:7b"])
def test_worker_native_translation_rejects_token_limit_output(model):
    class HTTP:
        def request(self, *args, **kwargs):
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": "1. Olá.", "done": True, "done_reason": "length"}})()

    client = Ollama("http://localhost", model, http=HTTP())
    with pytest.raises(RuntimeError, match="preserv"):
        translate([Cue(1, 1, 2, "Hello.")], "pt-BR", client, lambda *args: None, source_lang="en")


def test_worker_native_translation_uses_only_previous_source_without_translating_it():
    prompts = []
    replies = iter(["Um.", "Dois.", "Três.", "Quatro.", "Cinco."])

    class HTTP:
        def request(self, method, url, **kwargs):
            payload = kwargs["json"]
            assert url == "http://localhost/api/generate"
            assert payload["raw"] is True
            assert payload["options"]["stop"] == ["<|eos|>", "<|extra_5|>"]
            prompts.append(payload["prompt"])
            reply = next(replies)
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": reply, "done": True, "done_reason": "stop"}})()

    cues = [Cue(i, i, i + 0.5, text) for i, text in enumerate(["One.", "Two.", "Three.", "Four.", "Five."])]
    client = Ollama("http://localhost", "kaelri/hy-mt2:7b", http=HTTP())
    translated = translate(cues, "pt-BR", client, lambda *args: None, source_lang="en")
    assert [(cue.index, cue.text) for cue in translated] == [(0, "Um."), (1, "Dois."), (2, "Três."), (3, "Quatro."), (4, "Cinco.")]
    middle_header, middle_source = prompts[2].split("\n[Source Text]\n", 1)
    assert middle_source == "Three.<|extra_0|>"
    assert all(text in middle_header for text in ["One.", "Two."])
    assert all(text not in middle_header for text in ["Three.", "Four.", "Five."])
    last_header = prompts[4].split("\n[Source Text]\n", 1)[0]
    assert "Three." in last_header and "Four." in last_header
    assert "One." in last_header and "Two." in last_header
    assert "Five." not in last_header


def test_worker_translation_keeps_previous_source_across_native_blocks(monkeypatch):
    monkeypatch.setattr("subzero.worker.tracks.BLOCK", 1)
    prompts = []

    class HTTP:
        def request(self, method, url, **kwargs):
            prompts.append(kwargs['json']['prompt'])
            return type('R', (), {'status_code': 200, 'json': lambda self: {
                'response': 'O baú.', 'done': True, 'done_reason': 'stop'}})()

    cues = [Cue(20, 30, 31, 'The chest.'), Cue(21, 32, 33, 'Open it.')]
    context = {'title': 'Survivor', 'previous_cues': ['The tribe found a wooden chest.']}
    translated = translate(cues, 'pt-BR', Ollama('http://localhost', 'kaelri/hy-mt2:7b', http=HTTP()),
                           lambda *args: None, source_lang='en', context=context)
    assert [(c.index, c.start, c.end) for c in translated] == [(20, 30, 31), (21, 32, 33)]
    headers = [prompt.split('\n[Source Text]\n', 1)[0] for prompt in prompts]
    assert headers[0] != headers[1]
    assert 'Programme title: Survivor' in headers[0]
    assert context['previous_cues'][0] in headers[0]
    assert 'The chest.' not in headers[0] and 'The chest.' in headers[1]
    assert all('Open it.' not in h for h in headers)
    assert [prompt.split('\n[Source Text]\n', 1)[1] for prompt in prompts] == [
        'The chest.<|extra_0|>', 'Open it.<|extra_0|>']


def test_worker_hymt2_translates_continuation_once_and_keeps_original_anchors():
    prompts = []

    class HTTP:
        def request(self, method, url, **kwargs):
            prompts.append(kwargs['json']['prompt'])
            return type('R', (), {'status_code': 200, 'json': lambda self: {
                'response': 'Passei cinco anos em lares adotivos.', 'done': True, 'done_reason': 'stop'}})()

    cues = [Cue(20, 30, 31, 'I spent'), Cue(21, 31.1, 33, 'five years in foster care.')]
    translated = translate(cues, 'pt-BR', Ollama('http://localhost', 'hy-mt2:7b', http=HTTP()),
                           lambda *args: None, source_lang='en')
    assert len(prompts) == 1
    assert 'I spent five years in foster care.' in ' '.join(prompts[0].split())
    assert ' '.join(c.text for c in translated) == 'Passei cinco anos em lares adotivos.'
    assert [(c.index, c.start, c.end) for c in translated] == [(20, 30, 31), (21, 31.1, 33)]


def test_worker_translation_does_not_cut_sentence_at_twentieth_anchor():
    class Recorder:
        model = 'hy-mt2:7b'

        def __init__(self):
            self.blocks = []

        def translate_block(self, cues, *args, **kwargs):
            self.blocks.append(cues)
            return ['Fala traduzida.'] * len(cues)

    cues = [Cue(i, i, i + .9, f'Complete sentence {i}.') for i in range(19)]
    cues += [Cue(19, 19, 19.9, 'I spent'), Cue(20, 20, 20.9, 'five years in foster care.')]
    client = Recorder()
    translate(cues, 'pt-BR', client, lambda *args: None, source_lang='en')
    assert any(cues[19] in block and cues[20] in block for block in client.blocks)


def test_worker_qwen_joins_complete_units_without_context_injection(monkeypatch):
    import json
    from subzero.translate import _ollama_payload
    monkeypatch.setattr('subzero.worker.tracks.BLOCK', 1)
    requests = []
    generated = 'We found the next hidden key.'

    class HTTP:
        def request(self, method, url, **kwargs):
            requests.append(kwargs['json'])
            return type('R', (), {'status_code': 200, 'json': lambda self: {
                'response': json.dumps([{'id': 1, 'text': generated}]),
                'done': True, 'done_reason': 'stop'}})()

    cues = [Cue(20, 30, 31, 'ALEX: I found'), Cue(21, 31.1, 32, 'ALEX: the hidden key.'),
            Cue(22, 32.1, 33, 'BLAIR: This is'), Cue(23, 33.1, 34, 'our next clue.')]
    context = {'title': 'Unrelated programme', 'previous_cues': ['Unused earlier dialogue.']}
    client = Ollama('http://localhost', 'qwen3.5:9b', http=HTTP())
    translated = translate(cues, 'pt-BR', client, lambda *args: None, source_lang='en', context=context)
    joined = [Cue(20, 30, 32, 'ALEX: I found\nthe hidden key.'),
              Cue(22, 32.1, 34, 'BLAIR: This is\nour next clue.')]
    assert requests == [_ollama_payload([cue], 'pt-BR', 'qwen3.5:9b', '2m', 4096, 2048,
                                        source_lang='en') for cue in joined]
    assert all('Unused earlier dialogue.' not in req['prompt'] for req in requests)
    assert [word for cue in translated for word in cue.text.split()] == generated.split() * 2
    assert [(cue.index, cue.start, cue.end) for cue in translated] == [(cue.index, cue.start, cue.end) for cue in cues]


@pytest.mark.parametrize('response', [
    {'response': 'Unstructured native text.', 'done': True, 'done_reason': 'stop'},
    {'response': '[{"id": 2, "text": "A complete response."}]', 'done': True, 'done_reason': 'stop'},
    {'response': '[{"id": 1, "text": "A complete response."}]', 'done': True, 'done_reason': 'length'},
    {'response': '[{"id": 1, "text": "One"}]', 'done': True, 'done_reason': 'stop'},
])
def test_worker_qwen_joining_retains_json_completion_and_word_count_guards(response):
    class HTTP:
        def request(self, *args, **kwargs):
            return type('R', (), {'status_code': 200, 'json': lambda self: response})()

    cues = [Cue(1, 1, 2, 'I found'), Cue(2, 2.1, 3, 'the key.')]
    client = Ollama('http://localhost', 'qwen3.5:9b', http=HTTP())
    with pytest.raises(RuntimeError, match='preserv|fewer words'):
        translate(cues, 'pt-BR', client, lambda *args: None, source_lang='en')


def test_cpu_provider_keeps_sentence_boundaries_without_starting_ollama(monkeypatch):
    def no_gpu(*args, **kwargs):
        raise AssertionError('CPU translation must not acquire an Ollama phase')

    monkeypatch.setattr('subzero.worker.tracks.compute_phase', no_gpu)
    seen = []

    class CPUTranslator:
        model = 'argos-en-pb'
        needs_local_compute = False
        uses_sentence_units = True
        supports_context = False

        def translate_block(self, cues, target_lang, source_lang=None, **kwargs):
            seen.append(cues)
            assert not kwargs.get('context')
            return ['Fala traduzida.' for cue in cues]

    cues = [Cue(i, i, i + .9, 'Complete thought.') for i in range(19)]
    cues += [Cue(19, 19, 19.9, 'I found'), Cue(20, 20, 20.9, 'the hidden key.')]
    translated = translate(cues, 'pt-BR', CPUTranslator(), lambda *args: None, source_lang='en')
    assert [len(block) for block in seen] == [19, 2]
    assert [(cue.index, cue.start, cue.end) for cue in translated] == [(cue.index, cue.start, cue.end) for cue in cues]
