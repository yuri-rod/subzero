import threading
import time

import pytest

from subzero.worker.jellyfin import Media
from subzero.worker.srt import Cue, parse
from subzero.worker.tracks import (ModelHolder, Ollama, audio_start_offset, deliver, extract_embedded,
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

            def wait(self):
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
    def __init__(self, segments, language="ja"):
        self.segments = segments
        self.language = language
        self.unloaded = False
        self.lock = threading.Lock()

    def load(self):
        holder = self

        class Model:
            def transcribe(self, path, **kwargs):
                info = type("Info", (), {"language": holder.language, "duration": 10})()
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


def test_translate_falls_back_when_the_model_loses_lines():
    cues = [Cue(1, 0, 1, "hello"), Cue(2, 1, 2, "world")]
    out = translate(cues, "pt-BR", FakeOllama([["ola"]]), progress=lambda p, n: None)
    assert [c.text for c in out] == ["hello", "world"]


def test_translate_works_block_by_block():
    cues = [Cue(i, i, i + 1, f"line {i}") for i in range(45)]
    fake = FakeOllama([[f"linha {i}" for i in range(20)],
                       [f"linha {i}" for i in range(20)],
                       [f"linha {i}" for i in range(5)]])

    translate(cues, "pt-BR", fake, progress=lambda p, n: None)

    assert fake.blocks == 3


def test_translate_raises_when_every_block_falls_back():
    cues = [Cue(i, i, i + 1, f"line {i}") for i in range(45)]
    fake = FakeOllama([["so uma linha"], ["so uma linha"], ["so uma linha"]])

    with pytest.raises(RuntimeError, match="nao traduziu nenhum bloco"):
        translate(cues, "pt-BR", fake, progress=lambda p, n: None)


def test_ollama_prompt_numbers_the_lines():
    sent = {}

    class HTTP:
        def request(self, method, url, **kwargs):
            sent["url"] = url
            sent["json"] = kwargs["json"]
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"response": "1. ola\n2. mundo"}})()

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

        def wait(self):
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
                                  "json": lambda self: {"response": "1. ola"}})()

    http = Recorder()
    Ollama("http://o", "gemma3:12b", http=http).translate_block([Cue(1, 0, 1, "hi")], "pt-BR")

    assert http.sent["keep_alive"] == "30s"


def test_model_holder_compute_type_defaults():
    assert ModelHolder("tiny", device="cuda").compute_type == "float16"
    assert ModelHolder("tiny", device="cpu").compute_type == "int8"
    assert ModelHolder("tiny", device="cpu", compute_type="float32").compute_type == "float32"
def test_strict_translation_never_keeps_untranslated_fallback_lines():
    cues=[Cue(1,0,1,'hello'),Cue(2,1,2,'world')]
    with pytest.raises(RuntimeError,match='bloco'):
        translate(cues,'pt-BR',FakeOllama([['ola']]),lambda *args:None,strict=True)


def test_numbered_translation_rejects_reordered_or_duplicate_lines():
    assert Ollama._lines('2. mundo\n1. ola') == []
    assert Ollama._lines('1. ola\n1. mundo') == []

