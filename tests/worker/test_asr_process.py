import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from subzero.worker import asr_process, tracks


def child(monkeypatch, script):
    launch = subprocess.Popen
    children = []

    def start(argv, **kwargs):
        proc = launch([sys.executable, "-u", "-c", script], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(asr_process.subprocess, "Popen", start)
    return children


def holder():
    return SimpleNamespace(name="cached-model", device="cpu", compute_type="int8", model=None)


def messages(*events):
    return "\n".join(f"print({json.dumps(json.dumps(event))}, flush=True)" for event in events)


def test_transcription_returns_only_after_child_exit(monkeypatch):
    children = child(monkeypatch, messages(
        {"event": "progress", "percent": 42},
        {"event": "cue", "index": 1, "start": 1.25, "end": 2.5, "text": "A spoken sentence."},
        {"event": "complete", "language": "en", "cues": 1},
    ))
    progress = []
    cues, language = asr_process.transcribe_in_process("audio.wav", holder(),
                                                     lambda phase, percent: progress.append(percent))
    assert language == "en"
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(1.25, 2.5, "A spoken sentence.")]
    assert progress == [42]
    assert children[0].poll() == 0


@pytest.mark.parametrize("script, error", [
    ("print('not json')", "invalid JSON"),
    ("print('[]')", "invalid event"),
    (messages({"event": "progress", "percent": True}), "invalid progress"),
    (messages({"event": "cue", "index": 1, "start": -1, "end": 2, "text": "words"}), "invalid subtitle cue"),
    (messages({"event": "cue", "index": True, "start": 0, "end": 2, "text": "words"}), "invalid subtitle cue"),
    (messages({"event": "cue", "index": 1.0, "start": 0, "end": 2, "text": "words"}), "invalid subtitle cue"),
    (messages({"event": "complete", "language": "en", "cues": 1}), "incomplete transcription"),
    (messages({"event": "cue", "index": 1, "start": 0, "end": 2, "text": "words"}), "without a complete transcription"),
    (messages({"event": "error", "message": "cache is incomplete"}), "cache is incomplete"),
    ("print('x' * 65537)", "size limit"),
    ("import sys; sys.exit(3)", "status 3"),
])
def test_invalid_transcription_reaps_child_before_raising(monkeypatch, script, error):
    children = child(monkeypatch, script)
    with pytest.raises(RuntimeError, match=error):
        asr_process.transcribe_in_process("audio.wav", holder(), lambda *_: None)
    assert children[0].poll() is not None


def test_cancellation_terminates_and_reaps_transcription(monkeypatch):
    script = messages({"event": "progress", "percent": 1}) + "\nimport time; time.sleep(60)"
    children = child(monkeypatch, script)

    def cancel(*args):
        raise RuntimeError("Job cancelled")

    with pytest.raises(RuntimeError, match="Job cancelled"):
        asr_process.transcribe_in_process("audio.wav", holder(), cancel)
    assert children[0].poll() is not None


def test_a_complete_event_does_not_allow_more_cues(monkeypatch):
    children = child(monkeypatch, messages(
        {"event": "cue", "index": 1, "start": 0, "end": 1, "text": "words"},
        {"event": "complete", "language": "en", "cues": 1},
        {"event": "cue", "index": 2, "start": 1, "end": 2, "text": "extra words"},
    ))
    with pytest.raises(RuntimeError, match="invalid event"):
        asr_process.transcribe_in_process("audio.wav", holder(), lambda *_: None)
    assert children[0].poll() is not None


@pytest.mark.parametrize("count", [True, 1.0])
def test_completion_requires_an_integer_cue_count(monkeypatch, count):
    children = child(monkeypatch, messages(
        {"event": "cue", "index": 1, "start": 0, "end": 1, "text": "words"},
        {"event": "complete", "language": "en", "cues": count},
    ))
    with pytest.raises(RuntimeError, match="incomplete transcription"):
        asr_process.transcribe_in_process("audio.wav", holder(), lambda *_: None)
    assert children[0].poll() is not None


def test_cancellation_remains_responsive_after_stdout_closes(monkeypatch):
    script = messages(
        {"event": "cue", "index": 1, "start": 0, "end": 1, "text": "words"},
        {"event": "complete", "language": "en", "cues": 1},
    ) + "\nimport os, time; os.close(1); time.sleep(60)"
    children = child(monkeypatch, script)

    def cancel(*args):
        raise RuntimeError("Job cancelled after stdout closed")

    with pytest.raises(RuntimeError, match="Job cancelled after stdout closed"):
        asr_process.transcribe_in_process("audio.wav", holder(), cancel)
    assert children[0].poll() is not None


def test_strict_transcription_uses_child_inside_compute_phase(monkeypatch):
    active = []

    @contextmanager
    def phase(kind):
        active.append(kind)
        try:
            yield True
        finally:
            active.pop()

    def transcribe(audio, model, progress):
        assert active == ["whisper"]
        return ["cues"], "en"

    monkeypatch.setattr(tracks, "compute_phase", phase)
    monkeypatch.setattr(tracks, "transcribe_in_process", transcribe)
    model = holder()
    model.lock = threading.Lock()
    assert tracks.transcribe("audio.wav", model, lambda *_: None) == (["cues"], "en")
    assert active == []


def test_preloaded_parent_model_cannot_start_isolated_child():
    model = holder()
    model.model = object()
    with pytest.raises(RuntimeError, match="already loaded"):
        asr_process.transcribe_in_process("audio.wav", model, lambda *_: None)


def test_ffmpeg_progress_cancellation_reaps_child():
    children = []

    def launch(argv, **kwargs):
        proc = subprocess.Popen([sys.executable, "-u", "-c",
                                 "print('out_time_ms=1000000', flush=True); import time; time.sleep(60)"],
                                **kwargs)
        children.append(proc)
        return proc

    def cancel(*args):
        raise RuntimeError("Job cancelled")

    with pytest.raises(RuntimeError, match="Job cancelled"):
        tracks.run_ffmpeg(["ffmpeg"], 10, cancel, "extracting", popen=launch)
    assert children[0].poll() is not None


@pytest.mark.parametrize("close_stdout", [False, True])
def test_ffmpeg_cancellation_is_checked_while_child_is_silent(close_stdout):
    children = []

    def launch(argv, **kwargs):
        script = "import os, time; " + ("os.close(1); " if close_stdout else "") + "time.sleep(60)"
        proc = subprocess.Popen([sys.executable, "-u", "-c", script], **kwargs)
        children.append(proc)
        return proc

    def cancel(*args):
        raise RuntimeError("Silent job cancelled")

    with pytest.raises(RuntimeError, match="Silent job cancelled"):
        tracks.run_ffmpeg(["ffmpeg"], 10, cancel, "extracting", popen=launch)
    assert children[0].poll() is not None


@pytest.mark.parametrize("locale_encoding", ["cp1252", "utf-8"])
def test_ffmpeg_invalid_output_cannot_be_reported_successful(locale_encoding):
    children = []

    def launch(argv, **kwargs):
        kwargs.setdefault("encoding", locale_encoding)
        proc = subprocess.Popen([sys.executable, "-c", "import os; os.write(1, bytes([255, 10]))"], **kwargs)
        children.append(proc)
        return proc

    with pytest.raises(RuntimeError, match="progress stream failed"):
        tracks.run_ffmpeg(["ffmpeg"], 10, lambda *_: None, "extracting", popen=launch)
    assert children[0].poll() is not None


def test_ffmpeg_utf8_diagnostics_are_preserved_with_a_windows_locale():
    children = []

    def launch(argv, **kwargs):
        kwargs.setdefault("encoding", "cp1252")
        script = "import os, sys; os.write(2, 'arquivo não encontrado\\n'.encode('utf-8')); sys.exit(1)"
        proc = subprocess.Popen([sys.executable, "-c", script], **kwargs)
        children.append(proc)
        return proc

    with pytest.raises(RuntimeError, match="arquivo não encontrado"):
        tracks.run_ffmpeg(["ffmpeg"], 10, lambda *_: None, "extracting", popen=launch)
    assert children[0].poll() == 1
