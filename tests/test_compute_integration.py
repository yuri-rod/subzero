from contextlib import contextmanager

import pytest

from subzero import translate as core
from subzero.convert import Cue
from subzero.worker import tracks
from subzero.worker.srt import Cue as WorkerCue


def phase_log(monkeypatch, module):
    events = []

    @contextmanager
    def phase(kind, **options):
        events.append(("enter", kind, options))
        try:
            yield True
        finally:
            events.append(("exit", kind))

    monkeypatch.setattr(module, "compute_phase", phase)
    return events


def test_cli_translation_retains_one_phase_across_all_blocks(monkeypatch):
    events = phase_log(monkeypatch, core)
    client = core.OllamaClient(model="test-model")
    cues = [Cue("00:00:01,000", "00:00:02,000", f"Source {index}.") for index in range(3)]

    def translated(block, *args, **kwargs):
        assert events[0][0] == "enter"
        assert not any(event[0] == "exit" for event in events)
        events.append(("block", len(block)))
        return ["Translated words."] * len(block)

    monkeypatch.setattr(core, "_translate_lines", translated)
    assert len(core.translate_cues(cues, "pt-BR", client, batch_size=1)) == 3
    assert [event[0] for event in events] == ["enter", "block", "block", "block", "exit"]


def test_worker_translation_releases_phase_after_a_failed_block(monkeypatch):
    events = phase_log(monkeypatch, tracks)
    client = tracks.Ollama("http://127.0.0.1:11434", "test-model")
    cues = [WorkerCue(index + 1, index, index + 1, "Source words.") for index in range(25)]

    def translated(block, *args, **kwargs):
        events.append(("block", len(block)))
        if len(block) < 20:
            raise RuntimeError("model request failed")
        return ["Translated words."] * len(block)

    monkeypatch.setattr(tracks, "_translate_lines", translated)
    with pytest.raises(RuntimeError, match="model request failed"):
        tracks.translate(cues, "pt-BR", client, lambda *_: None)
    assert [event[0] for event in events] == ["enter", "block", "block", "exit"]


def test_strict_release_uses_verified_shutdown_without_http_unload(monkeypatch):
    events = phase_log(monkeypatch, tracks)

    class NoRequest:
        def request(self, *args, **kwargs):
            raise AssertionError("Release must not mutate another phase through HTTP")

    tracks.Ollama("http://127.0.0.1:11434", "test-model", http=NoRequest()).release()
    assert events == [("enter", "ollama", {"ollama_url": "http://127.0.0.1:11434", "start_ollama": False}),
                      ("exit", "ollama")]
