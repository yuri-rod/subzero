import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest


def rescue(tmp_path, **overrides):
    from subzero.caption_rescue import CaptionRescue

    return CaptionRescue(model="local-vision:9b", url="http://127.0.0.1:11434",
                         cache_dir=tmp_path / "rescue", model_digest="a" * 64, **overrides)


def frame(tmp_path, timestamp, *, admitted=True, content=b"caption pixels"):
    from subzero.caption_rescue import CaptionFrame

    tmp_path.mkdir(parents=True, exist_ok=True)
    image = tmp_path / f"frame-{timestamp}.png"
    image.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return CaptionFrame(timestamp, image, digest, {"source_sha256": digest, "admitted": admitted})


def test_rescue_constructor_is_offline_and_identity_tracks_model(tmp_path, monkeypatch):
    import http.client

    monkeypatch.setattr(http.client, "HTTPConnection", lambda *a, **kw: pytest.fail("Constructor connected"))
    first = rescue(tmp_path)
    from subzero.caption_rescue import CaptionRescue
    other = CaptionRescue(model="local-vision:9b", url="http://127.0.0.1:11434",
                          cache_dir=tmp_path / "other", model_digest="b" * 64)
    assert first.identity != other.identity
    assert first.identity == rescue(tmp_path / "elsewhere").identity
    assert not (tmp_path / "rescue").exists()


@pytest.mark.parametrize("url", ["https://example.com", "http://192.168.1.5:11434", "http://user:pass@127.0.0.1:11434",
                                 "http://127.0.0.1:11434/elsewhere", "http://127.0.0.1:11434?token=secret"])
def test_rescue_cannot_send_frame_to_nonlocal_or_credentialed_endpoint(tmp_path, url):
    from subzero.caption_rescue import CaptionRescue

    with pytest.raises(ValueError, match="loopback"):
        CaptionRescue(model="local-vision:9b", url=url, cache_dir=tmp_path, model_digest="a" * 64)


def test_rescue_reads_each_admitted_frame_once_and_preserves_real_changes(tmp_path, monkeypatch):
    client = rescue(tmp_path)
    frames = [frame(tmp_path, n / 10) for n in range(3)]
    supplied = iter(["We need Sam.", "We need Pam.", "We need Sam."])
    calls = []

    def request(path, payload=None):
        calls.append((path, payload))
        if path == "/api/tags":
            return {"models": [{"name": "local-vision:9b", "digest": "a" * 64}]}
        return {"response": next(supplied), "done": True, "done_reason": "stop"}

    monkeypatch.setattr(client, "_request", request)
    assert client.read_frames("video", frames) == {0: "We need Sam.", .1: "We need Pam.", .2: "We need Sam."}
    assert len([path for path, _ in calls if path == "/api/generate"]) == 3
    for path, payload in calls:
        if path == "/api/generate":
            assert len(payload["images"]) == 1
            assert "Sam" not in payload["prompt"]
            assert "context" not in payload
            assert payload["options"] == {"num_ctx": 4096, "num_predict": 256, "temperature": 0}
            assert payload["think"] is False
            assert payload["keep_alive"] == "2m"
    calls.clear()
    assert client.read_frames("video", frames)[.1] == "We need Pam."
    assert calls == []


def test_rescue_does_not_recognize_native_rejected_credit_or_blank(tmp_path, monkeypatch):
    client = rescue(tmp_path)
    monkeypatch.setattr(client, "_request", lambda *a: pytest.fail("Rejected frame reached model"))
    assert client.read_frames("video", [frame(tmp_path, 1, admitted=False)]) == {1: ""}


def test_rescue_rejects_changed_pixels_even_when_cached(tmp_path, monkeypatch):
    client = rescue(tmp_path)
    captured = frame(tmp_path, 1)
    captured.image.write_bytes(b"different pixels")
    monkeypatch.setattr(client, "_request", lambda *a: pytest.fail("Changed pixels reached model"))
    with pytest.raises(RuntimeError, match="changed"):
        client.read_frames("video", [captured])


def test_rescue_rejects_different_frames_at_one_timestamp(tmp_path):
    client = rescue(tmp_path)
    first = frame(tmp_path, 1)
    second = frame(tmp_path / "second", 1, content=b"other")
    with pytest.raises(RuntimeError, match="timestamp"):
        client.read_frames("video", [first, second])


@pytest.mark.parametrize("response", [
    {"response": "I _ got you.", "done": False},
    {"response": "I _ got you.", "done": True, "done_reason": "length"},
    {"response": ["I _ got you."], "done": True, "done_reason": "stop"},
    {"response": "text\x00hidden", "done": True, "done_reason": "stop"},
])
def test_rescue_protocol_failure_is_not_retried_or_cached(tmp_path, monkeypatch, response):
    client = rescue(tmp_path)
    calls = []

    def request(path, payload=None):
        calls.append(path)
        if path == "/api/tags":
            return {"models": [{"name": "local-vision:9b", "digest": "a" * 64}]}
        return response

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RuntimeError, match="caption"):
        client.read_frames("video", [frame(tmp_path, 1)])
    assert calls.count("/api/generate") == 1
    proofs = list((tmp_path / "rescue").rglob("evidence.json"))
    assert len(proofs) == 1
    assert json.loads(proofs[0].read_text())["response"] == response
    assert not [path for path in (tmp_path / "rescue").rglob("*.json") if path.name != "evidence.json"]


def test_rescue_verifies_model_digest_before_reading_pixels(tmp_path, monkeypatch):
    client = rescue(tmp_path)
    calls = []

    def request(path, payload=None):
        calls.append(path)
        return {"models": [{"name": "local-vision:9b", "digest": "b" * 64}]}

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RuntimeError, match="digest"):
        client.read_frames("video", [frame(tmp_path, 1)])
    assert calls == ["/api/tags"]


def test_rescue_does_not_trust_readings_when_model_digest_changes_midpass(tmp_path, monkeypatch):
    client = rescue(tmp_path)
    tag_calls = []

    def request(path, payload=None):
        if path == "/api/tags":
            tag_calls.append(path)
            return {"models": [{"name": "local-vision:9b", "digest": ("a" if len(tag_calls) == 1 else "b") * 64}]}
        return {"response": "Keep this secret.", "done": True, "done_reason": "stop"}

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RuntimeError, match="digest"):
        client.read_frames("video", [frame(tmp_path, 1)])
    proofs = list((tmp_path / "rescue").rglob("evidence.json"))
    assert len(proofs) == 1
    assert not [path for path in (tmp_path / "rescue").rglob("*.json") if path.name != "evidence.json"]


def test_rescue_cancellation_between_frames_releases_compute_without_next_request(tmp_path, monkeypatch):
    from subzero import caption_rescue

    client = rescue(tmp_path)
    events = []

    @contextmanager
    def phase(kind, **kwargs):
        events.append("started")
        try:
            yield True
        finally:
            events.append("stopped")

    def request(path, payload=None):
        if path == "/api/tags":
            return {"models": [{"name": "local-vision:9b", "digest": "a" * 64}]}
        events.append("read")
        return {"response": "Keep this secret.", "done": True, "done_reason": "stop"}

    def progress(done, total):
        assert total == 2
        if done == 1:
            raise RuntimeError("Job cancelled")

    monkeypatch.setattr(caption_rescue, "compute_phase", phase)
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RuntimeError, match="Job cancelled"):
        client.read_frames("video", [frame(tmp_path, 1), frame(tmp_path, 2)], progress=progress)
    assert events == ["started", "read", "stopped"]


def test_rescue_empty_pass_reports_completion_without_starting_services(tmp_path, monkeypatch):
    client = rescue(tmp_path)
    monkeypatch.setattr(client, "_request", lambda *a: pytest.fail("Empty pass contacted Ollama"))
    updates = []
    assert client.read_frames("video", [], progress=lambda done, total: updates.append((done, total))) == {}
    assert updates[-1] == (0, 0)
