import hashlib
from pathlib import Path
import struct
import sys

import pytest

from subzero import ocr
from subzero.convert import Cue


COARSE = [(0, ""), (.5, "I got you."), (1, "I got you."), (1.5, ""), (2, "")]
DENSE = [(n / 10, "I _ got you." if 5 <= n <= 11 else "") for n in range(21)]


def test_fill_gaps_uses_supplied_translation_client(tmp_path, monkeypatch):
    subtitle = tmp_path / "source.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nExisting dialogue.\n", encoding="utf-8")
    monkeypatch.setattr(ocr, "find_caption_gaps", lambda *a, **kw: [(2, 4)])
    monkeypatch.setattr(ocr, "extract_and_ocr_gaps", lambda *a, **kw: [
        Cue("00:00:02,000", "00:00:03,000", "Keep this secret.")])
    for name in ("OllamaClient", "OpenAIClient"):
        monkeypatch.setattr(ocr, name, lambda *a, **kw: pytest.fail("Unexpected provider construction"))
    supplied = object()

    def translate(cues, *, client, **kwargs):
        assert client is supplied
        return [Cue(cues[0].start, cues[0].end, "Guarde este segredo.")]

    monkeypatch.setattr(ocr, "translate_cues", translate)
    report = ocr.fill_subtitle_gaps("video", subtitle, target_lang="pt-BR", dry_run=True,
                                   translation_client=supplied)
    assert report.cues_recovered == 1
    assert report.cues[0].text == "Guarde este segredo."


def test_rescue_reconciles_coarse_and_dense_censor_evidence_before_loss_check(monkeypatch):
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: DENSE)
    with pytest.raises(RuntimeError, match="lost a confirmed"):
        ocr.refine_caption_timing("video", COARSE, 3)

    corrected = [(stamp, "I _ got you." if text else "") for stamp, text in COARSE]
    calls = []

    def rescue(*args):
        calls.append(args)
        return corrected, DENSE

    monkeypatch.setattr(ocr, "_rescue_caption_frames", rescue, raising=False)
    cues = ocr.refine_caption_timing("video", COARSE, 3, caption_rescue=object())
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        ("00:00:00,450", "00:00:01,150", "I _ got you.")]
    assert len(calls) == 1


def test_stable_native_captions_never_start_optional_rescue(monkeypatch):
    coarse = [(stamp, text.replace(" _", "")) for stamp, text in COARSE]
    dense = [(stamp, text.replace(" _", "")) for stamp, text in DENSE]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    monkeypatch.setattr(ocr, "_rescue_caption_frames", lambda *a: pytest.fail("Stable caption was rescued"), raising=False)
    cues = ocr.refine_caption_timing("video", coarse, 3, caption_rescue=object())
    assert [cue.text for cue in cues] == ["I got you."]


def test_rescue_failure_is_not_retried_or_reported_as_empty_success(monkeypatch):
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: DENSE)
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("Unresolved caption frame")

    monkeypatch.setattr(ocr, "_rescue_caption_frames", fail, raising=False)
    with pytest.raises(RuntimeError, match="Unresolved caption"):
        ocr.refine_caption_timing("video", COARSE, 3, caption_rescue=object())
    assert len(calls) == 1


def test_rescue_collects_separate_loss_intervals_without_global_text_mapping(monkeypatch):
    repeated = COARSE + [(10 + stamp, text) for stamp, text in COARSE]
    dense = DENSE + [(10 + stamp, text.replace(" _", "")) for stamp, text in DENSE]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    collected = []

    def rescue(video, coarse, scanned, duration, requests, intervals, client):
        collected.extend(intervals)
        return [(stamp, text.replace("I got", "I _ got") if stamp < 3 else text)
                for stamp, text in coarse], scanned

    monkeypatch.setattr(ocr, "_rescue_caption_frames", rescue, raising=False)
    cues = ocr.refine_caption_timing("video", repeated, 13, caption_rescue=object())
    assert [cue.text for cue in cues] == ["I _ got you.", "I got you."]
    assert collected and all(end < 3 for _, end in collected)


def test_unresolved_flicker_after_rescue_still_fails_original_gate(monkeypatch):
    coarse = [(0, ""), (.5, "Go on, please."), (1, "Go on, please."), (1.5, "")]
    dense = [(.4, ""), (.5, "Go on, please."), (.6, "Goron, please."), (.7, "Go on, please."), (.8, "")]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    monkeypatch.setattr(ocr, "_rescue_caption_frames", lambda *a: (coarse, dense), raising=False)
    with pytest.raises(RuntimeError, match="Unstable OCR"):
        ocr.refine_caption_timing("video", coarse, 2, caption_rescue=object())


def native_frame(text):
    return {"items": [{"text": text, "confidence": 1, "x": .3, "y": .1, "width": .4,
                       "height": .06, "captionInk": .2}] if text else [], "subtitleText": text}


@pytest.mark.skipif(sys.platform != "darwin", reason="ImageIO caption capture requires macOS")
def test_caption_capture_preserves_full_width_two_row_crop(tmp_path):
    source = tmp_path / "frame.ppm"
    source.write_bytes(b"P6\n1920 1080\n255\n" + b"\x80\x70\x60" * (1920 * 1080))
    captured = ocr._capture_caption_frame(source, 1470.5, native_frame("Two caption rows."), None, tmp_path)
    raw = captured.image.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", raw[16:24]) == (1920, 216)
    assert captured.timestamp == 1470.5
    assert captured.evidence["admitted"] is True
    assert captured.evidence["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.skipif(sys.platform != "darwin", reason="ImageIO caption capture requires macOS")
def test_caption_capture_matches_original_imageio_color_decode(tmp_path):
    source = Path(__file__).parent / "fixtures/caption-rescue-colors.jpg"
    captured = ocr._capture_caption_frame(source, 1, native_frame("Known test colors."), None, tmp_path)
    assert captured.image_digest == "07bbaf6c085e2f6e41bbb17101a7875847c90be6a69bd1a8c79991e800387c9a"


def test_caption_capture_keeps_title_veto_even_with_admitted_roi(tmp_path, monkeypatch):
    source = tmp_path / "frame.jpg"
    source.write_bytes(b"title pixels")
    full = native_frame("lower row")
    full["items"].append({"text": "A TITLE", "confidence": 1, "x": .2, "y": .4, "width": .6,
                          "height": .2, "captionInk": .2})
    monkeypatch.setattr(ocr.subprocess, "run", lambda *a, **kw: pytest.fail("Vetoed frame was cropped"))
    captured = ocr._capture_caption_frame(source, 1, full, native_frame("lower row"), tmp_path)
    assert captured.evidence["admitted"] is False


def install_capture_replay(tmp_path, monkeypatch, *, shifted=False):
    from subzero.caption_rescue import CaptionFrame

    events = []
    monkeypatch.setattr(ocr, "fingerprint", lambda video: "video-proof")

    def capture(source, stamp, full, roi, directory):
        raw = source.read_bytes()
        return CaptionFrame(stamp, source, hashlib.sha256(raw).hexdigest(),
                            {"source_sha256": hashlib.sha256(raw).hexdigest(), "admitted": bool(full["items"])})

    def scan(video, windows, *, fps=2, frame_sink=None, **kwargs):
        events.append(("vision", fps, windows))
        readings = COARSE if fps == 2 else DENSE
        for stamp, text in readings:
            source = tmp_path / f"pixels-{stamp}.png"
            source.write_bytes(f"pixels at {stamp:.3f}".encode())
            actual = stamp + .033 if shifted and stamp == .5 else stamp
            frame_sink(source, actual, native_frame(text), None)
        events.append(("vision-exited", fps))
        return readings

    monkeypatch.setattr(ocr, "_capture_caption_frame", capture, raising=False)
    monkeypatch.setattr(ocr, "_scan_caption_frames", scan)
    return events


def test_frame_replay_corrects_both_cadences_only_after_all_native_calls(tmp_path, monkeypatch):
    events = install_capture_replay(tmp_path, monkeypatch)

    class Client:
        def read_frames(self, video_key, frames):
            assert events[-1] == ("vision-exited", 10)
            assert len([event for event in events if event[0] == "vision"]) == 2
            events.append(("qwen",))
            return {frame.timestamp: "I _ got you." if frame.evidence["admitted"] else "background junk"
                    for frame in frames}

    coarse, dense = ocr._rescue_caption_frames("video", COARSE, DENSE, 3, [(0, 2)], [(0, 2)], Client())
    assert coarse == [(stamp, "I _ got you." if text else "") for stamp, text in COARSE]
    assert dense == DENSE
    assert events[-1] == ("qwen",)


def test_frame_replay_rejects_nearest_pts_substitution(tmp_path, monkeypatch):
    install_capture_replay(tmp_path, monkeypatch, shifted=True)

    class Client:
        def read_frames(self, *args):
            pytest.fail("Missing exact frame reached model")

    with pytest.raises(RuntimeError, match="exact.*frame"):
        ocr._rescue_caption_frames("video", COARSE, DENSE, 3, [(0, 2)], [(0, 2)], Client())


def test_frame_replay_cannot_erase_confirmed_coarse_caption(tmp_path, monkeypatch):
    install_capture_replay(tmp_path, monkeypatch)

    class Client:
        def read_frames(self, video_key, frames):
            return {frame.timestamp: "" for frame in frames}

    with pytest.raises(RuntimeError, match="confirmed.*caption"):
        ocr._rescue_caption_frames("video", COARSE, DENSE, 3, [(0, 2)], [(0, 2)], Client())


def test_rescue_window_does_not_include_unrelated_preceding_stable_cue():
    cues = [Cue("00:00:00,000", "00:00:30,000", "A separate stable sentence."),
            Cue("00:00:50,000", "00:00:50,100", "Go on, please."),
            Cue("00:00:50,100", "00:00:50,200", "Goron, please."),
            Cue("00:00:50,200", "00:00:50,300", "Go on, please.")]
    windows = ocr._caption_rescue_windows([], cues, [], 60)
    assert windows == [(49.25, 51.05)]


def test_frame_replay_reports_progress_after_chunks_and_model_frames(tmp_path, monkeypatch):
    install_capture_replay(tmp_path, monkeypatch)
    updates = []

    class Client:
        def read_frames(self, video_key, frames, *, progress=None):
            readings = {}
            for index, frame in enumerate(frames, 1):
                readings[frame.timestamp] = "I _ got you." if frame.evidence["admitted"] else ""
                progress(index, len(frames))
            return readings

    ocr._rescue_caption_frames("video", COARSE, DENSE, 3, [(0, 2)], [(0, 2)], Client(),
                               progress=lambda done, total: updates.append((done, total)))
    assert updates[:3] == [(0, 22), (1, 22), (2, 22)]
    assert updates[-1] == (22, 22)
    assert [done for done, _ in updates] == sorted(done for done, _ in updates)
