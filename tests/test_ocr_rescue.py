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


def _gap_run(monkeypatch, tmp_path, caption, translated):
    subtitle = tmp_path / "source.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nExisting dialogue.\n", encoding="utf-8")
    monkeypatch.setattr(ocr, "find_caption_gaps", lambda *a, **kw: [(2, 4)])
    monkeypatch.setattr(ocr, "extract_and_ocr_gaps", lambda *a, **kw: [
        Cue("00:00:02,000", "00:00:03,000", caption)])

    def translate(cues, *, client, **kwargs):
        return [Cue(cues[0].start, cues[0].end, translated)]

    monkeypatch.setattr(ocr, "translate_cues", translate)
    return ocr.fill_subtitle_gaps("video", subtitle, target_lang="pt-BR", dry_run=True,
                                  translation_client=object())


def test_identical_proper_noun_caption_is_kept(tmp_path, monkeypatch):
    report = _gap_run(monkeypatch, tmp_path, "HOLLYWOOD, CALIFORNIA", "HOLLYWOOD, CALIFORNIA")
    assert report.cues_recovered == 1
    assert report.cues[0].text == "HOLLYWOOD, CALIFORNIA"


def test_identical_sentence_with_function_words_still_fails(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="untranslated English"):
        _gap_run(monkeypatch, tmp_path, "You are safe tonight.", "You are safe tonight.")


def test_identical_phrase_with_preposition_still_fails(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="untranslated English"):
        _gap_run(monkeypatch, tmp_path, "Previously on Survivor", "Previously on Survivor")


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
    coarse = [(stamp, text.replace(" _", "").replace("I got you.", "I got you now."))
              for stamp, text in COARSE]
    dense = [(stamp, text.replace(" _", "").replace("I got you.", "I got you now."))
             for stamp, text in DENSE]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    monkeypatch.setattr(ocr, "_rescue_caption_frames", lambda *a: pytest.fail("Stable caption was rescued"), raising=False)
    cues = ocr.refine_caption_timing("video", coarse, 3, caption_rescue=object())
    assert [cue.text for cue in cues] == ["I got you now."]


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
    assert [cue.text for cue in cues] == ["I _ got you."]
    assert collected and all(end < 3 for _, end in collected)


def test_unresolved_flicker_after_rescue_still_fails_original_gate(monkeypatch):
    coarse = [(0, ""), (.5, "Go on with it please."), (1, "Go on with it please."), (1.5, "")]
    dense = [(.4, ""), (.5, "Go on with it please."), (.6, "Goron with it please."),
             (.7, "Go on with it please."), (.8, "Goron with it please."), (.9, "")]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    monkeypatch.setattr(ocr, "_rescue_caption_frames", lambda *a: (coarse, dense), raising=False)
    with pytest.raises(RuntimeError, match="Unstable OCR"):
        ocr.refine_caption_timing("video", coarse, 2, caption_rescue=object())


def test_short_flicker_is_dropped_instead_of_rescued(monkeypatch):
    coarse = [(0, ""), (.5, "Go on, please."), (1, "Go on, please."), (1.5, "")]
    dense = [(.4, ""), (.5, "Go on, please."), (.6, "Goron, please."), (.7, "Go on, please."), (.8, "")]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    monkeypatch.setattr(ocr, "_rescue_caption_frames",
                        lambda *a: pytest.fail("Short caption was rescued"), raising=False)
    assert ocr.refine_caption_timing("video", coarse, 2, caption_rescue=object()) == []


@pytest.mark.parametrize("first, changed", [
    ("I go.", "go."), ("Wait.", "Walt."), ("Sam.", "Pam."), ("7.", "8."),
    ("I can.", "I can't."),
])
def test_short_ambiguity_is_dropped_without_rescue(monkeypatch, first, changed):
    coarse = [(0, ""), (.5, first), (1, first), (1.5, "")]
    dense = [(.4, ""), (.5, first), (.6, first), (.7, changed), (.8, first), (.9, "")]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    monkeypatch.setattr(ocr, "_rescue_caption_frames",
                        lambda *a: pytest.fail("Short caption was rescued"))
    assert ocr.refine_caption_timing("video", coarse, 2, caption_rescue=object()) == []


def test_censored_flanks_survive_short_middle_variant_without_rescue(monkeypatch):
    coarse = [(0, ""), (.5, "I _ go."), (1, "I _ go."), (1.5, "")]
    dense = [(.4, ""), (.5, "I _ go."), (.6, "I _ go."), (.7, "I go."), (.8, "I _ go."), (.9, "")]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    monkeypatch.setattr(ocr, "_rescue_caption_frames",
                        lambda *a: pytest.fail("Short variant was rescued"))
    cues = ocr.refine_caption_timing("video", coarse, 2, caption_rescue=object())
    assert [cue.text for cue in cues] == ["I _ go.", "I _ go."]


def test_censored_position_ambiguity_still_needs_review(monkeypatch):
    coarse = [(0, ""), (.5, "I _ go."), (1, "I _ go."), (1.5, "")]
    dense = [(.4, ""), (.5, "I _ go."), (.6, "I _ go."), (.7, "_ I go."), (.8, "I _ go."), (.9, "")]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *a, **kw: dense)
    selected = []

    def rescue(video, observed, scanned, duration, windows, intervals, client):
        selected.extend(intervals)
        return observed, scanned

    monkeypatch.setattr(ocr, "_rescue_caption_frames", rescue)
    with pytest.raises(RuntimeError, match="Unstable OCR"):
        ocr.refine_caption_timing("video", coarse, 2, caption_rescue=object())
    assert any(start <= .7 < end for start, end in selected)
    assert dense[3] == (.7, "_ I go.")


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
            Cue("00:00:50,000", "00:00:50,100", "Go on with it please."),
            Cue("00:00:50,100", "00:00:50,200", "Goron with it please."),
            Cue("00:00:50,200", "00:00:50,300", "Go on with it please.")]
    windows = ocr._caption_rescue_windows([], cues, [], 60)
    assert windows == [(49.25, 51.05)]


def test_rescue_window_ignores_short_only_instability():
    cues = [Cue("00:00:50,000", "00:00:50,100", "Go on, please."),
            Cue("00:00:50,100", "00:00:50,200", "Goron, please."),
            Cue("00:00:50,200", "00:00:50,300", "Go on, please.")]
    assert ocr._caption_rescue_windows([], cues, [], 60) == []


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


def test_frame_replay_uses_the_same_roi_admission_for_both_capture_cadences(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "fingerprint", lambda video: "video-proof")
    monkeypatch.setattr(ocr, "crop_caption_image", lambda raw: b"cropped " + raw)

    def scan(video, windows, *, fps, retry_all, frame_sink):
        source = tmp_path / "frame.jpg"
        source.write_bytes(b"same exact source pixels")
        roi = native_frame("Bring your things.") if retry_all else None
        frame_sink(source, .5, native_frame(""), roi)
        return [(.5, "Bring your things." if roi else "")]

    monkeypatch.setattr(ocr, "_scan_caption_frames", scan)

    class Client:
        def read_frames(self, video_key, frames):
            assert len(frames) == 1
            assert frames[0].evidence["admitted"] is True
            return {.5: "Bring your things."}

    assert ocr._rescue_caption_frames("video", [(.5, "")], [(.5, "Bring your things.")],
                                     1, [(0, 1)], [(0, 1)], Client()) == (
        [(.5, "Bring your things.")], [(.5, "Bring your things.")])


@pytest.mark.parametrize("conflict", ["pixels", "admission"])
def test_frame_replay_still_rejects_real_same_timestamp_evidence_conflicts(tmp_path, monkeypatch, conflict):
    monkeypatch.setattr(ocr, "fingerprint", lambda video: "video-proof")
    monkeypatch.setattr(ocr, "crop_caption_image", lambda raw: b"cropped " + raw)

    def scan(video, windows, *, fps, retry_all, frame_sink):
        source = tmp_path / "frame.jpg"
        source.write_bytes(b"different pixels" if fps == 10 and conflict == "pixels" else b"same pixels")
        full = native_frame("Bring your things.")
        if fps == 10 and conflict == "admission":
            full["items"].append({"text": "TITLE", "confidence": 1, "x": .2, "y": .4,
                                  "width": .6, "height": .2, "captionInk": .2})
        frame_sink(source, .5, full, None)
        return [(.5, "Bring your things.")]

    monkeypatch.setattr(ocr, "_scan_caption_frames", scan)

    class Client:
        def read_frames(self, *args):
            pytest.fail("Conflicting evidence reached recognition")

    with pytest.raises(RuntimeError, match=r"different evidence.*0\.500"):
        ocr._rescue_caption_frames("video", [(.5, "Bring your things.")], [(.5, "Bring your things.")],
                                  1, [(0, 1)], [(0, 1)], Client())
