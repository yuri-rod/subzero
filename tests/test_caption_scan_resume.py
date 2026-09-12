import subprocess

import pytest

from subzero import ocr
from subzero.caption_scan_cache import CaptionScanCache


@pytest.mark.parametrize("changed", ["video", "vision", "macos", "ffmpeg", "recognition"])
def test_native_scan_identity_invalidates_changed_inputs(tmp_path, monkeypatch, changed):
    video = tmp_path / "video.mkv"
    video.write_bytes(b"original video")
    binary = tmp_path / "vision_ocr"
    binary.write_bytes(b"native executable")
    versions = {"macos": "26.0", "ffmpeg": "ffmpeg version test"}
    monkeypatch.setattr(ocr, "get_vision_ocr_bin", lambda: str(binary))
    monkeypatch.setattr(ocr.platform, "mac_ver", lambda: (versions["macos"], (), ""))
    monkeypatch.setattr(ocr.subprocess, "run", lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, 0, versions["ffmpeg"], ""))
    root = tmp_path / "cache"
    windows = [(0, 1)]
    cache = ocr._caption_scan_cache(video, root)
    cache.write(windows, [(0.1, "Keep this secret.")], fps=2)

    if changed == "video":
        video.write_bytes(b"different video")
    elif changed == "vision":
        binary.write_bytes(b"updated executable")
    elif changed == "recognition":
        monkeypatch.setattr(ocr, "CAPTION_SCAN_VERSION", ocr.CAPTION_SCAN_VERSION + 1)
    else:
        versions[changed] += " changed"

    assert ocr._caption_scan_cache(video, root).read(windows, fps=2) is None


def test_full_scan_resumes_after_an_interrupted_window(tmp_path, monkeypatch):
    cache = CaptionScanCache(tmp_path, "video", "recognition")
    monkeypatch.setattr(ocr, "_caption_scan_cache", lambda *args: cache, raising=False)
    scanned = []

    def scan(video, windows, **kwargs):
        window = windows[0]
        scanned.append(window)
        if window[0] == 119 and scanned.count(window) == 1:
            raise RuntimeError("Vision interrupted")
        return [(window[0] + 0.01, ""), (window[1] - 0.01, "")]

    monkeypatch.setattr(ocr, "_scan_caption_frames", scan)
    with pytest.raises(RuntimeError, match="Vision interrupted"):
        ocr.extract_all_captions("video.mkv", 240, cache_dir=tmp_path)

    assert ocr.extract_all_captions("video.mkv", 240, cache_dir=tmp_path) == []
    assert scanned == [(0, 121), (119, 240), (119, 240)]


def test_cached_dense_evidence_still_rejects_a_lost_caption(tmp_path, monkeypatch):
    cache = CaptionScanCache(tmp_path, "video", "recognition")
    coarse = [(9.5, ""), (10, "Keep this secret."), (10.5, "Keep this secret."), (11, "")]
    calls = []

    def scan(video, windows, **kwargs):
        calls.append(windows)
        return [(9.6, ""), (10.1, ""), (10.6, ""), (11.1, "")]

    monkeypatch.setattr(ocr, "_scan_caption_frames", scan)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="lost a confirmed caption"):
            ocr.refine_caption_timing("video.mkv", coarse, 30, scan_cache=cache)
    assert len(calls) == 1


def test_cached_dense_scan_matches_fresh_timing_and_reports_progress(tmp_path, monkeypatch):
    cache = CaptionScanCache(tmp_path, "video", "recognition")
    coarse = [(9.5, ""), (10, "Keep this secret."), (10.5, "Keep this secret."), (11, "")]
    dense = [(9.6, ""), (9.7, "Keep this secret."), (10.2, "Keep this secret."),
             (10.7, "Keep this secret."), (10.8, ""), (11.1, "")]
    monkeypatch.setattr(ocr, "_scan_caption_frames", lambda *args, **kwargs: dense)
    fresh = ocr.refine_caption_timing("video.mkv", coarse, 30)
    first = ocr.refine_caption_timing("video.mkv", coarse, 30, scan_cache=cache)

    def no_rescan(*args, **kwargs):
        raise AssertionError("Completed dense windows must be reused")

    progress = []
    monkeypatch.setattr(ocr, "_scan_caption_frames", no_rescan)
    second = ocr.refine_caption_timing("video.mkv", coarse, 30, scan_cache=cache,
                                      progress=lambda *step: progress.append(step))
    assert first == second == fresh
    assert progress[-1] == (100, 100)
