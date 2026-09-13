from fractions import Fraction
import re
import shutil
import subprocess

import pytest

from subzero import ocr


@pytest.mark.parametrize("rate,fps", [
    ("30", 10),
    ("30000/1001", 10),
    ("30000/1001", 2),
    ("24000/1001", 10),
])
def test_scan_selects_first_eligible_input_pts(tmp_path, monkeypatch, rate, fps):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg is required for the timestamp selection regression")
    video = tmp_path / "millisecond-pts.mkv"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i",
        f"color=size=32x32:rate={rate}:duration=2", "-an", "-c:v", "ffv1", "-threads", "1", str(video),
    ], check=True, capture_output=True, timeout=30)
    decoded = subprocess.run([
        ffmpeg, "-hide_banner", "-nostdin", "-hwaccel", "none", "-i", str(video),
        "-an", "-sn", "-vf", "showinfo", "-fps_mode", "vfr", "-f", "null", "-",
    ], check=True, capture_output=True, text=True, timeout=30)
    timebase = Fraction(re.search(r"config in time_base:\s*(\d+/\d+)", decoded.stderr).group(1))
    input_pts = [int(pts) for pts in re.findall(r"\bpts:\s*(-?\d+)\s+pts_time:", decoded.stderr)]
    assert timebase == Fraction(1, 1000)
    expected = []
    for pts in input_pts:
        if not expected or (pts - expected[-1]) * timebase >= Fraction(1, fps):
            expected.append(pts)

    def readings(binary, frames, caption_region=False):
        return {frame.name: {"file": str(frame), "subtitleText": ""} for frame in frames}

    monkeypatch.setattr(ocr, "_vision_frames", readings)
    detections = ocr._scan_caption_frames(video, [(0, 2)], ocr_bin="unused", fps=fps, tmp_dir=tmp_path)
    actual = [timestamp for timestamp, _ in detections]
    assert actual == pytest.approx([float(pts * timebase) for pts in expected], abs=1e-9)
    assert len(actual) == len(set(actual))
    assert all(second > first for first, second in zip(actual, actual[1:]))
    assert all(timestamp in {float(pts * timebase) for pts in input_pts} for timestamp in actual)
    if rate in ("30", "30000/1001") and fps == 10:
        assert actual[:4] == pytest.approx([0, 0.1, 0.2, 0.3], abs=1e-9)
