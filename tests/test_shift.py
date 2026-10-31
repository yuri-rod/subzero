import pytest

from subzero.shift import calculate_fps_factor, shift_timestamps, shift_file

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,000
First line

2
00:01:00.500 --> 00:01:05.000
Second line
"""

def test_shift_timestamps_positive():
    shifted, count = shift_timestamps(SAMPLE_SRT, 2.5)
    assert count == 2
    assert "00:00:03,500 --> 00:00:06,500" in shifted
    assert "00:01:03.000 --> 00:01:07.500" in shifted

def test_shift_timestamps_negative_clips_to_zero():
    shifted, count = shift_timestamps(SAMPLE_SRT, -2.0)
    assert count == 2
    assert "00:00:00,000 --> 00:00:02,000" in shifted

def test_fps_calculation_and_scaling():
    factor = calculate_fps_factor(25, 23.976)
    assert round(factor, 4) == round(25.0 / 23.976, 4)
    shifted, count = shift_timestamps(SAMPLE_SRT, delta_seconds=0.0, scale_factor=2.0)
    assert count == 2
    assert "00:00:02,000 --> 00:00:08,000" in shifted

def test_shift_file_in_place(tmp_path):
    p = tmp_path / "test.srt"
    p.write_text(SAMPLE_SRT, encoding="utf-8")
    out, count = shift_file(p, 1.0)
    assert out == p
    assert count == 2
    content = p.read_text(encoding="utf-8")
    assert "00:00:02,000 --> 00:00:05,000" in content


@pytest.mark.parametrize("options", [
    {"scale_factor": 0}, {"scale_factor": -1}, {"scale_factor": float("nan")},
    {"scale_factor": float("inf")}, {"delta_seconds": float("nan")},
    {"delta_seconds": float("inf")}, {"from_fps": "25"}, {"to_fps": "24"},
    {"from_fps": "nan", "to_fps": "24"}, {"from_fps": "25", "to_fps": "inf"},
])
def test_invalid_shift_keeps_original(tmp_path, options):
    subtitle = tmp_path / "movie.srt"
    subtitle.write_text(SAMPLE_SRT, encoding="utf-8")
    original = subtitle.read_bytes()
    with pytest.raises(ValueError):
        shift_file(subtitle, **options)
    assert subtitle.read_bytes() == original


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16", "utf-16-le", "utf-16-be"])
def test_shift_preserves_encoded_dialogue(tmp_path, encoding):
    subtitle = tmp_path / "movie.srt"
    subtitle.write_bytes(SAMPLE_SRT.replace("First line", "Café, please.").encode(encoding))
    shift_file(subtitle, delta_seconds=1)
    assert "Café, please." in subtitle.read_text(encoding="utf-8")
    assert "00:00:02,000 --> 00:00:05,000" in subtitle.read_text(encoding="utf-8")


def test_utf8_dialogue_with_a_nul_is_not_mistaken_for_utf16(tmp_path):
    subtitle = tmp_path / "movie.srt"
    subtitle.write_text(SAMPLE_SRT.replace("First line", "Hello\x00 world."), encoding="utf-8")
    _, count = shift_file(subtitle, delta_seconds=1)
    assert count == 2
    assert "Hello\x00 world." in subtitle.read_text(encoding="utf-8")


def test_shift_file_preserves_first_backup_and_rejects_symlinks(tmp_path):
    subtitle = tmp_path / "movie.srt"
    subtitle.write_text(SAMPLE_SRT, encoding="utf-8")
    backup_dir = tmp_path / "backups"

    shift_file(subtitle, delta_seconds=1.0, backup_dir=backup_dir)
    shift_file(subtitle, delta_seconds=1.0, backup_dir=backup_dir)

    assert "00:00:01,000 --> 00:00:04,000" in (backup_dir / "movie.srt").read_text(encoding="utf-8")
    assert "00:00:03,000 --> 00:00:06,000" in subtitle.read_text(encoding="utf-8")

    link = tmp_path / "link.srt"
    try:
        link.symlink_to(subtitle)
    except OSError:
        pytest.skip("Creating symlinks requires platform permissions")
    with pytest.raises(OSError, match="symbolic link"):
        shift_file(link, delta_seconds=1.0)


def test_shift_timestamps_with_piecewise_segments():
    from subzero.timing import Segment

    srt = """1
00:01:00,000 --> 00:01:03,000
Act 1 cue

2
00:15:00,000 --> 00:15:04,000
Act 2 cue
"""
    segments = [
        Segment(start=0.0, end=600.0, offset=-2.0, scale=1.0),
        Segment(start=600.0, end=1800.0, offset=+10.0, scale=1.0),
    ]
    shifted, count = shift_timestamps(srt, segments=segments)
    assert count == 2
    # Act 1 cue (60s) shifted by -2s -> 58s
    assert "00:00:58,000 --> 00:01:01,000" in shifted
    # Act 2 cue (900s) shifted by +10s -> 910s
    assert "00:15:10,000 --> 00:15:14,000" in shifted


def test_shift_timestamps_piecewise_enforces_monotonic_cue_order():
    from subzero.timing import Segment

    srt = """1
00:09:58,000 --> 00:10:00,000
Boundary 1

2
00:10:01,000 --> 00:10:03,000
Boundary 2
"""
    # Segment 1 has +10s offset (cues land at 10:08 - 10:10)
    # Segment 2 has -15s offset (cues without ordering would land at 09:46 - 09:48)
    segments = [
        Segment(start=0.0, end=600.0, offset=10.0),
        Segment(start=600.0, end=1200.0, offset=-15.0),
    ]
    shifted, count = shift_timestamps(srt, segments=segments)
    assert count == 2
    assert "00:10:08,000 --> 00:10:10,000" in shifted
    # Second cue must be ordered after 10:10
    assert "00:10:10,050 --> 00:10:10,150" in shifted
