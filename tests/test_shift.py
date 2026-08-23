from subzero.shift import shift_timestamps, shift_file

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

def test_shift_file_in_place(tmp_path):
    p = tmp_path / "test.srt"
    p.write_text(SAMPLE_SRT, encoding="utf-8")
    out, count = shift_file(p, 1.0)
    assert out == p
    assert count == 2
    content = p.read_text(encoding="utf-8")
    assert "00:00:02,000 --> 00:00:05,000" in content
