import pytest

from subzero.convert import (
    convert_file,
    convert_text,
    detect_format,
    parse_ass,
    parse_srt,
    parse_vtt,
    to_ass_time,
    to_srt_time,
    to_vtt_time,
)

SRT = """1
00:00:01,000 --> 00:00:03,000
[tense music playing]

2
00:00:03,500 --> 00:00:05,000
- MITCH: Yes, sir. - MAN: So I have a question.

"""

VTT = """WEBVTT

1
00:00:01.000 --> 00:00:03.000
Hello world

2
00:00:03.500 --> 00:00:05.000
Second cue

"""

ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello world
Dialogue: 0,0:00:03.50,0:00:05.00,Default,,0,0,0,,{\\i1}Second{\\i0} cue
Comment: 0,0:00:06.00,0:00:07.00,Default,,0,0,0,,ignored
"""


class TestTimestamps:
    def test_srt_passthrough(self):
        assert to_srt_time("00:00:01,000") == "00:00:01,000"

    def test_vtt_dot(self):
        assert to_srt_time("00:00:01.000") == "00:00:01,000"
        assert to_vtt_time("00:00:01,000") == "00:00:01.000"

    def test_ass_centiseconds(self):
        assert to_srt_time("0:00:01.00") == "00:00:01,000"
        assert to_ass_time("00:00:01,500") == "0:00:01.50"

    def test_bad_timestamp(self):
        with pytest.raises(ValueError):
            to_srt_time("not-a-time")


class TestParse:
    def test_parse_srt(self):
        cues = parse_srt(SRT)
        assert len(cues) == 2
        assert cues[0].text == "[tense music playing]"

    def test_parse_vtt(self):
        cues = parse_vtt(VTT)
        assert len(cues) == 2
        assert cues[0].text == "Hello world"
        assert cues[0].start == "00:00:01,000"

    def test_parse_ass_strips_overrides_and_comments(self):
        cues = parse_ass(ASS)
        assert len(cues) == 2
        assert cues[1].text == "Second cue"


class TestConvert:
    def test_srt_to_vtt(self):
        res = convert_text(SRT, "vtt", source="srt")
        assert res.cues == 2
        assert res.text.startswith("WEBVTT")
        assert "00:00:01.000 -->" in res.text

    def test_vtt_to_srt(self):
        res = convert_text(VTT, "srt", source="vtt")
        assert res.cues == 2
        assert "00:00:01,000 -->" in res.text

    def test_srt_to_ass_and_back(self):
        ass = convert_text(VTT, "ass", source="vtt")
        back = convert_text(ass.text, "srt", source="ass")
        assert back.cues == 2
        assert "Hello world" in back.text

    def test_detect_from_content(self):
        assert detect_format("x", VTT) == "vtt"
        assert detect_format("x", ASS) == "ass"
        assert detect_format("x", SRT) == "srt"

    def test_detect_from_extension(self, tmp_path):
        p = tmp_path / "a.vtt"
        p.write_text(VTT, encoding="utf-8")
        assert detect_format(p) == "vtt"

    def test_convert_file_writes(self, tmp_path):
        src = tmp_path / "a.srt"
        src.write_text(SRT, encoding="utf-8")
        res = convert_file(src, "vtt")
        out = tmp_path / "a.vtt"
        assert res.path == str(out)
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("WEBVTT")

    def test_convert_file_dry(self, tmp_path):
        src = tmp_path / "a.srt"
        src.write_text(SRT, encoding="utf-8")
        res = convert_file(src, "vtt", dry=True)
        assert not (tmp_path / "a.vtt").exists()
        assert res.cues == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no subtitle"):
            convert_text("not subtitles", "srt", source="srt")

    def test_unsupported_target(self):
        with pytest.raises(ValueError):
            convert_text(SRT, "xyz", source="srt")
