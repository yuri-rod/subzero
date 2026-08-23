import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from subzero.extract import (
    VIDEO_EXTENSIONS,
    SubtitleStream,
    ToolError,
    collect_videos,
    extract_from_video,
    extract_stream,
    is_video,
    list_subtitle_streams,
    require_ffmpeg,
)


def test_is_video_recognises_common_containers():
    assert is_video("show.mp4")
    assert is_video("show.MKV")
    assert is_video("clip.mlv")
    assert not is_video("show.srt")


def test_collect_videos(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.srt").write_text("nope")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.mkv").write_bytes(b"y")
    found = collect_videos([tmp_path])
    names = sorted(p.name for p in found)
    assert names == ["a.mp4", "c.mkv"]


def test_require_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr("subzero.extract.shutil.which", lambda n: None)
    with pytest.raises(ToolError, match="ffmpeg"):
        require_ffmpeg()


def _ffprobe_payload(streams):
    return json.dumps({"streams": streams})


class TestListStreams:
    def test_parses_ffprobe_json(self, tmp_path, monkeypatch):
        video = tmp_path / "a.mp4"
        video.write_bytes(b"fake")
        monkeypatch.setattr(
            "subzero.extract.require_ffmpeg",
            lambda: ("ffmpeg", "ffprobe"),
        )

        payload = _ffprobe_payload([
            {
                "index": 2,
                "codec_name": "subrip",
                "tags": {"language": "eng", "title": "English"},
                "disposition": {"default": 1},
            },
            {
                "index": 3,
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "eng"},
            },
        ])

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")

        monkeypatch.setattr("subzero.extract._run", fake_run)
        streams = list_subtitle_streams(video)
        assert len(streams) == 2
        assert streams[0].language == "eng"
        assert streams[0].is_text is True
        assert streams[1].is_text is False
        assert "English" in streams[0].label


def test_extract_from_video_dry(tmp_path, monkeypatch):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        "subzero.extract.require_ffmpeg",
        lambda: ("ffmpeg", "ffprobe"),
    )
    streams = [
        SubtitleStream(index=2, codec="subrip", language="eng", is_text=True),
        SubtitleStream(index=3, codec="subrip", language="por", is_text=True),
    ]
    monkeypatch.setattr("subzero.extract.list_subtitle_streams", lambda p: streams)

    res = extract_from_video(video, dry=True, all_streams=True)
    assert len(res.outputs) == 2
    assert all(o.endswith(".srt") for o in res.outputs)
    assert not any(Path(o).exists() for o in res.outputs)


def test_extract_language_filter(tmp_path, monkeypatch):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        "subzero.extract.require_ffmpeg",
        lambda: ("ffmpeg", "ffprobe"),
    )
    streams = [
        SubtitleStream(index=2, codec="subrip", language="eng", is_text=True),
        SubtitleStream(index=3, codec="subrip", language="por", is_text=True),
    ]
    monkeypatch.setattr("subzero.extract.list_subtitle_streams", lambda p: streams)
    res = extract_from_video(video, dry=True, languages=("por",), all_streams=True)
    assert len(res.outputs) == 1
    assert ".por." in res.outputs[0]


def test_extract_writes_via_ffmpeg(tmp_path, monkeypatch):
    video = tmp_path / "show.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        "subzero.extract.require_ffmpeg",
        lambda: ("ffmpeg", "ffprobe"),
    )
    stream = SubtitleStream(index=2, codec="subrip", language="eng", is_text=True)
    monkeypatch.setattr(
        "subzero.extract.list_subtitle_streams", lambda p: [stream]
    )

    srt_body = "1\n00:00:01,000 --> 00:00:02,000\n[music]\n\n"

    def fake_run(cmd, **kw):
        # last arg is output path
        out = Path(cmd[-1])
        out.write_text(srt_body, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("subzero.extract._run", fake_run)

    fixed_paths = []

    def fake_fix(path, opts=None):
        fixed_paths.append(str(path))
        text = Path(path).read_text(encoding="utf-8")
        Path(path).write_text(text.replace("[music]", "gone"), encoding="utf-8")
        return type("R", (), {"changed": True, "dropped": 1, "rewrapped": 0, "cues": 0})()

    res = extract_from_video(video, fix=fake_fix)
    assert len(res.outputs) == 1
    assert Path(res.outputs[0]).exists()
    assert res.fixed
    assert "[music]" not in Path(res.fixed[0]).read_text(encoding="utf-8")


def test_extract_stream_dry_path(tmp_path, monkeypatch):
    video = tmp_path / "a.mkv"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        "subzero.extract.require_ffmpeg",
        lambda: ("ffmpeg", "ffprobe"),
    )
    s = SubtitleStream(index=1, codec="ass", language="en", is_text=True)
    out = extract_stream(video, s, streams=[s], fmt="vtt", dry=True)
    assert str(out).endswith(".vtt")
    assert not out.exists()


def test_video_extensions_include_mlv_mp4():
    assert ".mlv" in VIDEO_EXTENSIONS
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".mkv" in VIDEO_EXTENSIONS
