import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from subzero.convert import Cue
from subzero.ocr import (
    clean_ocr_text,
    cluster_ocr_detections,
    cue_start_seconds,
    extract_and_ocr_gaps,
    fill_subtitle_gaps,
    find_speech_gaps,
    fmt_srt_time,
)


def test_fmt_srt_time():
    assert fmt_srt_time(0.0) == "00:00:00,000"
    assert fmt_srt_time(65.5) == "00:01:05,500"
    assert fmt_srt_time(3661.123) == "01:01:01,123"


def test_clean_ocr_text():
    raw = (
        "“Andy is looking for the idol.”\n"
        "TUKU TRIBE\n"
        " ạạạ\n"
        "DAY 1\n"
        "He has no idea what we are planning.\n"
    )
    cleaned = clean_ocr_text(raw)
    lines = cleaned.splitlines()
    assert "Andy is looking for the idol." in lines
    assert "He has no idea what we are planning." in lines
    assert "TUKU TRIBE" not in lines
    assert "DAY 1" not in lines
    assert "ạạạ" not in lines


def test_cluster_ocr_detections_empty():
    assert cluster_ocr_detections([]) == []


def test_cluster_ocr_detections_merge_and_split():
    detections = [
        (10.0, "We need to vote"),
        (11.0, "We need to vote out Andy"),
        (12.0, "We need to vote out Andy"),
        (25.0, "Are you with me?"),
    ]
    cues = cluster_ocr_detections(detections)
    assert len(cues) == 2

    c1, c2 = cues
    assert c1.text == "We need to vote out Andy"
    assert c1.start == "00:00:10,000"
    assert c1.end == "00:00:13,200"

    assert c2.text == "Are you with me?"
    assert c2.start == "00:00:25,000"
    assert c2.end == "00:00:26,400"


def test_cue_start_seconds():
    c = Cue(start="00:01:30,500", end="00:01:35,000", text="Test")
    assert abs(cue_start_seconds(c) - 90.5) < 0.001


def test_find_speech_gaps():
    subtitle = (
        "1\n00:00:00,000 --> 00:00:05,000\nHello world\n\n"
        "2\n00:00:20,000 --> 00:00:25,000\nSecond dialogue\n\n"
    )
    fake_ref = {
        "speech": [
            (0.0, 4.8),    # covered by cue 1
            (10.0, 15.0),  # gap (no subtitles)
            (20.0, 24.5),  # covered by cue 2
            (30.0, 30.5),  # < 1.0s, skipped
        ]
    }
    with patch("subzero.ocr.build_reference", return_value=fake_ref):
        gaps = find_speech_gaps("dummy_video.mkv", subtitle)
        assert gaps == [(10.0, 15.0)]


def test_extract_and_ocr_gaps(tmp_path):
    gaps = [(10.0, 12.0)]
    mock_ocr_output = json.dumps([
        {"file": "f_001.jpg", "items": [], "subtitleText": "Don't tell anyone."},
        {"file": "f_002.jpg", "items": [], "subtitleText": "Don't tell anyone."},
    ])

    def fake_subprocess_run(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if cmd[0] == "ffmpeg":
            # generate dummy jpg files in gap folder
            out_pattern = cmd[-1]
            gap_dir = Path(out_pattern).parent
            gap_dir.mkdir(parents=True, exist_ok=True)
            (gap_dir / "f_001.jpg").touch()
            (gap_dir / "f_002.jpg").touch()
            res.stdout = ""
            return res
        if "vision_ocr" in cmd[0]:
            res.stdout = mock_ocr_output
            return res
        return res

    with patch("subzero.ocr.get_vision_ocr_bin", return_value="/usr/local/bin/vision_ocr"), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        cues = extract_and_ocr_gaps("dummy.mkv", gaps, tmp_dir=tmp_path)
        assert len(cues) == 1
        assert cues[0].text == "Don't tell anyone."
        assert cues[0].start == "00:00:10,000"


def test_fill_subtitle_gaps_dry_run(tmp_path):
    sub_file = tmp_path / "test.srt"
    sub_file.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nPrimeira fala\n\n",
        encoding="utf-8",
    )

    with patch("subzero.ocr.find_speech_gaps", return_value=[(10.0, 14.0)]), \
         patch("subzero.ocr.extract_and_ocr_gaps", return_value=[
             Cue(start="00:00:10,000", end="00:00:13,000", text="Secret whisper"),
         ]), \
         patch("subzero.ocr.translate_cues", return_value=[
             Cue(start="00:00:10,000", end="00:00:13,000", text="Sussurro secreto"),
         ]):
        rep = fill_subtitle_gaps(
            video="dummy.mkv",
            subtitle_path=sub_file,
            target_lang="pt-BR",
            dry_run=True,
        )
        assert rep.total_gaps == 1
        assert rep.cues_recovered == 1
        assert rep.cues[0].text == "Sussurro secreto"
        # file remains untouched in dry run
        content = sub_file.read_text(encoding="utf-8")
        assert "Sussurro secreto" not in content


def test_fill_subtitle_gaps_apply(tmp_path):
    sub_file = tmp_path / "test.srt"
    sub_file.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nPrimeira fala\n\n",
        encoding="utf-8",
    )

    with patch("subzero.ocr.find_speech_gaps", return_value=[(10.0, 14.0)]), \
         patch("subzero.ocr.extract_and_ocr_gaps", return_value=[
             Cue(start="00:00:10,000", end="00:00:13,000", text="Secret whisper"),
         ]), \
         patch("subzero.ocr.translate_cues", return_value=[
             Cue(start="00:00:10,000", end="00:00:13,000", text="Sussurro secreto"),
         ]):
        rep = fill_subtitle_gaps(
            video="dummy.mkv",
            subtitle_path=sub_file,
            target_lang="pt-BR",
            dry_run=False,
            backup=True,
        )
        assert rep.cues_recovered == 1
        # Backup created
        bak_file = tmp_path / "test.srt.bak"
        assert bak_file.exists()
        # Original updated and sorted
        content = sub_file.read_text(encoding="utf-8")
        assert "Primeira fala" in content
        assert "Sussurro secreto" in content
