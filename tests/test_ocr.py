import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subzero.convert import Cue, parse_srt
from subzero import ocr
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


def test_ocr_resolver_rebuilds_stale_binary_from_packaged_source(tmp_path, monkeypatch):
    module = tmp_path / "src/subzero/ocr.py"
    source = module.with_name("vision_ocr.swift")
    source.parent.mkdir(parents=True)
    source.write_text("packaged source")
    repo_bin = tmp_path / "tools/vision_ocr"
    repo_bin.parent.mkdir()
    repo_bin.write_text("stale binary")
    repo_bin.chmod(0o755)
    os.utime(repo_bin, (1, 1))
    monkeypatch.setattr(ocr, "__file__", str(module))
    monkeypatch.delenv("SUBZERO_VISION_OCR", raising=False)
    monkeypatch.setattr(ocr.shutil, "which", lambda cmd: "/usr/bin/swiftc" if cmd == "swiftc" else None)
    monkeypatch.setattr(ocr.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    def compile_source(cmd, **kwargs):
        Path(cmd[-1]).write_text(Path(cmd[2]).read_text())
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(ocr.subprocess, "run", compile_source)
    binary = Path(ocr.get_vision_ocr_bin())
    assert binary != repo_bin
    assert binary.read_text() == "packaged source"


def test_ocr_resolver_prefers_packaged_source_over_legacy_source(tmp_path, monkeypatch):
    module = tmp_path / "src/subzero/ocr.py"
    source = module.with_name("vision_ocr.swift")
    source.parent.mkdir(parents=True)
    source.write_text("packaged source")
    legacy = tmp_path / "tools/vision_ocr.swift"
    legacy.parent.mkdir()
    legacy.write_text("legacy source")
    monkeypatch.setattr(ocr, "__file__", str(module))
    monkeypatch.delenv("SUBZERO_VISION_OCR", raising=False)
    monkeypatch.setattr(ocr.shutil, "which", lambda cmd: "/usr/bin/swiftc" if cmd == "swiftc" else None)
    monkeypatch.setattr(ocr.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    def compile_source(cmd, **kwargs):
        Path(cmd[-1]).write_text(Path(cmd[2]).read_text())
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(ocr.subprocess, "run", compile_source)
    binary = Path(ocr.get_vision_ocr_bin())
    assert binary.read_text() == "packaged source"


def test_ocr_resolver_reports_unavailable_on_windows(tmp_path, monkeypatch):
    module = tmp_path / "subzero/ocr.py"
    source = module.with_name("vision_ocr.swift")
    source.parent.mkdir()
    source.write_text("packaged source")
    monkeypatch.setattr(ocr, "__file__", str(module))
    monkeypatch.delenv("SUBZERO_VISION_OCR", raising=False)
    monkeypatch.setattr(ocr.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(ocr.sys, "platform", "win32")
    monkeypatch.delattr(ocr.os, "uname", raising=False)

    assert ocr.get_vision_ocr_bin() is None


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


def test_distribution_credit_heading_is_not_dialogue():
    assert clean_ocr_text("GLOBAL CONTENT DISTRIBUTION\nInternational Content Distribution") == ""
    dialogue = "I work in global content distribution."
    assert clean_ocr_text(dialogue) == dialogue


def test_ocr_corrects_lowercase_l_only_in_english_contraction():
    assert clean_ocr_text("the hardest thing l've ever done.") == "the hardest thing I've ever done."
    assert clean_ocr_text("l’ve seen l'été and cl've before.") == "I’ve seen l'été and cl've before."


@pytest.mark.parametrize("flicker", [
    "I don't know it I'm doing this right.",
    "I don't know iff I'm doing this right.",
    "I don't know f I'm doing this right.",
])
def test_single_frame_character_flicker_uses_surrounding_caption(flicker):
    caption = "I don't know if I'm doing this right."
    cues = cluster_ocr_detections([
        (729.66, caption), (730.16, flicker), (730.66, caption),
    ], sample_duration=0.5, max_gap=0.75, min_duration=0.5)
    assert len(cues) == 1
    assert cues[0].text == caption
    assert (cues[0].start, cues[0].end) == ("00:12:09,660", "00:12:11,160")


@pytest.mark.parametrize("first,middle", [
    ("We should not vote for him.", "We should now vote for him."),
    ("I never said we should leave.", "I ever said we should leave."),
    ("I don't think we should go.", "I dont think we should go."),
    ("We have none of those votes.", "We have one of those votes."),
])
def test_temporal_consensus_preserves_changed_negation(first, middle):
    cues = cluster_ocr_detections([(10, first), (10.5, middle), (11, first)],
                                  sample_duration=0.5, max_gap=0.75, min_duration=0.5)
    assert [cue.text for cue in cues] == [first, middle, first]


def test_temporal_consensus_needs_two_matching_neighbors():
    first, second = "We can see if he agrees.", "We can see it he agrees."
    cues = cluster_ocr_detections([(10, first), (10.5, second), (11, second), (11.5, first)],
                                  sample_duration=0.5, max_gap=0.75, min_duration=0.5)
    assert [cue.text for cue in cues] == [first, second, first]


def test_temporal_consensus_does_not_change_short_captions():
    cues = cluster_ocr_detections([(10, "I saw it."), (10.5, "I saw if."), (11, "I saw it.")],
                                  sample_duration=0.5, max_gap=0.75, min_duration=0.5)
    assert [cue.text for cue in cues] == ["I saw it.", "I saw if.", "I saw it."]


def test_temporal_consensus_does_not_replace_different_words():
    first, second = "We should vote for Andy today.", "We should vote for Rome today."
    cues = cluster_ocr_detections([(10, first), (10.5, second), (11, first)],
                                  sample_duration=0.5, max_gap=0.75, min_duration=0.5)
    assert [cue.text for cue in cues] == [first, second, first]


def test_temporal_consensus_does_not_bridge_missing_frames():
    first, second = "We can see if he agrees.", "We can see it he agrees."
    cues = cluster_ocr_detections([(10, first), (11, second), (11.5, first)],
                                  sample_duration=0.5, max_gap=2, min_duration=0.5)
    assert [cue.text for cue in cues] == [first, second, first]


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
    assert c1.end == "00:00:13,000"

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
            res.stderr = "[showinfo] n: 0 pts: 0 pts_time:0\n[showinfo] n: 1 pts: 1 pts_time:1\n"
            return res
        if "vision_ocr" in cmd[0]:
            res.stdout = mock_ocr_output
            return res
        return res

    with patch("subzero.ocr.get_vision_ocr_bin", return_value="/usr/local/bin/vision_ocr"), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        cues = extract_and_ocr_gaps("dummy.mkv", gaps, fps=1, tmp_dir=tmp_path)
        assert len(cues) == 1
        assert cues[0].text == "Don't tell anyone."
        assert cues[0].start == "00:00:10,000"


def test_fill_subtitle_gaps_dry_run(tmp_path):
    sub_file = tmp_path / "test.srt"
    sub_file.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nPrimeira fala\n\n",
        encoding="utf-8",
    )

    with patch("subzero.ocr.find_caption_gaps", return_value=[(10.0, 14.0)]), \
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

    with patch("subzero.ocr.find_caption_gaps", return_value=[(10.0, 14.0)]), \
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


def test_partial_speech_gap_keeps_uncovered_tail():
    subtitle = "1\n00:00:10,000 --> 00:00:13,000\nFala\n\n"
    with patch("subzero.ocr.build_reference", return_value={"speech": [(10, 20)]}):
        assert find_speech_gaps("video.mkv", subtitle) == [(13.0, 20.0)]


def test_caption_gaps_find_whispers_without_vad():
    subtitle = "1\n00:00:00,000 --> 00:00:03,000\nFala\n\n2\n00:00:08,000 --> 00:00:09,000\nFim\n\n"
    with patch("subzero.ocr.build_reference", return_value={"duration": 10, "speech": []}):
        assert ocr.find_caption_gaps("video.mkv", subtitle) == [(3.0, 8.0), (9.0, 10.0)]


def test_caption_gaps_union_overlapping_subtitles():
    subtitle = "1\n00:00:00,000 --> 00:00:06,000\nFala\n\n2\n00:00:03,000 --> 00:00:04,000\nOutra\n\n"
    with patch("subzero.ocr.build_reference", return_value={"duration": 10}):
        assert ocr.find_caption_gaps("video.mkv", subtitle) == [(6.0, 10.0)]


def test_short_uppercase_dialogue_is_not_a_tribe_label():
    assert clean_ocr_text("No\nRUN NOW\nTUKU TRIBE\nDAY 1") == "No\nRUN NOW"


def test_credit_headings_are_removed_without_dropping_dialogue():
    assert clean_ocr_text("Hosted By\nExecutive Producers\nDirector\nI called the producer.") == "I called the producer."


def test_different_captions_do_not_merge_or_overlap():
    cues = cluster_ocr_detections([(10.0, "We should vote for Andy."), (10.5, "We should vote for Sam.")])
    assert [c.text for c in cues] == ["We should vote for Andy.", "We should vote for Sam."]
    assert cues[0].end == cues[1].start


@pytest.mark.parametrize("first,second", [
    ("We're all in this together.", "We're all in this, together."),
    ("You’re the one I trust.", "You're the one I trust."),
    ("Keep going. Don't stop.", "Keep going, don't stop!"),
])
def test_punctuation_changes_do_not_repeat_the_same_caption(first, second):
    cues = cluster_ocr_detections([(1243.8, first), (1244.8, second)])
    assert len(cues) == 1
    assert cues[0].start == "00:20:43,800"
    assert cues[0].end == "00:20:45,800"


@pytest.mark.parametrize("first,second", [
    ("I trust you.", "I don't trust you."),
    ("We should vote for Andy.", "We should vote for Jon."),
    ("She said yes.", "He said yes."),
])
def test_similar_captions_with_different_words_stay_separate(first, second):
    cues = cluster_ocr_detections([(10, first), (10.5, second)])
    assert [cue.text for cue in cues] == [first, second]


def test_blank_frame_ends_caption():
    cues = cluster_ocr_detections([(10.0, "Keep this secret."), (10.5, ""), (11.0, "Keep this secret.")])
    assert len(cues) == 2
    assert cues[0].end == "00:00:10,500"


def test_caption_geometry_discards_background_text():
    frame = {"subtitleText": "CBS\nIdol engraving\nDon't tell anyone.\nKeep this secret.", "items": [
        {"text": "CBS", "confidence": 0.5, "x": 0.90, "y": 0.1, "width": 0.06, "height": 0.03},
        {"text": "Idol engraving", "confidence": 1, "x": 0.01, "y": 0.15, "width": 0.20, "height": 0.06},
        {"text": "Don't tell anyone.", "confidence": 1, "x": 0.20, "y": 0.168, "width": 0.60, "height": 0.064},
        {"text": "Keep this secret.", "confidence": 1, "x": 0.25, "y": 0.1, "width": 0.50, "height": 0.064},
    ]}
    assert ocr.caption_text(frame) == "Don't tell anyone.\nKeep this secret."


def test_caption_geometry_can_allow_off_center_dialogue():
    frame = {"subtitleText": "Keep this secret.", "items": [
        {"text": "Keep this secret.", "confidence": 1, "x": 0.05, "y": 0.1, "width": 0.35, "height": 0.05},
    ]}
    assert ocr.caption_text(frame) == ""
    assert ocr.caption_text(frame, center_tolerance=None) == "Keep this secret."


def test_caption_geometry_excludes_job_titles_credits_and_card_print():
    frame = {"subtitleText": "MARKETING MANAGER\nKAHATA PEARSON\nI'm gay.", "items": [
        {"text": "MARKETING MANAGER", "confidence": 1, "x": 0.346, "y": 0.176, "width": 0.331, "height": 0.036},
        {"text": "KAHATA PEARSON", "confidence": 1, "x": 0.256, "y": 0.328, "width": 0.487, "height": 0.093},
        {"text": "KEEP YOUR THREE TRIBAL", "confidence": 1, "x": 0.25, "y": 0.20, "width": 0.50, "height": 0.03},
        {"text": "I'm gay.", "confidence": 1, "x": 0.436, "y": 0.090, "width": 0.131, "height": 0.085},
    ]}
    assert ocr.caption_text(frame) == "I'm gay."


def test_large_centered_title_card_is_not_dialogue():
    frame = {"subtitleText": "SHOW TITLE\nTAGLINE", "items": [
        {"text": "SHOW TITLE", "confidence": 1, "x": 0.20, "y": 0.37, "width": 0.60, "height": 0.17},
        {"text": "TAGLINE", "confidence": 1, "x": 0.38, "y": 0.06, "width": 0.23, "height": 0.085},
    ]}
    assert ocr.caption_text(frame) == ""


def test_corner_logo_does_not_suppress_real_captions():
    frame = {"subtitleText": "LOGO\nKeep this secret.", "items": [
        {"text": "LOGO", "confidence": 1, "x": 0.90, "y": 0.07, "width": 0.07, "height": 0.05},
        {"text": "Keep this secret.", "confidence": 1, "x": 0.25, "y": 0.10, "width": 0.50, "height": 0.064},
    ]}
    assert ocr.caption_text(frame) == "Keep this secret."


@pytest.mark.parametrize("failure", ["ffmpeg", "vision", "json", "missing_frame"])
def test_ocr_failures_are_not_reported_as_no_captions(tmp_path, failure):
    def run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            if failure == "ffmpeg":
                return subprocess.CompletedProcess(cmd, 1, "", "cannot decode video")
            folder = Path(cmd[-1]).parent
            (folder / "f_001.jpg").touch()
            return subprocess.CompletedProcess(cmd, 0, "", "[showinfo] n: 0 pts: 0 pts_time:0")
        rows = [{"file": "f_001.jpg", "subtitleText": "Secret whisper."}]
        return subprocess.CompletedProcess(cmd, int(failure == "vision"), "bad JSON" if failure == "json" else json.dumps([] if failure == "missing_frame" else rows), "Vision failed")

    with patch("subprocess.run", side_effect=run), pytest.raises(RuntimeError):
        extract_and_ocr_gaps("video.mkv", [(1, 3)], ocr_bin="vision_ocr", tmp_dir=tmp_path)


def test_frame_file_identity_and_pts_control_caption_timing(tmp_path):
    def run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            folder = Path(cmd[-1]).parent
            (folder / "f_001.jpg").touch()
            (folder / "f_002.jpg").touch()
            return subprocess.CompletedProcess(cmd, 0, "", "[showinfo] n: 0 pts: 0 pts_time:0\n[showinfo] n: 1 pts: 1 pts_time:0.5")
        rows = [{"file": "f_002.jpg", "subtitleText": "Tell nobody."}, {"file": "f_001.jpg", "subtitleText": ""}]
        return subprocess.CompletedProcess(cmd, 0, json.dumps(rows), "")

    with patch("subprocess.run", side_effect=run):
        cues = extract_and_ocr_gaps("video.mkv", [(10, 10.8)], ocr_bin="vision_ocr", fps=2, tmp_dir=tmp_path)
    assert len(cues) == 1
    assert cues[0].start == "00:00:10,500"
    assert cues[0].end == "00:00:10,800"


def test_long_gap_frame_numbers_keep_their_actual_timestamps(tmp_path):
    def run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            folder = Path(cmd[-1]).parent
            for frame in range(1, 1001):
                (folder / f"f_{frame:03d}.jpg").touch()
            pts = "\n".join(f"pts_time:{frame / 2}" for frame in range(1000))
            return subprocess.CompletedProcess(cmd, 0, "", pts)
        rows = [{"file": filename, "subtitleText": "Keep this secret." if Path(filename).name == "f_1000.jpg" else ""}
                for filename in cmd[2:]]
        return subprocess.CompletedProcess(cmd, 0, json.dumps(rows), "")

    with patch("subprocess.run", side_effect=run):
        cues = extract_and_ocr_gaps("video.mkv", [(0, 500)], ocr_bin="vision_ocr", tmp_dir=tmp_path)
    assert [(cue.start, cue.end) for cue in cues] == [("00:08:19,500", "00:08:20,000")]


@pytest.mark.parametrize("mark", ["♪", "♫"])
def test_caption_gaps_respect_existing_musical_cues(mark):
    subtitle = (
        "1\n00:00:02,000 --> 00:00:04,000\nA normal cue.\n\n"
        f"2\n00:00:05,000 --> 00:00:08,000\n{mark} I am singing {mark}\n"
    )
    with patch("subzero.ocr.build_reference", return_value={"duration": 10}):
        gaps = ocr.find_caption_gaps("video.mkv", subtitle)
    assert gaps == [(0, 2), (4, 5), (8, 10)]


@pytest.fixture
def recovery(tmp_path):
    source = tmp_path / "episode.srt"
    original = "1\n00:00:01,000 --> 00:00:04,000\n<i>Primeira fala</i>\n\n"
    source.write_text(original)
    recovered = [Cue("00:00:03,500", "00:00:05,000", "Don't tell anyone.")]
    with patch("subzero.ocr.find_caption_gaps", return_value=[(4, 6)]), patch("subzero.ocr.extract_and_ocr_gaps", return_value=recovered):
        yield source, original, recovered


def test_fill_preserves_existing_cues_and_clips_new_overlaps(recovery):
    source, original, _ = recovery
    with patch("subzero.ocr.translate_cues", return_value=[Cue("00:00:03,500", "00:00:05,000", "Não conte a ninguém.")]):
        fill_subtitle_gaps("video.mkv", source, target_lang="pt-BR")
    cues = parse_srt(source.read_text())
    assert cues[0] == parse_srt(original)[0]
    assert cues[1].start == "00:00:04,000"
    assert source.with_suffix(".srt.bak").read_text() == original


def test_fill_wraps_recovered_text_without_changing_existing_cues(recovery):
    source, original, _ = recovery
    translated = "Não conte a ninguém o que conversamos porque precisamos manter esse segredo."
    with patch("subzero.ocr.translate_cues", return_value=[Cue("00:00:03,500", "00:00:05,000", translated)]):
        fill_subtitle_gaps("video.mkv", source, target_lang="pt-BR")
    cues = parse_srt(source.read_text())
    assert cues[0] == parse_srt(original)[0]
    assert all(len(line) <= 42 for line in cues[1].text.splitlines())
    assert " ".join(cues[1].text.split()) == translated


def test_fill_allows_valid_unchanged_spanish_word(recovery):
    source, _, recovered = recovery
    recovered[0] = Cue("00:00:03,500", "00:00:05,000", "No")
    with patch("subzero.ocr.translate_cues", return_value=recovered):
        report = fill_subtitle_gaps("video.mkv", source, target_lang="es")
    assert report.cues_recovered == 1


def test_fill_rejects_untranslated_english_echo(recovery):
    source, original, recovered = recovery
    with patch("subzero.ocr.translate_cues", return_value=recovered), pytest.raises(RuntimeError, match="untranslated"):
        fill_subtitle_gaps("video.mkv", source, target_lang="pt-BR")
    assert source.read_text() == original


def test_fill_refuses_source_changed_during_ocr(recovery):
    source, _, _ = recovery
    def translated(*args, **kwargs):
        source.write_text("changed by another worker")
        return [Cue("00:00:03,500", "00:00:05,000", "Não conte a ninguém.")]
    with patch("subzero.ocr.translate_cues", side_effect=translated), pytest.raises(RuntimeError, match="changed"):
        fill_subtitle_gaps("video.mkv", source, target_lang="pt-BR")
    assert source.read_text() == "changed by another worker"
    assert not list(source.parent.glob(".subtitle-*.tmp"))


def test_fill_atomic_replace_failure_keeps_original(recovery):
    source, original, _ = recovery
    with patch("subzero.ocr.translate_cues", return_value=[Cue("00:00:03,500", "00:00:05,000", "Não conte a ninguém.")]), patch("subzero.ocr.os.replace", side_effect=OSError("disk unavailable")), pytest.raises(OSError, match="disk unavailable"):
        fill_subtitle_gaps("video.mkv", source, target_lang="pt-BR")
    assert source.read_text() == original
    assert not list(source.parent.glob(".subtitle-*.tmp"))
