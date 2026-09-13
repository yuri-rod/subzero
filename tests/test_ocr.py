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


def test_ocr_resolver_ignores_legacy_repo_binary_even_with_newer_timestamp(tmp_path, monkeypatch):
    module = tmp_path / "src/subzero/ocr.py"
    source = module.with_name("vision_ocr.swift")
    source.parent.mkdir(parents=True)
    source.write_text("packaged source")
    repo_bin = tmp_path / "tools/vision_ocr"
    repo_bin.parent.mkdir()
    repo_bin.write_text("stale binary")
    repo_bin.chmod(0o755)
    newer = source.stat().st_mtime + 3600
    os.utime(repo_bin, (newer, newer))
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
    assert clean_ocr_text("GLOBAL CONTENT\nDISTRIBUTION") == ""
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


def caption_frame(text, *, height=0.065, y=0.1, candidates=()):
    return {"subtitleText": text, "items": [
        {"text": text, "confidence": 1, "x": 0.25, "y": y, "width": 0.5, "height": height,
         "candidates": [{"text": candidate, "confidence": 1} for candidate in candidates]},
    ]}


def test_neighbor_confirms_single_line_with_oversized_box():
    text = "I also really like Kishan."
    frame = caption_frame(text, height=0.115, y=0.08)
    neighbor = caption_frame(text)
    assert ocr.caption_text(frame) == ""
    assert ocr.caption_text(frame, previous=neighbor) == text
    assert ocr.caption_text(frame, following=neighbor) == text


def test_neighbor_preserves_both_lines_when_one_box_grows():
    first, second = "That gives us four in six.", "Are you down with that?"
    frame = caption_frame(first, y=0.168)
    frame["items"].extend(caption_frame(second, height=0.115, y=0.08)["items"])
    neighbor = caption_frame(first, y=0.168)
    neighbor["items"].extend(caption_frame(second)["items"])
    assert ocr.caption_text(frame, following=neighbor) == first + "\n" + second


def test_geometry_support_requires_matching_text_and_position():
    frame = caption_frame("MARKETING MANAGER", height=0.115, y=0.08)
    assert ocr.caption_text(frame, following=caption_frame("Are you down with that?")) == ""
    assert ocr.caption_text(frame, following=caption_frame("MARKETING MANAGER", y=0.23)) == ""


def test_native_alternative_uses_agreeing_neighbors():
    expected = "I really want to work with you."
    frame = caption_frame("Treally want to work with you.", candidates=[expected])
    neighbor = caption_frame(expected)
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == expected
    assert ocr.caption_text(frame, previous=neighbor) == frame["subtitleText"]


def test_native_alternative_restores_missing_first_person_contraction():
    expected = "I don't know if I'm doing this right."
    frame = caption_frame("I don't know if 'm doing this right.", candidates=[expected])
    neighbor = caption_frame(expected)
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == expected


def test_native_alternative_does_not_turn_an_opening_quote_into_a_letter():
    frame = caption_frame('"an Idol good for three', candidates=["wan Idol good for three"])
    neighbor = caption_frame("wan Idol good for three")
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == frame["subtitleText"]


def test_single_frame_missing_pronoun_and_spacing_use_neighboring_caption():
    first = "I really want to work with you."
    for flicker in ("really want to work with you.", "Treally want to work with you."):
        cues = cluster_ocr_detections([(10, first), (10.5, flicker), (11, first)],
                                      sample_duration=0.5, max_gap=0.75, min_duration=0.5)
        assert [cue.text for cue in cues] == [first]


def test_slanted_prop_text_is_not_a_caption():
    frame = caption_frame("Y IT SAFE,")
    frame["items"][0]["angle"] = 42
    assert ocr.caption_text(frame) == ""
    frame = caption_frame("Rome's gone again.")
    frame["items"][0]["angle"] = -5.47
    assert ocr.caption_text(frame) == "Rome's gone again."


def test_neighbor_consensus_does_not_invent_an_unlisted_reading():
    frame = caption_frame("Treally want to work with you.")
    neighbor = caption_frame("I really want to work with you.")
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == frame["subtitleText"]


def test_native_alternative_preserves_negation_changes():
    frame = caption_frame("We should now vote for him.", candidates=["We should not vote for him."])
    neighbor = caption_frame("We should not vote for him.")
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == frame["subtitleText"]


@pytest.mark.parametrize("first,second", [
    ("I think we should vote for Kyle.", "I think we should vote for Kyla."),
    ("I think we should vote for Sue.", "I think we should vote for Sam."),
    ("Kyle should be joining us soon.", "Kyla should be joining us soon."),
])
def test_native_alternative_preserves_name_changes(first, second):
    frame = caption_frame(second, candidates=[first])
    neighbor = caption_frame(first)
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == second
    cues = cluster_ocr_detections([(10, first), (10.5, second), (11, first)],
                                  sample_duration=0.5, max_gap=0.75, min_duration=0.5)
    assert [cue.text for cue in cues] == [first, second, first]


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


def test_large_upper_logo_suppresses_lower_branding_without_banning_television_dialogue():
    frame = caption_frame("TELEVISION")
    frame["items"].append({"text": "STUDIO BRAND", "confidence": 1, "x": 0.32,
                           "y": 0.75, "width": 0.36, "height": 0.16})
    assert ocr.caption_text(frame) == ""
    assert ocr.caption_text(caption_frame("TELEVISION")) == "TELEVISION"


def test_corner_logo_does_not_suppress_real_captions():
    frame = {"subtitleText": "LOGO\nKeep this secret.", "items": [
        {"text": "LOGO", "confidence": 1, "x": 0.90, "y": 0.07, "width": 0.07, "height": 0.05},
        {"text": "Keep this secret.", "confidence": 1, "x": 0.25, "y": 0.10, "width": 0.50, "height": 0.064},
    ]}
    assert ocr.caption_text(frame) == "Keep this secret."


def test_region_retry_restores_missing_second_line_without_changing_timing():
    first = "I'd just like to say my piece first"
    second = "before we, like, talk it through."
    complete = caption_frame(first, y=0.168)
    complete["items"].extend(caption_frame(second)["items"])
    partial = caption_frame(first, y=0.168)
    partial["items"].extend(caption_frame("before we, like, falk it through.", height=0.11, y=0.075)["items"])
    frames = [complete, partial, complete]
    stamps = [4211.3, 4211.8, 4212.3]
    assert ocr.caption_retry_indices(frames, stamps) == [0, 1, 2]
    readings = ocr.recover_caption_runs(frames, {1: complete}, stamps)
    cues = cluster_ocr_detections(list(zip(stamps, readings)), min_duration=0.5,
                                  max_gap=0.75, sample_duration=0.5)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        ("01:10:11,300", "01:10:12,800", first + "\n" + second)]


def test_region_retry_coalesces_native_observed_card_variants():
    caption = "It's only day one and already your first setback."
    variants = ["Pit's only day one and already your first setback.", caption,
                "iIt's only day one and already your first setback.", caption]
    frames = [caption_frame(text) for text in variants]
    retried = {index: caption_frame(caption) for index in range(4)}
    stamps = [1506.5 + index / 2 for index in range(4)]
    assert ocr.recover_caption_runs(frames, retried, stamps) == [caption] * 4


def test_region_retry_does_not_duplicate_a_complete_line_as_two_cropped_fragments():
    caption = "Keep this secret."
    complete = caption_frame(caption, height=0.070, y=0.093)
    complete["items"][0].update(x=0.339, width=0.320)
    split = {"items": [
        {"text": "Keep this", "confidence": 1, "x": 0.321, "y": 0.113, "width": 0.173, "height": 0.049},
        {"text": "secret.", "confidence": 1, "x": 0.518, "y": 0.091, "width": 0.143, "height": 0.087},
    ]}
    assert ocr.recover_caption_runs([complete] * 3, {index: split for index in range(3)},
                                    [0, 0.1, 0.2]) == [caption] * 3


def test_region_retry_keeps_repeated_words_on_a_separate_physical_line():
    first, second = "Keep this secret.", "This secret."
    complete = caption_frame(first, y=0.168)
    complete["items"].extend(caption_frame(second, y=0.1)["items"])
    partial = caption_frame(first, y=0.168)
    frames = [complete, partial, complete]
    readings = ocr.recover_caption_runs(frames, {1: complete}, [0, 0.1, 0.2])
    assert readings == [first + "\n" + second] * 3


@pytest.mark.parametrize("first,second", [
    ("I think we should vote for Kyle.", "I think we should vote for Kyla."),
    ("Kyle should be joining us soon.", "Kyla should be joining us soon."),
    ("I think we should vote for Sue.", "I think we should vote for Sam."),
    ("We should not vote for him.", "We should now vote for him."),
    ("We can get two extra votes.", "We can get ten extra votes."),
])
def test_region_retry_preserves_names_negation_and_number_changes(first, second):
    frames = [caption_frame(first), caption_frame(second), caption_frame(first)]
    retried = {index: caption_frame(first) for index in range(3)}
    assert ocr.recover_caption_runs(frames, retried, [0, 0.5, 1]) == [first, second, first]


@pytest.mark.parametrize("original, alternative", [
    ("Please _ keep this secret.", "Please keep this secret."),
    ("Please keep this secret.", "Please _ keep this secret."),
    ("Please _ keep this secret.", "Please keep _ this secret."),
])
def test_region_retry_preserves_censorship_marker_position(original, alternative):
    frames = [caption_frame(alternative), caption_frame(original), caption_frame(alternative)]
    retried = {index: caption_frame(alternative) for index in range(3)}
    assert ocr.recover_caption_runs(frames, retried, [0, 0.1, 0.2]) == [
        alternative, original, alternative]


@pytest.mark.parametrize("original, alternative", [
    ("Please _ keep this secret today.", "Please keep this secret today."),
    ("Please keep this secret today.", "Please _ keep this secret today."),
    ("Please _ keep this secret today.", "Please keep _ this secret today."),
])
def test_native_candidate_preserves_censorship_marker_position(original, alternative):
    frame = caption_frame(original, candidates=[alternative])
    neighbor = caption_frame(alternative)
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == original


def test_native_candidate_censorship_bar_does_not_count_as_a_spoken_word():
    original, alternative = "We _ must have him.", "We _ must save him."
    frame = caption_frame(original, candidates=[alternative])
    neighbor = caption_frame(alternative)
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == original


def test_cluster_censorship_bar_does_not_count_as_a_spoken_word():
    first, changed = "We _ must save him.", "We _ must have him."
    cues = cluster_ocr_detections([(0, first), (0.1, changed), (0.2, first)],
                                  min_duration=0, sample_duration=0.1)
    assert [cue.text for cue in cues] == [first, changed, first]


def test_region_retry_censorship_bar_does_not_reach_three_spoken_words():
    first, changed = "We _ win.", "We _ win.r"
    frames = [caption_frame(first), caption_frame(changed), caption_frame(first)]
    retried = {index: caption_frame(first) for index in range(3)}
    assert ocr.recover_caption_runs(frames, retried, [0, 0.1, 0.2]) == [first, changed, first]


def test_censorship_marker_width_and_surrounding_spacing_are_equivalent():
    readings = [(0, "Please _ keep this secret."), (0.1, "Please   ____  keep this secret.")]
    cues = cluster_ocr_detections(readings, min_duration=0, sample_duration=0.1)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        ("00:00:00,000", "00:00:00,200", readings[1][1])]


@pytest.mark.parametrize("marked", ["_ Please keep this secret", "Please keep this secret _"])
def test_caption_fragment_merge_preserves_censorship_marker_changes(marked):
    plain = "Please keep this secret"
    cues = cluster_ocr_detections([(0, plain), (0.1, marked), (0.2, plain)],
                                  min_duration=0, sample_duration=0.1)
    assert [cue.text for cue in cues] == [plain, marked, plain]


def test_region_retry_keeps_full_frame_title_card_veto():
    credit = caption_frame("TELEVISION")
    credit["items"].append({"text": "STUDIO BRAND", "confidence": 1, "x": 0.32,
                           "y": 0.75, "width": 0.36, "height": 0.16})
    frames = [credit, credit, credit]
    retried = {index: caption_frame("TELEVISION") for index in range(3)}
    assert ocr.caption_retry_indices(frames, [0, 0.5, 1]) == []
    assert ocr.recover_caption_runs(frames, retried, [0, 0.5, 1]) == ["", "", ""]


def test_region_retry_does_not_add_an_uncorroborated_prop_line():
    original = caption_frame("Keep this secret.")
    prop = caption_frame("KEEP YOUR VOTE SAFE", y=0.168)
    prop["items"].extend(original["items"])
    assert ocr.recover_caption_runs([original] * 3, {1: prop}, [0, 0.5, 1]) == ["Keep this secret."] * 3


def test_region_retry_needs_same_frame_evidence_before_merging_real_word_changes():
    first, second = "I could turn here right now.", "I could return here right now."
    frames = [caption_frame(first), caption_frame(second), caption_frame(first)]
    assert ocr.recover_caption_runs(frames, dict(enumerate(frames)), [0, 0.5, 1]) == [first, second, first]


def test_gap_extraction_runs_native_region_retry_for_partial_captions(tmp_path):
    first, second = "I'd just like to say my piece first", "before we, like, talk it through."
    complete = caption_frame(first, y=0.168)
    complete["items"].extend(caption_frame(second)["items"])
    partial = caption_frame(first, y=0.168)
    called = []

    def run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            for index in range(3):
                (Path(cmd[-1]).parent / f"f_{index + 1:03d}.jpg").touch()
            return subprocess.CompletedProcess(cmd, 0, "", "pts_time:0\npts_time:0.5\npts_time:1")
        region = "--caption-region" in cmd
        called.append(region)
        filenames = cmd[3:] if region else cmd[2:]
        rows = [{**(partial if not region and Path(filename).name == "f_002.jpg" else complete),
                 "file": filename} for filename in filenames]
        return subprocess.CompletedProcess(cmd, 0, json.dumps(rows), "")

    with patch("subprocess.run", side_effect=run):
        cues = extract_and_ocr_gaps("video.mkv", [(4211.3, 4212.8)], ocr_bin="vision_ocr", tmp_dir=tmp_path)
    assert called == [False, True]
    assert [cue.text for cue in cues] == [first + "\n" + second]
    assert (cues[0].start, cues[0].end) == ("01:10:11,300", "01:10:12,800")


def test_full_caption_scan_uses_bounded_overlapping_windows_and_preserves_cue_edges():
    scanned = []

    def scan(video, intervals, **kwargs):
        if kwargs.get("fps", 2) == 2:
            scanned.extend(intervals)
        return [(119.55, "Keep this secret."), (120.05, "Keep this secret."), (120.55, "")]

    with patch("subzero.ocr._scan_caption_frames", side_effect=scan):
        cues = ocr.extract_all_captions("video.mkv", 241)
    assert scanned == [(0, 121), (119, 241), (239, 241)]
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        ("00:01:59,550", "00:02:00,550", "Keep this secret.")]


def test_full_caption_scan_does_not_extend_a_short_observed_caption():
    with patch("subzero.ocr._scan_caption_frames", return_value=[(10.2, "Go!"), (10.35, "")]):
        cues = ocr.extract_all_captions("video.mkv", 30)
    assert [(cue.start, cue.end) for cue in cues] == [("00:00:10,200", "00:00:10,275")]


def test_caption_transition_windows_cover_sampling_brackets_and_merge_neighbors():
    frames = [(9.5, ""), (10.001, "Keep this secret."), (10.502, "Keep this secret!"),
              (11.003, "Tell nobody."), (11.504, ""), (12.005, "")]
    assert ocr.caption_transition_windows(frames, 30) == [(9.5, 12.004)]


def test_dense_timing_replaces_coarse_observations_and_stops_at_actual_blank():
    coarse = [(9.5, ""), (10.01, "Keep this secret."), (10.51, "Keep this secret."), (11.01, "")]
    dense = [(9.51, ""), (9.61, ""), (9.71, "Keep this secret."), (9.81, "Keep this secret."),
             (10.11, "Keep this secret."), (10.41, "Keep this secret."),
             (10.81, "Keep this secret."), (10.91, ""), (11.01, ""), (11.11, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense) as scan:
        cues = ocr.refine_caption_timing("video.mkv", coarse, 30)
    assert scan.call_args.kwargs["fps"] == 10
    assert scan.call_args.kwargs["retry_all"] is True
    assert [(cue.start, cue.end) for cue in cues] == [("00:00:09,660", "00:00:10,860")]


def test_dense_timing_preserves_the_coarse_middle_of_a_long_caption():
    coarse = [(0, ""), (0.5, "Keep this secret."), (1, "Keep this secret."),
              (1.5, "Keep this secret."), (2, "Keep this secret."), (2.5, "Keep this secret."),
              (3, "Keep this secret."), (3.5, "")]
    dense = [(0, ""), (0.1, ""), (0.2, "Keep this secret."), (0.8, "Keep this secret."),
             (3, "Keep this secret."), (3.1, "Keep this secret."), (3.2, ""), (3.9, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 5)
    assert [(cue.start, cue.end) for cue in cues] == [("00:00:00,150", "00:00:03,150")]


def test_dense_timing_clamps_transition_windows_and_last_caption_to_eof():
    coarse = [(0.025, "Keep this secret."), (0.526, "Keep this secret.")]
    assert ocr.caption_transition_windows(coarse, 0.8) == [(0, 0.8)]
    with patch("subzero.ocr._scan_caption_frames", return_value=[
            (0.025, "Keep this secret."), (0.725, "Keep this secret.")]):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 0.8)
    assert [(cue.start, cue.end) for cue in cues] == [("00:00:00,025", "00:00:00,800")]


def test_dense_timing_skips_video_with_no_recognized_captions():
    with patch("subzero.ocr._scan_caption_frames") as scan:
        assert ocr.refine_caption_timing("video.mkv", [(0, ""), (0.5, "")], 1) == []
    scan.assert_not_called()


def test_dense_timing_keeps_supported_caption_text_when_a_dense_frame_loses_a_line():
    complete = "I feel like we can work together\nif we keep this secret."
    coarse = [(0, ""), (0.5, complete), (1, complete), (1.5, complete), (2, "")]
    dense = [(0.1, ""), (0.2, complete), (0.3, complete), (0.7, "if we keep this secret."),
             (0.8, "if we keep this secret."), (0.9, complete), (1.1, complete),
             (1.6, complete), (1.7, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 3)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [("00:00:00,150", "00:00:01,650", complete)]


def test_dense_timing_does_not_add_a_second_line_before_any_full_caption_was_observed():
    first, second = "I feel like we can work together", "if we keep this secret."
    complete = first + "\n" + second
    coarse = [(0, ""), (0.5, complete), (1, complete), (1.5, "")]
    dense = [(0.1, ""), (0.2, second), (0.3, complete), (0.8, complete), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [cue.text for cue in cues] == [second, complete]


def test_dense_timing_confirmation_is_local_to_each_repeated_caption_run():
    first, second = "Keep this secret between us.", "Nobody else should know."
    complete = first + "\n" + second
    coarse = [(0, ""), (0.5, complete), (1, complete), (1.5, ""),
              (9.5, ""), (10, complete), (10.5, complete), (11, "")]
    dense = [(0.4, complete), (1.1, complete), (1.2, ""), (9.5, ""),
             (9.6, second), (9.7, second), (9.8, complete), (10.2, complete), (10.6, complete), (10.7, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 12)
    assert [cue.text for cue in cues] == [complete, second, complete]


@pytest.mark.parametrize("dense", [[], [(0.4, ""), (0.5, ""), (0.6, ""), (1, ""), (1.1, "")]])
def test_dense_timing_reports_a_stable_caption_lost_by_the_dense_scan(dense):
    coarse = [(0, ""), (0.5, "Keep this secret."), (1, "Keep this secret."), (1.5, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense), \
         pytest.raises(RuntimeError, match="Dense caption verification lost a confirmed caption"):
        ocr.refine_caption_timing("video.mkv", coarse, 2)


@pytest.mark.parametrize("changed", ["I really want to work with Kyla.", "I really don't want to work with Kyle."])
def test_dense_timing_retains_explicit_new_names_and_negation(changed):
    first = "I really want to work with Kyle."
    coarse = [(0, first), (0.5, first), (1, first), (1.5, "")]
    dense = [(0, first), (0.1, first), (0.2, changed), (0.3, first), (0.8, first), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [cue.text for cue in cues] == [first, changed, first]


@pytest.mark.parametrize("first, changed", [
    ("Please _ keep this secret today.", "Please keep this secret today."),
    ("Please keep this secret today.", "Please _ keep this secret today."),
    ("Please _ keep this secret today.", "Please keep _ this secret today."),
])
def test_dense_timing_retains_censorship_marker_changes(first, changed):
    coarse = [(0, first), (0.5, first), (1, first), (1.5, "")]
    dense = [(0, first), (0.1, first), (0.2, changed), (0.3, first), (0.8, first), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [cue.text for cue in cues] == [first, changed, first]
    assert cues[1].start == "00:00:00,150"
    assert cues[1].end == "00:00:00,250"


def test_dense_timing_rejects_loss_of_confirmed_censorship_marker():
    first, lost = "Please _ keep this secret.", "Please keep this secret."
    coarse = [(0, ""), (0.5, first), (1, first), (1.5, "")]
    dense = [(0.4, lost), (0.5, lost), (0.6, lost), (1, lost), (1.1, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense), \
         pytest.raises(RuntimeError, match="Dense caption verification lost a confirmed caption"):
        ocr.refine_caption_timing("video.mkv", coarse, 2)


def test_dense_timing_censorship_bar_does_not_count_as_a_spoken_word():
    first, changed = "We _ must save him.", "We _ must have him."
    coarse = [(0, first), (0.5, first), (1, first), (1.5, "")]
    dense = [(0, first), (0.1, first), (0.2, changed), (0.3, first), (0.8, first), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [cue.text for cue in cues] == [first, changed, first]


def test_dense_partial_line_censorship_bar_does_not_reach_three_spoken_words():
    partial = "We _ win."
    complete = partial + "\nKeep this between us."
    coarse = [(0, complete), (0.5, complete), (1, complete), (1.5, "")]
    dense = [(0, complete), (0.1, complete), (0.2, partial), (0.3, complete),
             (0.8, complete), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [cue.text for cue in cues] == [complete, partial, complete]


def test_frame_sampling_keeps_input_pts_instead_of_rewriting_a_fixed_fps_grid(tmp_path):
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[0] == "ffmpeg":
            (Path(cmd[-1]).parent / "f_001.jpg").touch()
            return subprocess.CompletedProcess(cmd, 0, "", "pts_time:0.033367")
        return subprocess.CompletedProcess(cmd, 0, json.dumps([
            {"file": cmd[-1], "subtitleText": "Keep this secret."}]), "")

    with patch("subprocess.run", side_effect=run):
        cues = extract_and_ocr_gaps("video.mkv", [(100, 101)], ocr_bin="vision_ocr", tmp_dir=tmp_path)
    filter_graph = commands[0][commands[0].index("-vf") + 1]
    assert "select=" in filter_graph and "showinfo" in filter_graph and "fps=" not in filter_graph
    assert commands[0][commands[0].index("-fps_mode") + 1] == "vfr"
    assert cues[0].start == "00:01:40,033"


def test_full_caption_recovery_preserves_distinct_overlapping_atomic_cues(tmp_path):
    source = tmp_path / "source.srt"
    original = "1\n00:24:28,133 --> 00:24:30,168\nMe too. Me too.\n\n"
    source.write_text(original)
    output = tmp_path / "complete.srt"
    recovered = Cue("00:24:28,534", "00:24:30,035", "I also like Rome.")
    with patch("subzero.ocr.build_reference", return_value={"duration": 5150.976}), \
         patch("subzero.ocr.extract_all_captions", return_value=[recovered]) as scan, \
         patch("subzero.ocr.fix_text", side_effect=AssertionError("Atomic cues must not be overlap-clipped")):
        report = fill_subtitle_gaps("video.mkv", source, output=output, all_captions=True,
                                   target_lang="en", backup=False)
    scan.assert_called_once_with("video.mkv", 5150.976, progress=None)
    assert parse_srt(output.read_text()) == parse_srt(original) + [recovered]
    assert report.cues == [recovered] and report.cues_recovered == 1
    assert source.read_text() == original


def test_full_caption_recovery_reports_only_additions_after_source_deduplication(tmp_path):
    source = tmp_path / "source.srt"
    original = "1\n00:00:01,000 --> 00:00:04,000\nKeep this secret.\n\n"
    source.write_text(original)
    output = tmp_path / "complete.srt"
    duplicate = Cue("00:00:01,050", "00:00:04,050", "Keep this secret.")
    with patch("subzero.ocr.build_reference", return_value={"duration": 120}), \
         patch("subzero.ocr.extract_all_captions", return_value=[duplicate]):
        report = fill_subtitle_gaps("video.mkv", source, output=output, all_captions=True,
                                   target_lang="en", backup=False)
    assert report.cues_recovered == 0 and report.cues == []
    assert not output.exists()


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
