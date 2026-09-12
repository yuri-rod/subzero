from unittest.mock import patch

import pytest

from subzero import ocr


def test_dense_character_dropout_does_not_follow_a_weak_coarse_fragment():
    full = "I also really like Kishan."
    partial = "also really like Kishan."
    coarse = [(0, ""), (0.5, full), (1, full), (1.5, full), (2, partial), (2.5, "")]
    dense = [(0.3, ""), (0.4, full), (0.5, full), (1.6, full), (1.7, full),
             (1.8, partial), (1.9, full), (2, full), (2.1, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 3)
    assert [cue.text for cue in cues] == [full]
    assert (cues[0].start, cues[0].end) == ("00:00:00,350", "00:00:02,050")


def test_dense_missing_line_uses_full_observations_before_coarse_support():
    full = "Keep this secret between us.\nNobody else should know."
    partial = full.splitlines()[1]
    coarse = [(0, ""), (1, full), (1.5, full), (2, partial), (2.5, full), (3, "")]
    dense = [(0.2, ""), (0.3, full), (0.4, partial), (0.5, partial), (0.6, full), (1, full),
             (1.5, full), (1.9, full), (2, partial), (2.1, full), (2.7, full), (2.8, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 4)
    assert [cue.text for cue in cues] == [full]


@pytest.mark.parametrize("middle", [
    "We should not vote for Sam.",
    "We should vote for Pam.",
    "We should vote for Sam twice.",
    "",
])
def test_dense_local_consensus_preserves_content_changes_and_blanks(middle):
    full = "We should vote for Sam."
    readings = [(0, full), (0.1, middle), (0.2, middle), (0.3, full)]
    assert ocr._stabilize_dense_readings(readings) == readings


def test_dense_local_consensus_preserves_one_line_before_and_after_full_caption():
    partial = "Keep this secret between us."
    full = partial + "\nNobody else should know."
    readings = [(0, partial), (0.1, full), (0.2, full), (0.3, partial)]
    assert ocr._stabilize_dense_readings(readings) == readings


def test_dense_local_consensus_does_not_bridge_missing_frames():
    full = "Keep this secret between us.\nNobody else should know."
    readings = [(0, full), (0.4, full.splitlines()[1]), (0.5, full)]
    assert ocr._stabilize_dense_readings(readings) == readings


def test_dense_local_consensus_does_not_replace_a_sustained_reading():
    full = "Keep this secret between us.\nNobody else should know."
    readings = [(0, full)] + [(n / 10, full.splitlines()[1]) for n in range(1, 8)] + [(0.8, full)]
    assert ocr._stabilize_dense_readings(readings) == readings


def test_dense_timing_does_not_extend_an_unfinished_line_before_its_native_observation():
    partial = "Keep this secret between us"
    full = partial + "\nuntil we reach camp."
    coarse = [(0, ""), (0.5, full), (1, full), (1.5, "")]
    dense = [(0.1, ""), (0.2, partial), (0.3, full), (0.8, full), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        ("00:00:00,150", "00:00:00,250", partial),
        ("00:00:00,250", "00:00:01,300", full),
    ]


def test_dense_consensus_does_not_restore_an_extra_letter_from_noisy_neighbors():
    correct = "A single fourth key will decide\nwho earns the supplies"
    noisy = correct + "i"
    readings = [(0, noisy), (0.1, correct), (0.2, noisy)]
    assert ocr._stabilize_dense_readings(readings) == readings


def test_dense_timeline_does_not_reintroduce_a_rejected_extra_letter():
    correct = "A single fourth key will decide\nwho earns the supplies"
    noisy = correct + "i"
    coarse = [(0, ""), (0.5, noisy), (1, "")]
    dense = [(0.1, noisy), (0.2, correct), (0.3, noisy), (0.4, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [cue.text for cue in cues] == [noisy, correct, noisy]


def test_caption_consensus_preserves_a_changed_number():
    first, changed = "We need 7 votes to win.", "We need 8 votes to win."
    cues = ocr.cluster_ocr_detections([(0, first), (0.1, changed), (0.2, first)],
                                     min_duration=0, max_gap=0.15, sample_duration=0.1)
    assert [cue.text for cue in cues] == [first, changed, first]


def test_dense_consensus_corrects_a_coarse_character_error_from_repeated_native_readings():
    correct = "A single fourth key will decide\nwho earns the supplies"
    noisy = correct + "i"
    coarse = [(0, ""), (0.5, noisy), (1, noisy), (1.5, "")]
    dense = [(0.2, ""), (0.3, correct), (0.4, correct), (0.5, correct),
             (1, correct), (1.2, correct), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        ("00:00:00,250", "00:00:01,250", correct)]


@pytest.mark.parametrize("changed", ["We need 8 votes to win.", "We need no votes to win."])
def test_dense_anchor_refinement_does_not_hide_a_missing_confirmed_number(changed):
    first = "We need 7 votes to win."
    coarse = [(0, ""), (0.5, first), (1, first), (1.5, "")]
    dense = [(0.2, ""), (0.3, changed), (0.4, changed), (1.2, changed), (1.3, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense), \
         pytest.raises(RuntimeError, match="lost a confirmed caption"):
        ocr.refine_caption_timing("video.mkv", coarse, 2)


def test_dense_anchor_tie_preserves_both_native_readings():
    correct = "A single fourth key will decide\nwho earns the supplies"
    noisy = correct + "i"
    coarse = [(0, ""), (0.5, noisy), (1, noisy), (1.5, "")]
    dense = [(0.4, ""), (0.5, noisy), (0.6, noisy), (0.7, correct), (0.8, correct), (0.9, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense):
        cues = ocr.refine_caption_timing("video.mkv", coarse, 2)
    assert [cue.text for cue in cues] == [noisy, correct]


def test_insufficient_dense_evidence_does_not_recreate_a_missing_coarse_reading():
    correct = "A single fourth key will decide\nwho earns the supplies"
    coarse = [(0, ""), (0.5, correct + "i"), (1, correct + "i"), (1.5, "")]
    dense = [(0.4, ""), (0.5, correct), (0.6, "")]
    with patch("subzero.ocr._scan_caption_frames", return_value=dense), \
         pytest.raises(RuntimeError, match="lost a confirmed caption"):
        ocr.refine_caption_timing("video.mkv", coarse, 2)
