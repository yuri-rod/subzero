import random

import pytest

from subzero import timing


def dialogue(seed=8, duration=1800):
    rng = random.Random(seed)
    spans = []
    start = 10.0
    while start < duration - 10:
        end = start + rng.uniform(0.8, 3.8)
        spans.append((start, end))
        start = end + rng.uniform(0.4, 4.0)
    return spans


def test_accepts_aligned_dialogue_with_small_reading_delay():
    ref = dialogue()
    report = timing.evaluate([(a + .2, b + .3) for a, b in ref], ref)
    assert report.status == 'pass'


@pytest.mark.parametrize('offset,scale', [(3, 1), (-9, 1), (12, 24/23.976), (0, 25/24)])
def test_rejects_offset_and_framerate_drift(offset, scale):
    ref = dialogue()
    shifted = [(a * scale + offset, b * scale + offset) for a, b in ref]
    assert timing.evaluate(shifted, ref).status == 'reject'


def test_rejects_a_late_cut_change():
    ref = dialogue()
    shifted = [(a + (6 if a > 1500 else 0), b + (6 if a > 1500 else 0)) for a, b in ref]
    assert timing.evaluate(shifted, ref).status == 'reject'


def test_weak_sparse_or_unrelated_evidence_never_passes():
    assert timing.evaluate([], []).status == 'inconclusive'
    assert timing.evaluate([(1, 2)], [(1, 2)]).status == 'inconclusive'
    assert timing.evaluate(dialogue(2), dialogue(3)).status != 'pass'
    assert timing.evaluate([(0, 1800)], dialogue()).status != 'pass'


def test_truncated_subtitle_does_not_pass():
    ref = dialogue()
    assert timing.evaluate(ref[:len(ref)//2], ref).status == 'reject'


def test_repair_is_validated_on_windows_not_used_for_fitting():
    ref = dialogue()
    shifted = [(a * 1.001 + 8, b * 1.001 + 8) for a, b in ref]
    correction = timing.correction(timing.evaluate(shifted, ref))
    assert correction is not None
    scale, offset = correction
    assert timing.evaluate([(a*scale+offset,b*scale+offset) for a,b in shifted], ref).status == 'pass'


def test_constant_offset_repair_is_supported():
    ref = dialogue()
    # Simple constant delay of 3 seconds
    shifted = [(a + 3.0, b + 3.0) for a, b in ref]
    corr = timing.correction(timing.evaluate(shifted, ref))
    assert corr is not None
    scale, offset = corr
    assert abs(scale - 1.0) < 0.005
    assert abs(offset - (-3.0)) < 0.2
    assert timing.evaluate([(a*scale+offset, b*scale+offset) for a, b in shifted], ref).status == 'pass'



def test_irregular_cut_is_not_given_an_affine_repair():
    ref = dialogue()
    shifted = [(a + (8 if 600 < a < 1200 else 0), b + (8 if 600 < a < 1200 else 0)) for a,b in ref]
    assert timing.correction(timing.evaluate(shifted, ref)) is None


def test_invalid_timestamps_are_rejected():
    ref = dialogue()
    assert timing.evaluate([(float('nan'), 4)] + ref, ref).status == 'reject'


def test_repeated_identical_activity_is_ambiguous():
    ref=[(float(t),float(t+2)) for t in range(10,1800,4)]
    assert timing.evaluate(ref,ref).status == 'inconclusive'


def test_severe_offset_in_single_window_is_rejected():
    ref = dialogue()
    shifted = [(a + (15 if a >= 900 else 0), b + (15 if a >= 900 else 0)) for a, b in ref]
    report = timing.evaluate(shifted, ref)
    assert report.status == 'reject'
    assert 'Severe timing offset' in report.reason or 'mismatch' in report.reason


def test_piecewise_correction_detects_act_breaks_and_commercial_cuts():
    ref = dialogue(seed=42, duration=1800)
    # Segment 1 (0 to 600s): offset +3s
    # Segment 2 (600 to 1200s): offset +33s (30s commercial gap)
    # Segment 3 (1200 to 1800s): offset +63s (another 30s gap)
    shifted = []
    for a, b in ref:
        if a < 600:
            shifted.append((a + 3.0, b + 3.0))
        elif a < 1200:
            shifted.append((a + 33.0, b + 33.0))
        else:
            shifted.append((a + 63.0, b + 63.0))

    report = timing.evaluate(shifted, ref)
    assert report.status == 'reject'
    assert timing.correction(report) is None

    segments = timing.piecewise_correction(report)
    assert segments is not None
    assert len(segments) == 3
    assert abs(segments[0].offset - (-3.0)) < 0.3
    assert abs(segments[1].offset - (-33.0)) < 0.3
    assert abs(segments[2].offset - (-63.0)) < 0.3

    # Apply piecewise shift and verify
    repaired = []
    for a, b in shifted:
        chosen = segments[-1]
        for s in segments:
            if s.start <= a < s.end:
                chosen = s
                break
        repaired.append((a + chosen.offset, b + chosen.offset))

    assert timing.evaluate(repaired, ref).status == 'pass'


def test_piecewise_correction_rejects_sparse_or_inconsistent_segments():
    # Noise with single jump does not meet minimum windows per segment
    windows = [
        timing.Window(90.0, 1.0, 0.8, 0.2, True),
        timing.Window(180.0, 1.0, 0.8, 0.2, True),
        timing.Window(270.0, 25.0, 0.8, 0.2, True),  # isolated single window
        timing.Window(360.0, 1.0, 0.8, 0.2, True),
    ]
    report = timing.Report('reject', 'mismatch', windows=windows)
    assert timing.piecewise_correction(report) is None
