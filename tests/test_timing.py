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
