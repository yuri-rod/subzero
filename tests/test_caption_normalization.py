from unittest.mock import patch

import pytest

from subzero import ocr


@pytest.mark.parametrize('text, expected', [
    ("I'II do one.", "I'll do one."),
    ("We'Il do yoga.", "We'll do yoga."),
    ("they'II come.", "they'll come."),
    ("She’Il arrive.", "She’ll arrive."),
    ("And, like, didn t stop.", "And, like, didn't stop."),
    ("I don t know.", "I don't know."),
    ("They won t go.", "They won't go."),
    ("I didn\tt see it.", "I didn't see it."),
])
def test_normalizes_recognized_english_contraction_glyphs(text, expected):
    assert ocr._normalize_english_caption(text) == expected


@pytest.mark.parametrize('text', [
    "Don T and John T", "John t", "Émilie and ÉMILIE", "Don T. went to Rome.",
    "Henry III and Louis II", "7 and 8, I'11", "I II and We II", "We'III and I'Ill",
    "SHE'II and IT'II", "ÉI'II and I'IIs", "I love Vine!", "I don't know.",
    "green t-shirt", "I won t-shirt", "run t", "version t", "plan t",
    "can tangerines grow?", "The sign says DON T", "didn\nt", "This isn't mine.",
])
def test_normalization_preserves_names_numbers_other_words_and_line_boundaries(text):
    assert ocr._normalize_english_caption(text) == text


def test_normalization_reuses_original_scan_windows_and_joins_punctuation_flicker():
    full = "He didn't stop."
    coarse = [(0, ''), (0.5, full), (1, 'He didn t stop.'), (1.5, full), (2, '')]
    dense = [(0.3, ''), (0.4, full), (0.9, full), (1, 'He didn t stop.'), (1.1, full), (1.8, '')]
    windows = ocr.caption_transition_windows(coarse, 3)
    with patch('subzero.ocr._scan_caption_frames', return_value=dense) as scan:
        cues = ocr.refine_caption_timing('video.mkv', coarse, 3)
    assert scan.call_args.args[1] == windows
    assert [cue.text for cue in cues] == [full]
