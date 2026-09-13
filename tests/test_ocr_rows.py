import pytest

from subzero import ocr


def region(text, *, x=0.25, y=0.10, width=0.5, height=0.065, candidates=()):
    return {"text": text, "confidence": 1, "x": x, "y": y, "width": width,
            "height": height, "angle": 0, "captionInk": 0.20,
            "candidates": [{"text": candidate, "confidence": 1} for candidate in candidates]}


def test_caption_fragments_follow_horizontal_order_within_the_same_row():
    frame = {"items": [region("Look at this", y=0.168),
                       region("an Immunity", x=0.348, width=0.196),
                       region("Idol...", x=0.542, y=0.1085, width=0.084, height=0.049)]}
    assert ocr.caption_text(frame) == "Look at this\nan Immunity Idol..."


def test_retry_restores_observed_full_row_over_exact_contained_fragments():
    complete = {"items": [region("an Immunity Idol...", x=0.348, width=0.306)]}
    split = {"items": [region("an Immunity", x=0.348, width=0.196),
                       region("Idol...", x=0.542, y=0.1085, width=0.084, height=0.049)]}
    frames = [complete, split, complete]
    assert ocr.recover_caption_runs(frames, dict(enumerate([complete] * 3)), [0, 0.1, 0.2]) == ["an Immunity Idol..."] * 3


def test_retry_uses_repeated_taller_native_row_without_dropping_the_first_line():
    first = region("Read these instructions", y=0.15, height=0.11)
    second = region("before you open the box.")
    complete = {"items": [first, second]}
    frames = [{"items": [second]}] * 3
    assert ocr.recover_caption_runs(frames, dict(enumerate([complete] * 3)), [0, 0.1, 0.2]) == [
        "Read these instructions\nbefore you open the box."] * 3


def test_retry_selects_repeated_native_reading_without_punctuation_attached_glyph():
    noisy = {"items": [region('"Without your basic tools...t')]}
    clean = {"items": [region('"Without your basic tools..."')]}
    assert ocr.recover_caption_runs([clean, noisy, clean], dict(enumerate([clean] * 3)), [0, 0.1, 0.2]) == [
        "Without your basic tools..."] * 3


def test_retry_removes_only_native_confirmed_non_ascii_prefix_before_a_capitalized_word():
    noisy = {"items": [region("ẠỌWithout your basic tools...")]}
    clean = {"items": [region("Without your basic tools...")]}
    assert ocr.recover_caption_runs([clean, noisy, clean], dict(enumerate([clean] * 3)), [0, 0.1, 0.2]) == [
        "Without your basic tools..."] * 3


@pytest.mark.parametrize(("first", "second"), [
    ("Émilie has the key.", "Emilie has the key."),
    ("ÉMILIE HAS THE KEY.", "MILIE HAS THE KEY."),
    ("ỌWithout your basic tools...", "With your basic tools..."),
    ("ỌWe need 7 votes to win.", "We need 8 votes to win."),
    ("ỌKyle has the key.", "Kyla has the key."),
    ("ỌWithout your basic tools...", "Without your basic supplies..."),
])
def test_native_prefix_recovery_preserves_names_negation_numbers_and_other_words(first, second):
    original = {"items": [region(first)]}
    retry = {"items": [region(second)]}
    assert ocr.recover_caption_runs([original] * 3, dict(enumerate([retry] * 3)), [0, 0.1, 0.2]) == [first] * 3


def test_native_prefix_recovery_requires_same_frame_geometry_and_repeated_crop_support():
    noisy = {"items": [region("ỌWithout your basic tools...")]}
    clean = {"items": [region("Without your basic tools...")]}
    assert ocr.recover_caption_runs([noisy] * 3, {1: clean}, [0, 0.1, 0.2]) == [
        "ỌWithout your basic tools..."] * 3
    shifted = {"items": [region("Without your basic tools...", y=0.168)]}
    assert ocr.recover_caption_runs([noisy] * 3, dict(enumerate([shifted] * 3)), [0, 0.1, 0.2]) == [
        "Without your basic tools...\nỌWithout your basic tools..."] * 3


@pytest.mark.parametrize(("first", "second"), [
    ("We can vote for Kyle.", "We can vote for Kyla."),
    ("We should vote for him.", "We should not vote for him."),
    ("We need 7 votes to win.", "We need 8 votes to win."),
    ("We need seven votes to win.", "We need eight votes to win."),
])
def test_native_candidate_keeps_names_negation_and_numbers(first, second):
    frame = {"items": [region(first, candidates=[second])]}
    neighbor = {"items": [region(second)]}
    assert ocr.caption_text(frame, previous=neighbor, following=neighbor) == first
    assert ocr.recover_caption_runs([neighbor, frame, neighbor], dict(enumerate([neighbor] * 3)),
                                    [0, 0.1, 0.2])[1] == first


@pytest.mark.parametrize(("fragment", "complete"), [
    ("Immunity Idol", "no Immunity Idol"),
    ("seven votes", "eight votes"),
    ("Kyle stays", "Kyla stays"),
])
def test_fragment_retry_does_not_add_negation_or_change_names_or_numbers(fragment, complete):
    original = {"items": [region(fragment, x=0.35, width=0.3)]}
    retried = {"items": [region(complete)]}
    readings = ocr.recover_caption_runs([original] * 3, dict(enumerate([retried] * 3)), [0, 0.1, 0.2])
    assert readings == [fragment] * 3


@pytest.mark.parametrize(("first", "second"), [
    ("We should leave. go", "We should leave."),
    ("We should leave. I", "We should leave."),
])
def test_punctuation_glyph_retry_does_not_delete_words(first, second):
    original = {"items": [region(first)]}
    retried = {"items": [region(second)]}
    readings = ocr.recover_caption_runs([original] * 3, dict(enumerate([retried] * 3)), [0, 0.1, 0.2])
    assert readings == [first] * 3


def test_split_retry_keeps_its_same_frame_censor_without_duplicating_the_full_row():
    original = {"items": [region("Look at that thing.", x=.33, width=.33)]}
    split = {"items": [region("Look at", x=.33, width=.16, y=.113, height=.049),
                       region("_ that thing.", x=.51, width=.15, y=.091, height=.086)]}

    readings = ocr.recover_caption_runs([original] * 3, {1: split}, [1, 1.1, 1.2])

    assert readings == ["Look at that thing.", "Look at _ that thing.", "Look at that thing."]
    cues = ocr.cluster_ocr_detections(list(zip([1, 1.1, 1.2], readings)), min_duration=0,
                                     sample_duration=.1, stabilize_text=False)
    with pytest.raises(RuntimeError, match="Unstable OCR"):
        ocr.validate_caption_readings(cues)
    assert ocr._caption_rescue_windows([], cues, [], 3) == [(0.25, 2.05)]


def test_split_retry_preserves_the_censor_on_every_same_frame_reading():
    original = {"items": [region("Look at that thing.", x=.33, width=.33)]}
    split = {"items": [region("Look at", x=.33, width=.16, y=.113, height=.049),
                       region("_ that thing.", x=.51, width=.15, y=.091, height=.086)]}

    assert ocr.recover_caption_runs([original] * 3, dict(enumerate([split] * 3)),
                                    [1, 1.1, 1.2]) == ["Look at _ that thing."] * 3


@pytest.mark.parametrize(("caption", "left", "right"), [
    ("Sam holds 7 keys.", "Pam holds", "_ 7 keys."),
    ("Sam holds 7 keys.", "Sam holds", "_ 8 keys."),
    ("We should not go.", "We should", "_ go."),
    ("Look at that thing.", "Look at", "_ that person."),
    ("Look _ at that thing.", "Look at", "_ that thing."),
])
def test_split_retry_does_not_change_words_or_move_an_existing_censor(caption, left, right):
    original = {"items": [region(caption, x=.33, width=.33)]}
    split = {"items": [region(left, x=.33, width=.16, y=.113, height=.049),
                       region(right, x=.51, width=.15, y=.091, height=.086)]}

    assert ocr.recover_caption_runs([original], {0: split}, [1]) == [caption]


@pytest.mark.parametrize("geometry", [
    {"x": .46, "width": .15, "y": .091, "height": .086},
    {"x": .51, "width": .15, "y": .168, "height": .065},
    {"x": .63, "width": .08, "y": .091, "height": .086},
])
def test_split_retry_censor_requires_nonoverlapping_fragments_of_one_full_row(geometry):
    original = {"items": [region("Look at that thing.", x=.33, width=.33)]}
    split = {"items": [region("Look at", x=.33, width=.16, y=.113, height=.049),
                       region("_ that thing.", **geometry)]}

    assert ocr.recover_caption_runs([original], {0: split}, [1]) == ["Look at that thing."]
