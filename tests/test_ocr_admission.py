import pytest

from subzero import ocr


def region(text, *, y=0.10, ink=0.20, x=0.25, width=0.5):
    return {"text": text, "confidence": 1, "x": x, "y": y, "width": width,
            "height": 0.06, "angle": 0, "captionInk": ink}


@pytest.mark.parametrize("text", ["PERSON OCCUPATION", "OUTDOORS", "Keep this secret."])
def test_caption_admission_rejects_colored_fill_without_using_the_words(text):
    assert ocr.caption_text({"items": [region(text, ink=0.02)]}) == ""


@pytest.mark.parametrize("text", ["KEEP THIS SECRET.", "Keep this secret.", '"Read this instruction."'])
def test_caption_admission_keeps_white_caption_fill_regardless_of_case(text):
    assert ocr.caption_text({"items": [region(text)]}) == text


def test_caption_admission_preserves_both_white_lines_and_censored_fragments():
    frame = {"items": [region("Keep this secret.", y=0.168), region("Tell nobody.")]}
    assert ocr.caption_text(frame) == "Keep this secret.\nTell nobody."
    off_center = {"items": [region("Find the", x=0.22, width=0.34)]}
    assert ocr.caption_text(off_center) == "Find the"


def test_caption_admission_vetoes_four_row_credit_layout_before_region_retry():
    frame = {"items": [region(f"Text row {index}", y=y) for index, y in enumerate((0.10, 0.16, 0.22, 0.28))]}
    assert ocr.caption_text(frame) == ""
    assert ocr._caption_title_card(frame)
    assert ocr.caption_retry_indices([frame, frame], [0, 0.5]) == []


def test_credit_layout_uses_full_frame_context_even_when_region_retry_has_two_rows():
    full = {"items": [region(f"Text row {index}", y=y) for index, y in enumerate((0.10, 0.16, 0.22, 0.28))]}
    cropped = {"items": full["items"][:2]}
    assert ocr.recover_caption_runs([full] * 3, {index: cropped for index in range(3)}, [0, 0.1, 0.2]) == [""] * 3


def test_caption_admission_does_not_count_colored_background_rows_as_credits():
    frame = {"items": [region("Keep this secret.")] +
             [region("Background text", y=y, ink=0.01) for y in (0.16, 0.22, 0.28)]}
    assert ocr.caption_text(frame) == "Keep this secret."


@pytest.mark.parametrize("ink", [float("nan"), float("inf"), -1, 1.1, "invalid", True, "0.20"])
def test_caption_admission_rejects_invalid_style_metadata(ink):
    with pytest.raises(RuntimeError, match="caption ink"):
        ocr.caption_text({"items": [region("Keep this secret.", ink=ink)]})


@pytest.mark.parametrize("entry", [None, {"captionInk": 0.5}, {"text": "Caption", "captionInk": 0.5}])
@pytest.mark.parametrize("read", [ocr.caption_text, ocr._caption_title_card])
def test_caption_admission_reports_malformed_regions_before_credit_check(entry, read):
    with pytest.raises(RuntimeError, match="Vision OCR returned.*text"):
        read({"items": [entry]})
