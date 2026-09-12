import pytest

from subzero.caption_quality import validate_caption_readings
from subzero.convert import Cue


def cue(start_ms, end_ms, text):
    def stamp(ms):
        seconds, millis = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}'
    return Cue(stamp(start_ms), stamp(end_ms), text)


@pytest.mark.parametrize('texts', [
    ['I saw this big rock.', '1 saw this big rock.', 'I saw this big rock.'],
    ["I'll come back at night.", "I'II come back at night.", "I'll come back at night."],
    ['have our four.', 'have our tour.', 'have our four.'],
    ['You will race on separate paths...', 'You will race on separate paths ii', 'You will race on separate paths...'],
])
def test_rejects_rapid_return_to_an_earlier_word_reading(texts):
    readings = [cue(10000, 10700, texts[0]), cue(10700, 10800, texts[1]), cue(10800, 11300, texts[2])]
    with pytest.raises(RuntimeError, match=r'OCR.*10\.700 seconds'):
        validate_caption_readings(readings)


def test_long_stable_readings_do_not_hide_a_brief_oscillation():
    readings = [cue(682165, 685168, 'Go on, Wonder Woman!'),
                cue(685168, 685268, 'Goron, Wonder Woman!'),
                cue(685268, 690000, 'Go on, Wonder Woman!')]
    with pytest.raises(RuntimeError, match=r'685\.168 seconds'):
        validate_caption_readings(readings)


def test_rejects_a_return_after_multiple_similar_readings():
    readings = [cue(10000, 10500, 'Bring the supplies to camp.'),
                cue(10500, 10600, 'Bring the supplles to camp.'),
                cue(10600, 10700, 'Bring the suppliesi to camp.'),
                cue(10700, 11300, 'Bring the supplies to camp.')]
    with pytest.raises(RuntimeError, match='OCR'):
        validate_caption_readings(readings)


def test_detects_instability_when_formatting_differs_without_mutating_cues():
    readings = [cue(10000, 10700, r'{\an8}<i>I saw this big rock.</i>'),
                cue(10700, 10800, '1 saw this big rock.'),
                cue(10800, 11300, 'I SAW THIS BIG ROCK!')]
    original = readings.copy()
    with pytest.raises(RuntimeError, match='OCR'):
        validate_caption_readings(readings)
    assert readings == original
    assert all(a is b for a, b in zip(readings, original))


@pytest.mark.parametrize('texts', [
    ['Yes.', 'No.', 'Go!'],
    ['Sam.', 'Pam.', 'Sam.'],
    ['Come here now.', 'Get the other rope.', 'We won the challenge!'],
    ['Bring seven bags.', 'Bring eight bags.', 'Bring nine bags.'],
    ['I see Sam.', 'I see Pam.', 'I see Tom.'],
    ['Take the key.', 'Leave the key.', 'Use the key.'],
    ['I can do this.', "I can't do this.", 'I will do this.'],
])
def test_preserves_short_and_genuinely_changing_dialogue(texts):
    readings = [cue(10000 + i * 150, 10150 + i * 150, text) for i, text in enumerate(texts)]
    original = readings.copy()
    assert validate_caption_readings(readings) is None
    assert readings == original


@pytest.mark.parametrize('texts', [
    ['Go now.', 'GO NOW!', '<i>Go now...</i>'],
    ["He didn't stop.", 'He didn t stop.', "He didn't stop!"],
    ['I found it.', 'I found it.', 'I found it.'],
])
def test_punctuation_case_and_spacing_do_not_count_as_lexical_changes(texts):
    validate_caption_readings([cue(10000 + i * 150, 10150 + i * 150, text) for i, text in enumerate(texts)])


@pytest.mark.parametrize('texts', [
    ['I found the key', 'I found the key\nunder the boat', 'I found the key\nunder the boat today.'],
    ['I found the key', 'I found the key\nunder the boat', 'under the boat'],
    ['We can', 'We can do this', 'We can do this together.'],
])
def test_preserves_progressive_one_and_two_line_captions(texts):
    validate_caption_readings([cue(10000 + i * 150, 10150 + i * 150, text) for i, text in enumerate(texts)])


def test_only_two_conflicting_readings_are_not_enough():
    validate_caption_readings([cue(10000, 10100, 'I saw this big rock.'),
                               cue(10100, 10600, '1 saw this big rock.')])


def test_readings_without_a_short_variant_are_not_rejected():
    validate_caption_readings([cue(10000, 10800, 'I saw this big rock.'),
                               cue(10800, 11600, '1 saw this big rock.'),
                               cue(11600, 12400, 'I saw this big rock.')])


def test_a_return_after_more_than_three_seconds_is_not_a_rapid_burst():
    validate_caption_readings([cue(10000, 10600, 'I saw this big rock.'),
                               cue(10600, 14100, '1 saw this big rock.'),
                               cue(14100, 14200, 'I saw this big rock.')])


def test_a_real_gap_breaks_the_cluster():
    validate_caption_readings([cue(10000, 10500, 'I saw this big rock.'),
                               cue(10800, 10900, '1 saw this big rock.'),
                               cue(10900, 11500, 'I saw this big rock.')])


def test_overlapping_voices_are_not_an_ocr_oscillation():
    validate_caption_readings([cue(10000, 10900, 'I saw this big rock.'),
                               cue(10700, 10800, '1 saw this big rock.'),
                               cue(10800, 11400, 'I saw this big rock.')])


def test_explicit_speaker_changes_break_the_cluster():
    validate_caption_readings([cue(10000, 10500, 'SAM: I saw this big rock.'),
                               cue(10500, 10600, 'PAM: 1 saw this big rock.'),
                               cue(10600, 11100, 'SAM: I saw this big rock.')])


def test_one_speaker_label_can_persist_over_an_unlabelled_continuation():
    with pytest.raises(RuntimeError, match='OCR'):
        validate_caption_readings([cue(10000, 10500, 'SAM: I saw this big rock.'),
                                   cue(10500, 10600, '1 saw this big rock.'),
                                   cue(10600, 11100, 'I saw this big rock.')])


def test_empty_and_single_short_caption_are_valid_for_this_gate():
    assert validate_caption_readings([]) is None
    assert validate_caption_readings([cue(10000, 10040, 'Run!')]) is None
