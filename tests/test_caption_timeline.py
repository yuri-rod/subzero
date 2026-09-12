import pytest

from subzero import caption_timeline
from subzero.convert import Cue


def cue(start, end, text, style='Default'):
    return Cue(f'00:00:{start:06.3f}'.replace('.', ','),
               f'00:00:{end:06.3f}'.replace('.', ','), text, style)


def test_rome_and_me_too_remain_distinct_atoms_then_display_together():
    original = cue(28.133, 30.168, 'Me too. Me too.')
    recovered = cue(28.634, 30.035, 'I also like Rome.')
    atoms = caption_timeline.merge_caption_sources([original], [recovered])
    assert atoms == [original, recovered]
    assert atoms[0] is original
    timeline = caption_timeline.compose_caption_timeline(atoms)
    assert timeline == [cue(28.133, 28.634, original.text),
                        cue(28.634, 30.035, '- Me too. Me too.\n- I also like Rome.'),
                        cue(30.035, 30.168, original.text)]
    caption_timeline.validate_caption_timeline(atoms, timeline)


def test_whisper_overlapping_probst_keeps_both_voices():
    original = cue(35.605, 37.306, 'PROBST: No.')
    recovered = cue(36.306, 37.307, 'I gave it everything.')
    atoms = caption_timeline.merge_caption_sources([original], [recovered])
    assert atoms == [original, recovered]
    assert caption_timeline.compose_caption_timeline(atoms) == [
        cue(35.605, 36.306, 'PROBST: No.'),
        cue(36.306, 37.306, '- PROBST: No.\n- I gave it everything.'),
        cue(37.306, 37.307, 'I gave it everything.')]


def test_identical_ocr_is_removed_without_changing_original_format_or_timing():
    original = cue(1, 3, r'{\an8}<i>PROBST: I can\Nhelp Rome.</i>', 'Narrator')
    recovered = cue(1.2, 3.2, 'I can help Rome.')
    atoms = caption_timeline.merge_caption_sources([original], [recovered])
    assert atoms == [original]
    assert atoms[0] is original
    assert caption_timeline.compose_caption_timeline(atoms) == [original]


@pytest.mark.parametrize('first,second', [
    ('I can help Rome.', "I can't help Rome."),
    ('I also like Rome.', 'I also like Genevieve.'),
    ('Yes.', 'No.'),
    ('Four in six.', 'Four in five.'),
    ('I do not like Rome.', 'I like Rome.'),
    ('I do not\nlike Rome.', 'like Rome.'),
])
def test_distinct_words_names_numbers_and_negation_are_not_fuzzy_deduplicated(first, second):
    original, recovered = cue(1, 3, first), cue(1.1, 2.9, second)
    assert caption_timeline.merge_caption_sources([original], [recovered]) == [original, recovered]


def test_same_dialogue_far_away_is_not_treated_as_an_ocr_duplicate():
    original, recovered = cue(1, 2, 'Hello.'), cue(10, 11, 'Hello.')
    assert caption_timeline.merge_caption_sources([original], [recovered]) == [original, recovered]


def test_tiny_overlap_is_not_enough_to_deduplicate_a_repeated_utterance():
    original, recovered = cue(1, 2, 'No.'), cue(1.99, 3, 'No.')
    assert caption_timeline.merge_caption_sources([original], [recovered]) == [original, recovered]


def test_complete_ocr_sentence_already_in_consecutive_original_fragments_is_deduplicated():
    original = [cue(1, 2, 'GABE: I want to be'), cue(2.1, 4, 'the very first head.')]
    recovered = cue(1.2, 4.1, 'I want to be the very first head.')
    assert caption_timeline.merge_caption_sources(original, [recovered]) == original


def test_separated_original_dialogue_is_not_joined_to_erase_new_ocr():
    original = [cue(1, 2, 'I want'), cue(10, 11, 'to win.')]
    recovered = cue(1.1, 10.9, 'I want to win.')
    assert caption_timeline.merge_caption_sources(original, [recovered]) == [original[0], recovered, original[1]]


def test_original_multiple_speakers_can_cover_one_complete_ocr_voice():
    original = cue(1, 3, '- Oh, yes.\n- I gave it everything.')
    assert caption_timeline.merge_caption_sources([original], [cue(1.2, 2.8, 'I gave it everything.')]) == [original]


def test_repeated_overlapping_ocr_detections_do_not_create_extra_translation_atoms():
    first, repeated = cue(1, 3, 'I also like Rome.'), cue(2, 4, 'I also like Rome.')
    later = cue(5, 6, 'I also like Rome.')
    assert caption_timeline.merge_caption_sources([], [repeated, first, first, later]) == [
        cue(1, 4, 'I also like Rome.'), later]


def test_different_ocr_text_is_not_removed_as_a_stale_caption():
    recovered = [cue(1, 3, 'I like Rome.'), cue(2, 4, "I don't like Rome.")]
    assert caption_timeline.merge_caption_sources([], recovered) == recovered


def test_composition_preserves_inline_styles_and_existing_dialogue_dashes():
    first = cue(1, 3, '<i>Ana: Oi.</i>', 'Italic')
    second = cue(2, 4, '- Sim.\n- Vamos.')
    assert caption_timeline.compose_caption_timeline([second, first]) == [
        cue(1, 2, first.text, 'Italic'), cue(2, 3, '- <i>Ana: Oi.</i>\n- Sim.\n- Vamos.'), cue(3, 4, second.text)]


def test_composition_coalesces_identical_neighbors_but_keeps_real_gaps():
    atoms = [cue(1, 2, 'Hello.'), cue(2, 3, 'Hello.'), cue(4, 5, 'Hello.')]
    assert caption_timeline.compose_caption_timeline(atoms) == [cue(1, 3, 'Hello.'), cue(4, 5, 'Hello.')]


def test_two_speakers_saying_identical_words_keep_their_labels():
    atoms = [cue(1, 2, 'ANA: Yes.'), cue(1, 2, 'ROME: Yes.')]
    assert caption_timeline.compose_caption_timeline(atoms) == [cue(1, 2, '- ANA: Yes.\n- ROME: Yes.')]


def test_validation_rejects_lost_voice_changed_text_and_shortened_coverage():
    atoms = [cue(1, 3, 'Original.'), cue(2, 4, 'Recovered.')]
    valid = caption_timeline.compose_caption_timeline(atoms)
    for broken in ([valid[0], cue(2, 3, 'Original.'), valid[2]],
                   [*valid[:-1], cue(3, 3.9, 'Recovered.')],
                   [*valid, cue(2.5, 3.5, 'Additional.')]):
        with pytest.raises(ValueError, match='timeline'):
            caption_timeline.validate_caption_timeline(atoms, broken)


@pytest.mark.parametrize('bad', [cue(1, 1, 'Hello.'), cue(2, 1, 'Hello.'), cue(1, 2, '  '),
                                  Cue('bad', '00:00:02,000', 'Hello.')])
def test_invalid_cues_fail_before_source_merging_or_composition(bad):
    with pytest.raises(ValueError):
        caption_timeline.merge_caption_sources([bad], [])
    with pytest.raises(ValueError):
        caption_timeline.compose_caption_timeline([bad])


def test_validation_accepts_wrapping_and_display_markers_without_losing_word_order():
    atoms = [cue(1, 3, 'I also like Rome.'), cue(2, 3, 'Me too.')]
    timeline = [cue(1, 2, '- I also\nlike Rome.'),
                cue(2, 3, '- I also like\nRome.\n- Me too.')]
    caption_timeline.validate_caption_timeline(atoms, timeline)
    with pytest.raises(ValueError, match='timeline'):
        caption_timeline.validate_caption_timeline(atoms, [timeline[0], cue(2, 3, '- Me too.\n- I also like Rome.')])


def test_explicitly_different_speakers_are_not_deduplicated_in_either_source():
    first, second = cue(1, 3, 'GABE: Yes.'), cue(1.2, 2.8, 'SUE: Yes.')
    assert caption_timeline.merge_caption_sources([first], [second]) == [first, second]
    assert caption_timeline.merge_caption_sources([], [first, second]) == [first, second]


def test_original_fragments_from_different_speakers_do_not_hide_a_new_full_sentence():
    original = [cue(1, 2, 'GABE: I want'), cue(2.1, 3, 'SUE: to win.')]
    recovered = cue(1.2, 3.1, 'I want to win.')
    assert caption_timeline.merge_caption_sources(original, [recovered]) == [original[0], recovered, original[1]]


def test_composition_keeps_independent_voices_after_speaker_labels_are_removed():
    atoms = [cue(1, 3, 'Sim.'), cue(2, 4, 'Sim.')]
    assert caption_timeline.compose_caption_timeline(atoms) == [
        cue(1, 2, 'Sim.'), cue(2, 3, '- Sim.\n- Sim.'), cue(3, 4, 'Sim.')]
    with pytest.raises(ValueError, match='timeline'):
        caption_timeline.validate_caption_timeline(atoms, [cue(1, 4, 'Sim.')])


def test_matching_words_from_another_named_speaker_do_not_erase_ocr_voice():
    original = cue(1, 3, 'GABE: I agree.\nSUE: No.')
    recovered = cue(1.2, 2.8, 'SUE: I agree.')
    assert caption_timeline.merge_caption_sources([original], [recovered]) == [original, recovered]
