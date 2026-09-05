from subzero.worker.srt import Cue, chunks, dump, parse, timestamp

SAMPLE = """1
00:00:01,000 --> 00:00:03,500
Primeira linha
segunda linha

2
00:01:02,500 --> 00:01:04,000
Outra fala
"""


def test_parse_reads_every_cue():
    cues = parse(SAMPLE)
    assert len(cues) == 2
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[0].text == "Primeira linha\nsegunda linha"
    assert cues[1].start == 62.5


def test_parse_survives_bom_and_crlf():
    cues = parse("﻿1\r\n00:00:00,000 --> 00:00:01,000\r\nOi\r\n\r\n")
    assert len(cues) == 1
    assert cues[0].text == "Oi"


def test_parse_drops_empty_cues():
    cues = parse("1\n00:00:00,000 --> 00:00:01,000\n\n\n2\n00:00:02,000 --> 00:00:03,000\nOi\n")
    assert [c.text for c in cues] == ["Oi"]


def test_parse_accepts_a_dot_separator():
    assert parse("1\n00:00:01.250 --> 00:00:02.000\nOi\n")[0].start == 1.25


def test_timestamp_formats_srt_style():
    assert timestamp(3661.5) == "01:01:01,500"
    assert timestamp(0) == "00:00:00,000"


def test_round_trip():
    assert parse(dump(parse(SAMPLE))) == parse(SAMPLE)


def test_dump_renumbers_from_one():
    out = dump([Cue(9, 0, 1, "a"), Cue(4, 1, 2, "b")])
    assert out.startswith("1\n")
    assert "\n2\n" in out


def test_chunks_splits_by_count():
    cues = [Cue(i, i, i + 1, str(i)) for i in range(5)]
    assert [len(c) for c in chunks(cues, 2)] == [2, 2, 1]


def test_strip_hearing_impaired_removes_sound_and_speaker_marks():
    from subzero.worker.srt import Cue, strip_hearing_impaired

    cues = [Cue(1, 0, 1, "[DOOR CREAKS]"),
            Cue(2, 1, 2, "JOHN: Get down!"),
            Cue(3, 2, 3, "♪ upbeat music ♪"),
            Cue(4, 3, 4, "(sighs) I'm fine."),
            Cue(5, 4, 5, "- MARY: Run!\n- [GUNSHOT]")]

    out = strip_hearing_impaired(cues)

    # sobrando uma fala so, o travessao de dialogo nao serve para nada
    assert [c.text for c in out] == ["Get down!", "I'm fine.", "Run!"]
    # cue que era so descricao de som some e os indices sao refeitos
    assert [c.index for c in out] == [1, 2, 3]
    assert (out[0].start, out[0].end) == (1, 2)


def test_strip_hearing_impaired_leaves_normal_dialogue_alone():
    from subzero.worker.srt import Cue, strip_hearing_impaired

    cues = [Cue(1, 0, 1, "Vamos embora."), Cue(2, 1, 2, "- Onde?\n- Pra casa.")]

    assert [c.text for c in strip_hearing_impaired(cues)] == ["Vamos embora.", "- Onde?\n- Pra casa."]


def test_strip_hearing_impaired_keeps_a_colon_inside_a_sentence():
    """Nao pode comer 'Regra: ...' achando que e nome de quem fala."""
    from subzero.worker.srt import Cue, strip_hearing_impaired

    cues = [Cue(1, 0, 1, "Escuta: isso acaba mal.")]

    assert [c.text for c in strip_hearing_impaired(cues)] == ["Escuta: isso acaba mal."]


def test_strip_hearing_impaired_strips_html_and_ass_tags():
    from subzero.worker.srt import Cue, strip_hearing_impaired

    cues = [
        Cue(1, 0, 1, "<i>Texto em itálico</i>"),
        Cue(2, 1, 2, "<font color=\"#ffff00\">{\\an8}Fala no topo</font>"),
        Cue(3, 2, 3, "<b>[SOM ALTO]</b>\n<font color=\"red\">Atenção!</font>"),
    ]

    out = strip_hearing_impaired(cues)
    assert [c.text for c in out] == ["Texto em itálico", "Fala no topo", "Atenção!"]
