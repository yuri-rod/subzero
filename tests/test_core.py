import pytest

from subzero import Options, analyze, fix_text, keep_breaks, rewrap, strip_sdh

OPTS = Options()


def cue(body, start="00:00:01,000", end="00:00:02,000"):
    return f"1\n{start} --> {end}\n{body}\n\n"


def one(body, opts=OPTS):
    """Text of the single cue that survives, with plain newlines."""
    out = fix_text(cue(body), opts).text.replace("\r\n", "\n")
    parts = out.strip().split("\n", 2)
    return parts[2] if len(parts) > 2 else ""


class TestSdhRemoval:
    @pytest.mark.parametrize("body", [
        "[tense music playing]", "(door creaks)", "♪♪",
        "[explosões]", "- [woman groans] - [laughter]",
    ])
    def test_pure_sdh_cue_is_dropped(self, body):
        assert fix_text(cue(body)).cues == 0

    def test_inline_sdh_is_removed_but_cue_survives(self):
        assert one("[sighs] I am fine.") == "I am fine."

    def test_keep_flags_disable_removal(self):
        opts = Options(strip_brackets=False)
        assert "[sighs]" in one("[sighs] I am fine.", opts)

    def test_dropped_cues_force_renumbering(self):
        src = cue("[music]") + "2\n00:00:03,000 --> 00:00:04,000\nReal line.\n\n"
        out = fix_text(src)
        assert out.cues == 1 and out.dropped == 1
        assert out.text.replace("\r\n", "\n").startswith("1\n00:00:03,000")


class TestLabels:
    @pytest.mark.parametrize("body,want", [
        ("PROBST: Come on in.", "Come on in."),
        ("MAN: Watch out!", "Watch out!"),
        ("man: watch out!", "watch out!"),
        ("homem: cuidado!", "cuidado!"),
        ("voz masculina: alguém aí?", "alguém aí?"),
        ("man 2: over here.", "over here."),
    ])
    def test_labels_are_stripped(self, body, want):
        assert one(body) == want

    @pytest.mark.parametrize("body", [
        "Primeiro voto: Jeff.",
        "Eu falei: não vou.",
        "Ele disse: pode ir.",
        "First vote: Jeff.",
        "I told him: no.",
    ])
    def test_dialogue_with_a_colon_is_left_alone(self, body):
        """The colon alone is not evidence: real dialogue uses it too."""
        assert one(body) == body

    def test_keep_labels_flag(self):
        assert one("PROBST: Come on in.", Options(strip_labels=False)) == "PROBST: Come on in."


class TestLineBreaks:
    def test_two_speakers_on_one_line_are_split(self):
        assert one("- MITCH: Yes, sir. - MAN: A question.") == "- Yes, sir.\n- A question."

    def test_dash_without_a_space_still_splits(self):
        assert one("Oh my. -PROBST: This is serious.") == "- Oh my.\n- This is serious."

    def test_long_line_is_broken_in_two(self):
        got = one("I really do not know what he was thinking when he went alone.")
        assert "\n" in got and all(len(l) <= 45 for l in got.split("\n"))

    def test_break_prefers_punctuation(self):
        got = one("He said it was over, and then he simply walked away from us.")
        assert got.split("\n")[0].endswith(",")

    def test_an_aside_dash_is_not_a_speaker_change(self):
        got = one("The plan - such as it was - failed.")
        assert not got.startswith("- ")

    def test_short_line_is_untouched(self):
        assert one("Yes.") == "Yes."


class TestPreserveBreaks:
    def test_existing_good_breaks_survive(self):
        body = "I do not know what to say\nabout any of this right now."
        assert one(body) == body

    def test_existing_breaks_are_rebuilt_when_a_line_is_too_long(self):
        body = "Yes.\nI really do not know what he was thinking when he went alone."
        got = one(body)
        assert got != body and all(len(l) <= 46 for l in got.split("\n"))

    def test_an_unbreakable_run_is_left_as_it_is(self):
        """Nothing to split on, so the line stays long rather than being mangled."""
        body = "short\n" + "x" * 60
        assert one(body) == body

    def test_lone_dash_on_a_single_line_cue_is_kept(self):
        """It marks an exchange carried over from the previous cue."""
        assert one("- Fine.") == "- Fine."

    def test_rewrap_all_reflows_a_badly_placed_break(self):
        body = "I do not know\nwhat to say about any of this right now."
        got = one(body, Options(preserve_breaks=False))
        assert got != body
        assert max(len(l) for l in got.split("\n")) < len(
            "what to say about any of this right now.")

    def test_preserve_keeps_that_same_bad_break(self):
        """Odd breaks are still someone's choice, so preserve mode respects them."""
        body = "I do not know\nwhat to say about any of this right now."
        assert one(body) == body


class TestOutput:
    def test_output_is_crlf(self):
        assert "\r\n" in fix_text(cue("Hello.")).text

    def test_bom_is_dropped(self):
        assert not fix_text("﻿" + cue("Hello.")).text.startswith("﻿")

    def test_idempotent(self):
        src = cue("- MITCH: Yes, sir. - MAN: A question.") + \
            "2\n00:00:05,000 --> 00:00:06,000\n[music]\n\n"
        once = fix_text(src).text
        assert fix_text(once).text == once == fix_text(fix_text(once).text).text

    def test_timing_lines_are_untouched(self):
        out = fix_text(cue("Hello.", "01:02:03,456", "01:02:04,789")).text
        assert "01:02:03,456 --> 01:02:04,789" in out

    def test_timing_lines_without_milliseconds_are_supported(self):
        out = fix_text(cue("Hello.", "01:02:03", "01:02:04")).text
        assert "01:02:03 --> 01:02:04" in out


class TestAnalyze:
    def test_counts_what_needs_work(self):
        src = cue("[music]") + "2\n00:00:03,000 --> 00:00:04,000\n- A. - B.\n\n"
        st = analyze(src)
        assert st.cues == 2 and st.sdh >= 1 and st.collapsed == 1

    def test_pct_is_safe_on_empty(self):
        assert analyze("").pct("sdh") == 0.0
