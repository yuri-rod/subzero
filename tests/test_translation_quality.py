import json

import pytest

from subzero.translation_quality import (
    SAMPLE_CUES,
    JudgeUnavailable,
    Verdict,
    check_translation,
    deterministic_hits,
    judge_pairs,
    parse_judgments,
    sample_indexes,
)


def test_flags_extreme_length_ratio():
    hits = deterministic_hits(
        ["Are you absolutely sure about this decision right now?"],
        ["Sim."],
    )
    assert 0 in hits


def test_ignores_normal_length_ratio():
    hits = deterministic_hits(
        ["Are you sure about this?"],
        ["Você tem certeza disso?"],
    )
    assert hits == {}


def test_ignores_short_cues_and_flags_empty_translation():
    assert deterministic_hits(["Yes!"], ["Sim, é isso mesmo!"]) == {}
    hits = deterministic_hits(["Where are we going tonight?"], [""])
    assert 0 in hits


def test_flags_mistranslated_game_terms():
    hits = deterministic_hits(
        ["He found an immunity idol on the beach."],
        ["Ele encontrou um prêmio na praia."],
    )
    assert 0 in hits
    hits = deterministic_hits(
        ["See you at tribal council tonight."],
        ["Vejo você na reunião hoje à noite."],
    )
    assert 0 in hits


def test_accepts_correct_game_terms():
    hits = deterministic_hits(
        ["He found an immunity idol.", "See you at tribal council."],
        ["Ele encontrou um ídolo de imunidade.", "Vejo você no conselho tribal."],
    )
    assert hits == {}


def test_flags_repeated_translation_for_distinct_dialogue():
    sources = [
        "The weather is beautiful this morning.",
        "I need to buy some groceries today.",
        "The car broke down on the highway.",
    ]
    hits = deterministic_hits(sources, ["Está tudo bem por aqui."] * 3)
    assert set(hits) == {0, 1, 2}


def test_ignores_repeated_short_acknowledgments():
    sources = ["Yes.", "Yeah.", "Yes, I agree."]
    hits = deterministic_hits(sources, ["Sim."] * 3)
    assert hits == {}


def test_sample_spans_file_and_keeps_suspects():
    sample = sample_indexes(100, [3, 77])
    assert len(sample) == SAMPLE_CUES
    assert 3 in sample and 77 in sample
    assert sample[0] < 10
    assert sample[-1] > 89
    assert sample == sorted(sample)
    assert sample_indexes(100, [3, 77]) == sample


def test_sample_caps_at_forty_with_many_suspects():
    sample = sample_indexes(100, list(range(60)))
    assert len(sample) == SAMPLE_CUES


def test_sample_returns_everything_when_short():
    assert sample_indexes(12, [4]) == list(range(12))


def good_judge(pairs, **kwargs):
    from subzero.translation_quality import Judgment
    return [Judgment(index, 5, False, "same meaning") for index, _, _ in pairs]


def bad_judge(pairs, **kwargs):
    from subzero.translation_quality import Judgment
    return [Judgment(index, 2, False, "wrong meaning") for index, _, _ in pairs]


def test_clean_translation_passes(monkeypatch):
    monkeypatch.setattr("subzero.translation_quality.judge_pairs", good_judge)
    verdict = check_translation(
        ["Are you sure about this?"], ["Você tem certeza disso?"],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert verdict.ok
    assert verdict.score == 5.0


def test_low_mean_score_needs_review(monkeypatch):
    monkeypatch.setattr("subzero.translation_quality.judge_pairs", bad_judge)
    verdict = check_translation(
        ["Are you sure?", "Where is he?"], ["Você tem certeza?", "Onde ele está?"],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert not verdict.ok
    assert verdict.score == 2.0
    assert "2.0" in verdict.reason


def test_two_criticals_need_review_even_with_high_mean(monkeypatch):
    from subzero.translation_quality import Judgment

    def critical_judge(pairs, **kwargs):
        out = []
        for n, (index, _source, _translated) in enumerate(pairs):
            out.append(Judgment(index, 5, n < 2, "critical" if n < 2 else "fine"))
        return out

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", critical_judge)
    sources = [f"Line number {n} here." for n in range(10)]
    verdict = check_translation(
        sources, [f"Linha número {n} aqui." for n in range(10)],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert not verdict.ok
    assert len(verdict.worst) <= 5


def test_judge_outage_degrades_to_deterministic_pass(monkeypatch):
    def down(_pairs, **_kwargs):
        raise JudgeUnavailable("ollama is down")

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", down)
    verdict = check_translation(
        ["Are you sure about this?"], ["Você tem certeza disso?"],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert isinstance(verdict, Verdict)
    assert verdict.ok
    assert verdict.score is None


def test_judge_outage_with_many_suspects_needs_review(monkeypatch):
    def down(_pairs, **_kwargs):
        raise JudgeUnavailable("ollama is down")

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", down)
    sources = ["Are you absolutely sure about this decision right now?"] * 6
    verdict = check_translation(
        sources, ["Sim."] * 6,
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert not verdict.ok
    assert verdict.score is None


def test_mismatched_cue_counts_raise():
    with pytest.raises(ValueError):
        check_translation(["one", "two"], ["um"],
                            url="http://127.0.0.1:11434", model="m")


def test_parse_judgments_accepts_valid_batch():
    body = {"response": json.dumps([
        {"id": 4, "adequacy": 5, "critical": False, "reason": "same meaning"},
        {"id": 9, "adequacy": 2, "critical": True, "reason": "negation dropped"},
    ])}
    judgments = parse_judgments([4, 9], body)
    assert [(j.cue, j.adequacy, j.critical) for j in judgments] == [(4, 5, False), (9, 2, True)]


@pytest.mark.parametrize("body", [
    {"response": "not json"},
    {"response": json.dumps([{"id": 4, "adequacy": 9, "critical": False, "reason": "x"}])},
    {"response": json.dumps([{"id": 4, "adequacy": 5, "critical": False, "reason": "x"}])},
    {"nope": True},
])
def test_parse_judgments_rejects_garbage(body):
    with pytest.raises(JudgeUnavailable):
        parse_judgments([4, 9], body)


def test_judge_pairs_batches_twenty_per_request(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size=-1):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        ids = [item["id"] for item in json.loads(payload["prompt"].rsplit("PAIRS:", 1)[1])]
        calls.append(len(ids))
        return FakeResponse({"response": json.dumps([
            {"id": i, "adequacy": 4, "critical": False, "reason": "ok"} for i in ids
        ])})

    monkeypatch.setattr("subzero.translation_quality.urllib.request.urlopen", fake_urlopen)
    pairs = [(n, f"source {n}", f"tradução {n}") for n in range(40)]
    judgments = judge_pairs(pairs, url="http://x", model="m")
    assert calls == [20, 20]
    assert len(judgments) == 40
